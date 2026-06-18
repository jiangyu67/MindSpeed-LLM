"""Metis FP4/FP8 quantization primitives.

Implements:
  * FP4 E2M1 encode/decode (sign + 2 exp + 1 mantissa) with blockwise scaling.
  * FP8 (E4M3-like) blockwise quantize/dequantize.
  * Randomized SVD for spectral subspace extraction (Metis low-rank split).
  * Subspace projection helpers used by the optimizer hooks.

The FP4 path is a simulation: tensors stay in float32 but values are snapped to
the 8 representable E2M1 magnitudes (per-block scaled). When real FP4 hardware
is unavailable this gives a faithful numerical proxy of the quantization error.
"""
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

# E2M1 representable magnitudes (sign bit handled separately).
# Index 0..7 maps to the 4-bit code; 0.0 is shared with -0.0.
FP4_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
FP4_MAX = 6.0


def _quantize_to_e2m1(values: torch.Tensor) -> torch.Tensor:
    """Snap normalized magnitudes to the 8 E2M1 levels, preserving sign."""
    sign = torch.sign(values)
    abs_values = values.abs()
    levels = FP4_LEVELS.to(values.device, dtype=values.dtype)
    # bucketize gives the insertion index; clip to valid range.
    idx = torch.bucketize(abs_values, levels, right=True)
    idx = torch.clamp(idx, 0, levels.numel() - 1)
    quantized = levels[idx]
    return quantized * sign


def fp4_quantize_blockwise(tensor: torch.Tensor, block_size: int = 16) -> torch.Tensor:
    """Blockwise FP4 (E2M1) quantization simulation.

    Each block of ``block_size`` elements is scaled by its max abs value so the
    normalized magnitudes fall in [0, 1], then snapped to E2M1 levels.
    """
    original_shape = tensor.shape
    flat = tensor.reshape(-1)
    if flat.numel() == 0:
        return tensor.clone()
    block_size = max(1, block_size)
    n_blocks = math.ceil(flat.numel() / block_size)
    padded = F.pad(flat, (0, n_blocks * block_size - flat.numel()), value=0.0)
    blocks = padded.view(n_blocks, block_size)
    scales = blocks.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    normalized = blocks / scales
    quantized = _quantize_to_e2m1(normalized)
    restored = quantized * scales
    return restored.view(-1)[: flat.numel()].reshape(original_shape)


def fp8_quantize_blockwise(tensor: torch.Tensor, block_size: int = 16) -> torch.Tensor:
    """Blockwise FP8 (E4M3-like, 127 levels) quantization simulation."""
    original_shape = tensor.shape
    flat = tensor.reshape(-1)
    if flat.numel() == 0:
        return tensor.clone()
    block_size = max(1, block_size)
    n_blocks = math.ceil(flat.numel() / block_size)
    padded = F.pad(flat, (0, n_blocks * block_size - flat.numel()), value=0.0)
    blocks = padded.view(n_blocks, block_size)
    scales = blocks.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    normalized = blocks / scales
    quantized = torch.clamp((normalized * 127.0).round(), -127.0, 127.0)
    restored = quantized * (scales / 127.0)
    return restored.view(-1)[: flat.numel()].reshape(original_shape)


def quantize_blockwise(tensor: torch.Tensor, qdtype: str = 'fp4', block_size: int = 16) -> torch.Tensor:
    """Dispatch to fp4/fp8 blockwise quantizer."""
    if qdtype == 'fp8':
        return fp8_quantize_blockwise(tensor, block_size=block_size)
    return fp4_quantize_blockwise(tensor, block_size=block_size)


def randomized_svd(
    matrix: torch.Tensor,
    rank: int,
    proj_dim: Optional[int] = None,
    seed: Optional[int] = None,
    n_oversamples: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Randomized SVD via Halko et al. (random projection + small dense SVD).

    Returns (U[:, :rank], S[:rank], Vt[:rank, :]).
    """
    if seed is not None:
        torch.manual_seed(seed)
    if not matrix.is_floating_point() or matrix.dtype == torch.bfloat16:
        matrix = matrix.to(torch.float32)
    m, n = matrix.shape
    rank = max(1, min(rank, min(m, n)))
    if proj_dim is None:
        proj_dim = min(rank + n_oversamples, n)
    proj_dim = max(rank, min(proj_dim, n))
    omega = torch.randn(n, proj_dim, device=matrix.device, dtype=matrix.dtype)
    y = matrix @ omega
    q, _ = torch.linalg.qr(y, mode='reduced')
    b = q.T @ matrix
    u_tilde, s, vt = torch.linalg.svd(b, full_matrices=False)
    u = q @ u_tilde[:, :rank]
    return u[:, :rank], s[:rank], vt[:rank, :]


def metis_decompose_matrix(
    matrix: torch.Tensor,
    rank_frac: float = 0.015,
    sample_ratio: float = 0.01,
    proj_dim: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split ``matrix`` into (low_rank, residual) via sampled randomized SVD.

    The dominant spectral subspace is estimated from a row subset (Metis
    sparsely-random-sampling property). The right singular vectors V (shape
    (rank, n)) span the dominant column space and are used to project the
    *full* matrix: low_rank = (matrix @ V^T) @ V, which has the same shape as
    ``matrix``. The residual carries the remaining spectral energy.
    """
    m, n = matrix.shape
    rank = max(1, int(min(m, n) * rank_frac))
    if sample_ratio < 1.0:
        sample_rows = max(1, int(m * sample_ratio))
        idx = torch.randperm(m, device=matrix.device)[:sample_rows]
        sample = matrix[idx]
    else:
        sample = matrix
    if proj_dim is None:
        proj_dim = max(rank + 5, min(n, 256))
    # randomized_svd returns (U_sample, S, Vt) where Vt spans the dominant
    # right-singular subspace — valid for the full matrix (Metis property).
    _, s, vt = randomized_svd(sample, rank=rank, proj_dim=proj_dim)
    # Project full matrix onto the dominant right-singular subspace.
    # low_rank = matrix @ V^T @ diag(1/s) @ diag(s) @ V = matrix @ V^T @ V
    v = vt.T  # (n, rank)
    low_rank = (matrix @ v) @ v.T
    residual = matrix - low_rank
    return low_rank, residual


def apply_metis_quantization(
    matrix: torch.Tensor,
    rank_frac: float = 0.015,
    block_size: int = 16,
    sample_ratio: float = 0.01,
    qdtype: str = 'fp4',
) -> torch.Tensor:
    """Full Metis pipeline: spectral split + blockwise quantize both parts."""
    low_rank, residual = metis_decompose_matrix(
        matrix, rank_frac=rank_frac, sample_ratio=sample_ratio
    )
    q_low_rank = quantize_blockwise(low_rank, qdtype=qdtype, block_size=block_size)
    q_residual = quantize_blockwise(residual, qdtype=qdtype, block_size=block_size)
    return q_low_rank + q_residual


def project_to_subspace(matrix: torch.Tensor, U: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project ``matrix`` onto the column space of ``U``.

    Returns (projection, residual) where projection = U @ (U^T @ matrix).
    ``U`` must be orthonormal (columns) — as produced by randomized_svd.
    """
    if U.dtype != matrix.dtype:
        U = U.to(matrix.dtype)
    if U.device != matrix.device:
        U = U.to(matrix.device)
    proj = U @ (U.T @ matrix)
    return proj, matrix - proj


def metis_quantize_with_subspace(
    matrix: torch.Tensor,
    U: torch.Tensor,
    qdtype: str = 'fp4',
    block_size: int = 16,
) -> torch.Tensor:
    """Quantize using a precomputed dominant subspace (avoids re-running SVD).

    Splits ``matrix`` via ``U`` into (low_rank, residual) and quantizes each
    blockwise. Used by the optimizer hook when a cached subspace is available.
    """
    proj, resid = project_to_subspace(matrix, U)
    q_proj = quantize_blockwise(proj, qdtype=qdtype, block_size=block_size)
    q_resid = quantize_blockwise(resid, qdtype=qdtype, block_size=block_size)
    return q_proj + q_resid


def compute_subspace(
    matrix: torch.Tensor,
    rank_frac: float = 0.015,
    sample_ratio: float = 0.01,
    proj_dim: Optional[int] = None,
) -> torch.Tensor:
    """Compute and return the dominant left-singular subspace U (orthonormal).

    Only ``U`` is returned (not S/Vt) since the optimizer hook only needs the
    projection basis. The subspace is computed on CPU float32 for stability.
    """
    m, n = matrix.shape
    rank = max(1, int(min(m, n) * rank_frac))
    if sample_ratio < 1.0:
        sample_rows = max(1, int(m * sample_ratio))
        idx = torch.randperm(m, device=matrix.device)[:sample_rows]
        sample = matrix[idx]
    else:
        sample = matrix
    if proj_dim is None:
        proj_dim = max(rank + 5, min(n, 256))
    u, _, _ = randomized_svd(sample.to(torch.float32), rank=rank, proj_dim=proj_dim)
    return u


def spectrum_stats(matrix: torch.Tensor, topk: int = 32) -> Dict:
    """Compact spectral summary used for logging.

    For large matrices the full SVD is prohibitively expensive, so we use the
    randomized SVD path to extract only the top-``topk`` singular values.
    """
    if matrix.ndim != 2:
        return {}
    m = matrix.to(torch.float32)
    rows, cols = m.shape
    # Use randomized SVD for large matrices to avoid O(min(m,n)*m*n) cost.
    if min(rows, cols) > 256:
        try:
            _, sval, _ = randomized_svd(m, rank=min(topk, min(rows, cols)),
                                        proj_dim=min(topk + 10, cols))
        except Exception:
            sval = torch.linalg.svdvals(m)
    else:
        sval = torch.linalg.svdvals(m)
    energy = sval.pow(2)
    total = energy.sum().item() if energy.numel() > 0 else 1.0
    top1 = (energy[0].item() / total) if energy.numel() > 0 else 0.0
    top5 = (energy[:5].sum().item() / total) if energy.numel() >= 5 else (energy.sum().item() / total)
    cum = energy.cumsum(dim=0)
    energy_k = int((cum < total * 0.9).sum().item() + 1) if energy.numel() > 0 else 0
    return {
        'shape': list(matrix.shape),
        'top1': top1,
        'top5': top5,
        'energy_k': energy_k,
        'singular_values': sval.cpu().tolist()[: min(topk, sval.numel())],
    }
