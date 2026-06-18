"""Unit tests for Metis quantization primitives (quant_impl.py).

Run with:
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 python \
        mindspeed_llm/features_manager/metis/test_quant_impl.py
"""
import math
import os
import sys
import importlib.util

# Ensure CPU-only execution so tests run without NPU/CUDA.
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')

# Import quant_impl directly from file to avoid mindspeed_llm/__init__.py
# which triggers torch_npu NPU initialization.
_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    'metis_quant_impl', os.path.join(_HERE, 'quant_impl.py'))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

FP4_LEVELS = _mod.FP4_LEVELS
apply_metis_quantization = _mod.apply_metis_quantization
compute_subspace = _mod.compute_subspace
fp4_quantize_blockwise = _mod.fp4_quantize_blockwise
fp8_quantize_blockwise = _mod.fp8_quantize_blockwise
metis_decompose_matrix = _mod.metis_decompose_matrix
metis_quantize_with_subspace = _mod.metis_quantize_with_subspace
project_to_subspace = _mod.project_to_subspace
quantize_blockwise = _mod.quantize_blockwise
randomized_svd = _mod.randomized_svd
spectrum_stats = _mod.spectrum_stats

import torch

# ---------------------------------------------------------------------------
# FP4 E2M1 blockwise quantization
# ---------------------------------------------------------------------------

def test_fp4_levels_count():
    """E2M1 has exactly 8 representable magnitudes (3 exp/mant bits)."""
    assert FP4_LEVELS.numel() == 8
    assert FP4_LEVELS[0].item() == 0.0
    assert FP4_LEVELS[-1].item() == 6.0


def test_fp4_preserves_shape():
    t = torch.randn(4, 17)  # non-multiple of block_size
    q = fp4_quantize_blockwise(t, block_size=16)
    assert q.shape == t.shape


def test_fp4_zero_tensor():
    t = torch.zeros(32)
    q = fp4_quantize_blockwise(t, block_size=16)
    assert torch.all(q == 0)


def test_fp4_snaps_to_levels():
    """After scaling, quantized values must be products of FP4 levels and scale."""
    t = torch.tensor([0.0, 0.3, 0.6, 1.0, 1.4, 1.9, 2.5, 3.5, 4.5, 5.9])
    q = fp4_quantize_blockwise(t, block_size=len(t))
    # max abs = 5.9 -> scale; normalized values snapped to FP4_LEVELS
    scale = t.abs().max().item()
    normalized = q / scale
    # Each normalized magnitude must be in FP4_LEVELS
    for v in normalized.abs().tolist():
        assert v in FP4_LEVELS.tolist(), f'{v} not an FP4 level'


def test_fp4_bounded_error():
    """FP4 quantization error should be bounded relative to tensor norm."""
    torch.manual_seed(0)
    t = torch.randn(128, 128)
    q = fp4_quantize_blockwise(t, block_size=16)
    rel_err = (t - q).norm().item() / t.norm().item()
    # FP4 E2M1 has only 8 levels; for standard normal data the relative
    # Frobenius error is typically < 0.8. We assert a generous upper bound.
    assert rel_err < 0.9, f'FP4 relative error too high: {rel_err}'


def test_fp4_sign_preservation():
    """Sign of non-zero elements should be preserved."""
    t = torch.randn(64)
    q = fp4_quantize_blockwise(t, block_size=16)
    # Where original is nonzero and quantized is nonzero, signs match.
    mask = (t != 0) & (q != 0)
    assert torch.all(t[mask].sign() == q[mask].sign())


# ---------------------------------------------------------------------------
# FP8 blockwise quantization
# ---------------------------------------------------------------------------

def test_fp8_preserves_shape():
    t = torch.randn(3, 20)
    q = fp8_quantize_blockwise(t, block_size=16)
    assert q.shape == t.shape


def test_fp8_lower_error_than_fp4():
    """FP8 (127 levels) should have lower error than FP4 (8 levels)."""
    torch.manual_seed(1)
    t = torch.randn(256, 256)
    q4 = fp4_quantize_blockwise(t, block_size=16)
    q8 = fp8_quantize_blockwise(t, block_size=16)
    err4 = (t - q4).norm().item() / t.norm().item()
    err8 = (t - q8).norm().item() / t.norm().item()
    assert err8 < err4, f'FP8 err {err8} should be < FP4 err {err4}'


def test_quantize_blockwise_dispatch():
    t = torch.randn(32)
    q4 = quantize_blockwise(t, qdtype='fp4', block_size=16)
    q8 = quantize_blockwise(t, qdtype='fp8', block_size=16)
    assert q4.shape == t.shape
    assert q8.shape == t.shape


# ---------------------------------------------------------------------------
# Randomized SVD
# ---------------------------------------------------------------------------

def test_randomized_svd_shapes():
    torch.manual_seed(2)
    m = torch.randn(50, 30)
    U, S, Vt = randomized_svd(m, rank=5, proj_dim=10)
    assert U.shape == (50, 5)
    assert S.shape == (5,)
    assert Vt.shape == (5, 30)


def test_randomized_svd_orthonormal_columns():
    torch.manual_seed(3)
    m = torch.randn(40, 40)
    U, _, _ = randomized_svd(m, rank=5, proj_dim=10)
    # U should have orthonormal columns: U^T U ≈ I
    utu = U.T @ U
    assert torch.allclose(utu, torch.eye(5), atol=1e-4)


def test_randomized_svd_approximates_top_singular_values():
    """Randomized SVD should recover top singular values approximately."""
    torch.manual_seed(4)
    # Low-rank matrix: rank 3
    A = torch.randn(100, 3)
    B = torch.randn(3, 80)
    m = A @ B
    _, S_true, _ = torch.linalg.svd(m, full_matrices=False)
    _, S_rand, _ = randomized_svd(m, rank=5, proj_dim=10)
    # Top 3 singular values should match closely
    assert torch.allclose(S_true[:3], S_rand[:3], atol=1e-3)


def test_randomized_svd_rank_clamped():
    """rank larger than min(m,n) should be clamped."""
    m = torch.randn(10, 8)
    U, S, Vt = randomized_svd(m, rank=100, proj_dim=200)
    assert U.shape == (10, 8)
    assert S.shape == (8,)


# ---------------------------------------------------------------------------
# Metis decomposition & subspace projection
# ---------------------------------------------------------------------------

def test_metis_decompose_residual_orthogonal_to_low_rank():
    """Residual should have small projection onto the dominant subspace."""
    torch.manual_seed(5)
    # Construct a matrix with clear low-rank structure
    A = torch.randn(64, 4)
    B = torch.randn(4, 64)
    m = A @ B + 0.01 * torch.randn(64, 64)
    low_rank, residual = metis_decompose_matrix(m, rank_frac=0.06, sample_ratio=1.0)
    # Residual norm should be much smaller than original
    assert residual.norm().item() < 0.5 * m.norm().item()


def test_project_to_subspace():
    torch.manual_seed(6)
    m = torch.randn(20, 10)
    U, _, _ = randomized_svd(m, rank=3, proj_dim=8)
    proj, resid = project_to_subspace(m, U)
    # proj + resid == m
    assert torch.allclose(proj + resid, m, atol=1e-4)
    # residual should be orthogonal to U columns
    ortho = U.T @ resid
    assert ortho.abs().max().item() < 1e-3


def test_compute_subspace_orthonormal():
    torch.manual_seed(7)
    m = torch.randn(50, 30)
    U = compute_subspace(m, rank_frac=0.1, sample_ratio=1.0)
    assert U.shape[1] == max(1, int(min(50, 30) * 0.1))
    utu = U.T @ U
    assert torch.allclose(utu, torch.eye(U.shape[1]), atol=1e-4)


def test_metis_quantize_with_subspace_matches_decompose():
    """Using a cached subspace should give similar result to full decompose."""
    torch.manual_seed(8)
    m = torch.randn(64, 64)
    U = compute_subspace(m, rank_frac=0.05, sample_ratio=1.0)
    q_with_sub = metis_quantize_with_subspace(m, U, qdtype='fp4', block_size=16)
    q_full = apply_metis_quantization(m, rank_frac=0.05, block_size=16, sample_ratio=1.0, qdtype='fp4')
    # Both should be close (not identical due to randomized sampling in decompose)
    rel_diff = (q_with_sub - q_full).norm().item() / max(q_full.norm().item(), 1e-8)
    assert rel_diff < 0.5


def test_apply_metis_quantization_shape():
    t = torch.randn(7, 11)
    q = apply_metis_quantization(t, rank_frac=0.1, block_size=16, sample_ratio=1.0)
    assert q.shape == t.shape


def test_apply_metis_quantization_fp4_vs_fp8():
    """FP8 Metis should have lower error than FP4 Metis."""
    torch.manual_seed(9)
    t = torch.randn(128, 128)
    q4 = apply_metis_quantization(t, rank_frac=0.05, block_size=16, sample_ratio=1.0, qdtype='fp4')
    q8 = apply_metis_quantization(t, rank_frac=0.05, block_size=16, sample_ratio=1.0, qdtype='fp8')
    err4 = (t - q4).norm().item() / t.norm().item()
    err8 = (t - q8).norm().item() / t.norm().item()
    assert err8 < err4


# ---------------------------------------------------------------------------
# Spectrum stats
# ---------------------------------------------------------------------------

def test_spectrum_stats_basic():
    t = torch.randn(20, 15)
    stats = spectrum_stats(t, topk=10)
    assert stats['shape'] == [20, 15]
    assert 0.0 <= stats['top1'] <= 1.0
    assert 0.0 <= stats['top5'] <= 1.0
    assert stats['energy_k'] >= 1
    assert len(stats['singular_values']) <= 10


def test_spectrum_stats_identity_matrix():
    """Identity matrix has uniform spectrum -> top1 ≈ 1/n."""
    t = torch.eye(10)
    stats = spectrum_stats(t)
    assert abs(stats['top1'] - 0.1) < 1e-6
    assert abs(stats['top5'] - 0.5) < 1e-6


def test_spectrum_stats_low_rank():
    """Rank-1 matrix should have top1 ≈ 1.0."""
    a = torch.randn(20, 1)
    b = torch.randn(1, 20)
    t = a @ b
    stats = spectrum_stats(t)
    assert stats['top1'] > 0.99


def test_spectrum_stats_non_2d_returns_empty():
    t = torch.randn(10)
    assert spectrum_stats(t) == {}


if __name__ == '__main__':
    # Allow running directly without pytest.
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    passed = 0
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f'PASS  {fn.__name__}')
            passed += 1
        except Exception as e:
            print(f'FAIL  {fn.__name__}: {type(e).__name__}: {e}')
            traceback.print_exc()
            failed += 1
    print(f'\n{passed} passed, {failed} failed')
    raise SystemExit(1 if failed else 0)
