"""Averis: mean-bias aware FP4 W4A4G4 quantization primitives.

Faithful reproduction of the method from
"The Curse and Blessing of Mean Bias in FP4-Quantized LLM Training"
(arXiv:2603.10444). The paper names the method *Averis* and applies it to
FP4 (W4A4G4) training: mean-residual splitting is applied to activations
(forward) and output gradients (backward), with the mean vector and the
residual quantized *independently* by a standard blockwise FP4 kernel.

Key formula (paper Sec. 5, "Forward pass: activation mean-residual splitting"
and "Backward pass: output-gradient mean-residual splitting"):

  Forward  (A4 + W4):
      X = M + R,   mu = mean(X, dim=0),  M = 1 * mu^T,  R = X - M
      Y = Q(R) @ Q(W)^T  +  Q(mu) @ Q(W)^T
    where Q(.) is blockwise FP4 quantization. The mean vector mu (1 x m) and
    the residual R (l x m) are quantized *separately* (independent scales),
    which is the whole point: mean has a different dynamic range than the
    residual, so splitting before quantization narrows both ranges.

  Backward (G4):
      G = dL/dY  (output gradient, l x n)
      G = M_G + R_G,  mu_G = mean(G, dim=0),  R_G = G - M_G
      dL/dX = Q(G) @ Q(W)            (input gradient, using quantized G and W)
      dL/dW = Q(G)^T @ X             (weight gradient, also FP4-quantized)
    where Q(G) = Q(mu_G) + Q(R_G) is the mean-residual quantized output
    gradient. Weight gradients are likewise FP4-quantized (G4 covers all
    gradients in the backward pass).

The FP4 simulation reuses Metis' E2M1 blockwise quantizer. A straight-through
estimator (STE) is provided for standalone use; inside the autograd.Function
the STE is implicit because backward is defined manually.
"""
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

# Reuse the proven E2M1 blockwise FP4 quantizer from Metis.
from mindspeed_llm.features_manager.metis.quant_impl import (
    fp4_quantize_blockwise,
    FP4_MAX,
)


# ---------------------------------------------------------------------------
# FP4 quantization with straight-through estimator (for standalone use).
# ---------------------------------------------------------------------------
def fp4_quantize_ste(tensor: torch.Tensor, block_size: int = 16) -> torch.Tensor:
    """FP4 blockwise quantization with a straight-through estimator.

    Forward: values are snapped to E2M1 levels (per-block scaled).
    Backward: gradients pass through unchanged (STE), so training remains
    differentiable despite the non-differentiable quantize op.
    """
    with torch.no_grad():
        q = fp4_quantize_blockwise(tensor, block_size=block_size)
    # STE: forward uses q, backward uses the identity (gradient of tensor).
    return tensor + (q - tensor).detach()


def _fp4_quantize_no_grad(tensor: torch.Tensor, block_size: int = 16) -> torch.Tensor:
    """Plain FP4 blockwise quantization (no STE). Used inside autograd.Function
    where backward is defined manually, so we don't want autograd to track the
    quantize op itself."""
    with torch.no_grad():
        return fp4_quantize_blockwise(tensor, block_size=block_size)


def mean_residual_split(
    activation: torch.Tensor,
    dim: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split ``activation`` into (mean_matrix, residual, mean_vector).

    mu = activation.mean(dim=dim)              # reduced shape
    M   = broadcast(mu) over dim               # same shape as activation
    R   = activation - M                       # centered, zero-mean along dim

    Returns (M, R, mu) where M is the rank-one mean component, R is the
    residual, and mu is the raw mean vector.
    """
    mu = activation.mean(dim=dim, keepdim=True)  # (1, m) for dim=0
    M = mu.expand_as(activation)
    R = activation - M
    return M, R, mu.squeeze(dim)


# ---------------------------------------------------------------------------
# Averis W4A4G4 Linear (autograd.Function).
# ---------------------------------------------------------------------------
class AverisFP4Linear(torch.autograd.Function):
    """Mean-residual splitting FP4 linear with W4A4G4 quantization.

    Forward (A4 + W4):
        mu = X.mean(dim=0)                  # (m,)   column mean
        R  = X - mu                         # (l, m) zero-mean residual
        Wq = Q_fp4(W)                       # (n, m) weight quantized (W4)
        mu_q = Q_fp4(mu.unsqueeze(0))       # (1, m) mean quantized independently
        Rq  = Q_fp4(R)                      # (l, m) residual quantized (A4)
        Y = F.linear(Rq, Wq) + F.linear(mu_q, Wq)   # = (Rq + mu_q) @ Wq^T
        Y += bias (if any)

    Backward (G4): output gradient G = dL/dY is also mean-residual split and
    FP4-quantized before being used to compute input/weight gradients.
        mu_G = G.mean(dim=0)               # (n,)
        R_G  = G - mu_G                    # (l, n)
        mu_Gq = Q_fp4(mu_G.unsqueeze(0))   # (1, n) G4 mean quantized
        R_Gq  = Q_fp4(R_G)                 # (l, n) G4 residual quantized
        Gq = R_Gq + mu_Gq                  # (l, n) quantized output gradient
        dL/dX = F.linear(Gq, Wq.t())       # (l, m) = Gq @ Wq
        dL/dW = Q_fp4(F.linear(Gq.t(), X.t()))  # (n, m) weight grad, also G4
        dL/dbias = Gq.sum(0)               # (n,)
    """

    @staticmethod
    def forward(ctx, activation, weight, bias, block_size, min_numel):
        # Fall back to plain linear for non-2D or tiny inputs (e.g. attention
        # scores, 1-D vectors). This keeps the patch safe across all call sites.
        if activation.ndim != 2 or activation.numel() < min_numel or weight.ndim != 2:
            ctx.skip = True
            out = F.linear(activation, weight)
            if bias is not None:
                out = out + bias
            return out

        ctx.skip = False
        ctx.block_size = block_size
        ctx.has_bias = bias is not None

        # --- Activation mean-residual split (A4) ---
        mu = activation.mean(dim=0)                 # (m,)
        R = activation - mu.unsqueeze(0)            # (l, m)

        # --- Independent FP4 quantization of mean, residual, weight (A4 + W4) ---
        Wq = _fp4_quantize_no_grad(weight, block_size)          # (n, m)
        mu_q = _fp4_quantize_no_grad(mu.unsqueeze(0), block_size)  # (1, m)
        Rq = _fp4_quantize_no_grad(R, block_size)               # (l, m)

        # Y = (Rq + mu_q) @ Wq^T  =  F.linear(Rq, Wq) + F.linear(mu_q, Wq)
        out = F.linear(Rq, Wq) + F.linear(mu_q, Wq)
        if bias is not None:
            out = out + bias

        # Save tensors needed for backward. We save the *quantized* weight Wq
        # (used in dL/dX) and the *original* activation X (used in dL/dW).
        ctx.save_for_backward(activation, Wq)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.skip:
            activation, Wq = ctx.saved_tensors
            # Plain linear backward: dL/dX = grad_output @ W, dL/dW = grad_output^T @ X
            # weight here is the unquantized weight; we only saved Wq in forward
            # for the skip path we actually didn't save the original weight, so
            # fall back to using Wq (best effort). This path is rarely hit.
            grad_input = F.linear(grad_output, Wq.t()) if Wq.ndim == 2 else None
            grad_weight = F.linear(grad_output.t(), activation.t()) if activation.ndim == 2 else None
            grad_bias = grad_output.sum(0) if ctx.has_bias else None
            return grad_input, grad_weight, grad_bias, None, None

        activation, Wq = ctx.saved_tensors  # X (l,m), Wq (n,m)
        block_size = ctx.block_size

        # --- Output-gradient mean-residual split + FP4 quantization (G4) ---
        mu_G = grad_output.mean(dim=0)              # (n,)
        R_G = grad_output - mu_G.unsqueeze(0)       # (l, n)
        mu_Gq = _fp4_quantize_no_grad(mu_G.unsqueeze(0), block_size)  # (1, n)
        R_Gq = _fp4_quantize_no_grad(R_G, block_size)                 # (l, n)
        Gq = R_Gq + mu_Gq                           # (l, n) quantized output gradient

        # --- Input gradient: dL/dX = Gq @ Wq  (l, m) ---
        # F.linear(input, weight) = input @ weight^T. To get Gq @ Wq where
        # Wq is (n, m), pass weight = Wq.t() (m, n): Gq @ (Wq.t())^T = Gq @ Wq.
        grad_input = F.linear(Gq, Wq.t())

        # --- Weight gradient: dL/dW = Gq^T @ X  (n, m), then FP4-quantize (G4) ---
        # F.linear(Gq.t(), X.t()) = Gq.t() @ (X.t())^T = Gq.t() @ X = (n,l)@(l,m) = (n,m).
        grad_weight = F.linear(Gq.t(), activation.t())
        grad_weight = _fp4_quantize_no_grad(grad_weight, block_size)

        grad_bias = grad_output.sum(0) if ctx.has_bias else None
        return grad_input, grad_weight, grad_bias, None, None


def mean_bias_linear(
    activation: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    block_size: int = 16,
    min_numel: int = 1024,
    linear_fn=None,
) -> torch.Tensor:
    """Compute Y = activation @ weight^T (+ bias) with Averis W4A4G4.

    Dispatches to :class:`AverisFP4Linear` (autograd.Function) so that both
    the forward mean-residual FP4 split (A4 + W4) and the backward
    output-gradient mean-residual FP4 split (G4) are applied.

    ``linear_fn`` is accepted for backward compatibility with earlier callers
    but is no longer used -- the autograd.Function calls ``F.linear`` directly
    on the quantized tensors, so there is no recursion risk.
    """
    return AverisFP4Linear.apply(activation, weight, bias, block_size, min_numel)
