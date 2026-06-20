"""Offline verification of mean-bias phenomenon on real Qwen2.5-7B activations.

Validates the core claims of arXiv:2603.10444 ("The Curse and Blessing of Mean
Bias in FP4-Quantized LLM Training") on real model activations:

  1. Mean-bias energy share: how much of activation Frobenius energy sits in
     the rank-one column-mean component  ||mu||^2 / ||X||_F^2.
  2. Dynamic-range shrinkage: |X|_inf  vs  |X - mu|_inf  (the L_inf norm that
     governs blockwise FP4 scale).
  3. FP4 quantization error: direct quantize(X)  vs  mean-residual
     quantize(X - mu) + mean path. Mean-subtraction should reduce error,
     especially in layers where mean-bias dominates.

Runs on the HF Qwen2.5-7B model (fully pretrained -> mean-bias most pronounced,
per the paper's "mean-bias grows with training" finding). Self-contained: only
needs torch + transformers, no mindspeed_llm / NPU deps.

Usage (on a GPU/NPU machine with the HF model):
    python verify_mean_bias.py
    # or point to a different model:
    python verify_mean_bias.py --model /path/to/qwen-hf --device cuda
"""
import argparse
import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# FP4 (E2M1) blockwise quantization simulation (inlined from Metis).
# ---------------------------------------------------------------------------
FP4_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def fp4_quantize_blockwise(tensor: torch.Tensor, block_size: int = 16) -> torch.Tensor:
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
    levels = FP4_LEVELS.to(tensor.device, dtype=tensor.dtype)
    idx = torch.bucketize(normalized.abs(), levels, right=True)
    idx = torch.clamp(idx, 0, levels.numel() - 1)
    quantized = levels[idx] * torch.sign(normalized)
    restored = quantized * scales
    return restored.view(-1)[: flat.numel()].reshape(original_shape)


# ---------------------------------------------------------------------------
# Mean-bias metrics for a single 2-D activation (l, m).
# ---------------------------------------------------------------------------
def analyze_activation(X: torch.Tensor, block_size: int = 16):
    X = X.detach().to(torch.float32)
    if X.ndim != 2:
        return None
    l, m = X.shape
    mu = X.mean(dim=0)                 # (m,)
    R = X - mu.unsqueeze(0)            # (l, m)

    # Energy shares (orthogonal decomposition: mean is rank-one, residual is
    # orthogonal to all-ones). ||X||_F^2 = ||mu||^2 * l + ||R||_F^2.
    mean_energy = (mu.norm() ** 2 * l).item()
    total_energy = (X.norm() ** 2).item()
    mean_share = mean_energy / max(total_energy, 1e-12)

    # Dynamic range (L_inf governs blockwise quant scale).
    x_inf = X.abs().amax().item()
    r_inf = R.abs().amax().item()

    # FP4 quantization error: direct vs mean-residual.
    q_direct = fp4_quantize_blockwise(X, block_size)
    err_direct = (q_direct - X).norm().item()

    q_resid = fp4_quantize_blockwise(R, block_size)
    # mean path kept high precision: reconstruct = q_resid + mu
    recon_mb = q_resid + mu.unsqueeze(0)
    err_mb = (recon_mb - X).norm().item()

    return {
        'shape': (l, m),
        'mean_share': mean_share,
        'x_inf': x_inf,
        'r_inf': r_inf,
        'inf_shrink': r_inf / max(x_inf, 1e-12),
        'err_direct': err_direct,
        'err_mean_residual': err_mb,
        'err_improvement': err_direct / max(err_mb, 1e-12),
    }


# ---------------------------------------------------------------------------
# Forward-hook capture of nn.Linear inputs.
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='/home/zs/model_from_hf/qwen2.5-7b-hf/',
                    help='HF model path')
    ap.add_argument('--device', default='cuda',
                    help='torch device (cuda / npu / cpu)')
    ap.add_argument('--block-size', type=int, default=16)
    ap.add_argument('--max-new-tokens', type=int, default=0,
                    help='forward-only (0) just runs the encoder once')
    ap.add_argument('--prompt', default='The meaning of life is')
    ap.add_argument('--out', default='/home/zs/mean_bias/verify_report.txt')
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f'Loading model from {args.model} ...')
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(args.device).eval()

    # Register forward hooks on every nn.Linear to capture its input activation.
    captured = {}

    def make_hook(name):
        def hook(module, inp, out):
            x = inp[0]
            if x.ndim >= 2:
                # flatten leading dims -> (l, m)
                x2d = x.reshape(-1, x.shape[-1])
                captured[name] = x2d.detach().to('cpu')
        return hook

    handles = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            handles.append(mod.register_forward_hook(make_hook(name)))

    # Run one forward pass.
    ids = tok(args.prompt, return_tensors='pt').input_ids.to(args.device)
    with torch.no_grad():
        model(ids)

    for h in handles:
        h.remove()

    print(f'Captured {len(captured)} linear-layer activations. Analyzing ...')

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    lines = []
    lines.append(f'Mean-bias verification on {args.model}')
    lines.append(f'prompt: {args.prompt!r}  block_size={args.block_size}')
    lines.append('=' * 110)
    hdr = (f'{"layer":<52} {"shape":>14} {"mean%":>7} {"|X|inf":>9} '
           f'{"|R|inf":>9} {"shrink":>7} {"err_direct":>11} {"err_mb":>11} {"improve":>8}')
    lines.append(hdr)
    lines.append('-' * 110)

    # Sort: deep layers / output first (where mean-bias is strongest).
    items = sorted(captured.items(), key=lambda kv: kv[0])
    total_improve = []
    mean_shares = []
    for name, x2d in items:
        st = analyze_activation(x2d, block_size=args.block_size)
        if st is None:
            continue
        mean_shares.append(st['mean_share'])
        total_improve.append(st['err_improvement'])
        lines.append(
            f'{name:<52} {str(st["shape"]):>14} {st["mean_share"]*100:>6.1f}% '
            f'{st["x_inf"]:>9.3f} {st["r_inf"]:>9.3f} {st["inf_shrink"]:>7.3f} '
            f'{st["err_direct"]:>11.2f} {st["err_mean_residual"]:>11.2f} '
            f'{st["err_improvement"]:>7.2f}x'
        )

    lines.append('-' * 110)
    if mean_shares:
        lines.append(f'avg mean-energy share : {sum(mean_shares)/len(mean_shares)*100:.1f}%')
        lines.append(f'avg FP4 error improve : {sum(total_improve)/len(total_improve):.2f}x')
        lines.append(f'max FP4 error improve : {max(total_improve):.2f}x')
    lines.append('=' * 110)
    lines.append('Interpretation:')
    lines.append('  mean%   = rank-one column-mean energy share (paper: grows with depth/training)')
    lines.append('  shrink  = |R|_inf / |X|_inf  (<1 means mean-subtraction shrinks dynamic range)')
    lines.append('  improve = direct-FP4 error / mean-residual-FP4 error (>1 means method helps)')

    report = '\n'.join(lines)
    print(report)
    with open(args.out, 'w') as fh:
        fh.write(report + '\n')
    print(f'\nReport saved to {args.out}')


if __name__ == '__main__':
    main()
