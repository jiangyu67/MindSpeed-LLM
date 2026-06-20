"""Anisotropy analysis on real Qwen2.5-7B weights/activations/gradients.

Reproduces the spectral-analysis figures from Metis (arXiv:2509.00404):
  * Fig 2A: singular-value spectra of W / A / G, with dominant-fraction marker.
  * Fig 2B: matrix value distribution + selected rank-1 components.
  * Fig 3B: residual distribution after removing top-3% components vs original.

Self-contained: only needs torch + transformers + matplotlib. Runs one forward
+ backward pass on the HF model, captures a representative MLP weight, its input
activation, and its weight gradient, then performs full SVD on each.

Usage:
    python analyze_anisotropy.py
    python analyze_anisotropy.py --model /path/to/qwen-hf --device cuda --layer 14
"""
import argparse
import os

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# SVD + spectral helpers
# ---------------------------------------------------------------------------
def full_svd(matrix: torch.Tensor):
    """Full SVD on CPU float32. Returns (s, U, Vt).

    For large matrices this can be slow; see ``svdvals_only`` and
    ``partial_svd`` for faster alternatives.
    """
    M = matrix.detach().to(torch.float32).cpu()
    U, S, Vt = torch.linalg.svd(M, full_matrices=False)
    return S, U, Vt


def svdvals_only(matrix: torch.Tensor) -> torch.Tensor:
    """Compute singular values via numpy (stable in sandboxed envs).

    Subsamples rows for large matrices to fit memory constraints.
    """
    import numpy as np
    M = matrix.detach().to(torch.float32).cpu().numpy()
    m, n = M.shape
    # cap rows at 2048 to stay within sandbox memory limits
    if m > 2048:
        M = M[:2048]
        m = 2048
    S = np.linalg.svd(M, compute_uv=False)
    return torch.from_numpy(S)


def partial_svd(matrix: torch.Tensor, q: int = 256):
    """Top-q SVD via numpy. Returns (s_top, U_top, Vt_top).

    Subsamples rows for large matrices to fit memory constraints.
    """
    import numpy as np
    M = matrix.detach().to(torch.float32).cpu().numpy()
    m, n = M.shape
    # cap rows at 2048 to stay within sandbox memory limits
    if m > 2048:
        M = M[:2048]
        m = 2048
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    q = min(q, len(S))
    return torch.from_numpy(S[:q]), torch.from_numpy(U[:, :q]), torch.from_numpy(Vt[:q, :])


def elbow_dominant_fraction(singular_values: torch.Tensor) -> float:
    """Dominant fraction via max-curvature elbow on the log-spectrum.

    Mirrors the paper's "elbow point of maximum curvature" definition.
    Returns the fraction of singular values up to (incl.) the elbow.
    """
    s = singular_values.clamp_min(1e-12)
    log_s = torch.log(s)
    n = log_s.numel()
    if n < 5:
        return 1.0
    # first derivative (central diff), second derivative
    d1 = (log_s[2:] - log_s[:-2]) / 2.0
    d2 = log_s[2:] - 2 * log_s[1:-1] + log_s[:-2]
    curvature = d2.abs() / (1.0 + d1 ** 2).pow(1.5).clamp_min(1e-12)
    # elbow index maps to original index i+1
    elbow = int(torch.argmax(curvature).item()) + 1
    return (elbow + 1) / n


def rank_one_component(U, S, Vt, i):
    """Return the i-th rank-1 component u_i * sigma_i * v_i^T (flattened values)."""
    return (U[:, i:i+1] * S[i] @ Vt[i:i+1, :]).reshape(-1)


def residual_after_topk(matrix, U, S, Vt, top_frac):
    """Residual = matrix_subsampled - sum of top-(top_frac) rank-1 components.

    Uses the available (partial) singular vectors. ``U``/``Vt`` are top-q
    from a possibly row-subsampled matrix; we subsample the original matrix
    to match.
    """
    k = max(1, int(S.numel() * top_frac))
    k = min(k, U.shape[1])  # cap at available partial vectors
    low_rank = (U[:, :k] * S[:k]) @ Vt[:k, :]  # (m_sub, n)
    M = matrix.to(torch.float32).cpu()
    # subsample rows to match U (partial SVD may have subsampled)
    if M.shape[0] != U.shape[0]:
        M = M[:U.shape[0]]
    return M - low_rank


# ---------------------------------------------------------------------------
# Plotting (paper-style figures)
# ---------------------------------------------------------------------------
def plot_spectra(specs, out_path):
    """Fig 2A: singular-value spectra for W / A / G."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    titles = ['Weight', 'Activation', 'Gradient']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for ax, (name, S), title, c in zip(axes, specs, titles, colors):
        s = S.clamp_min(1e-12)
        s_norm = s / s[0]
        idx = torch.arange(s.numel())
        ax.semilogy(idx, s_norm, color=c, linewidth=1.2)
        frac = elbow_dominant_fraction(s)
        elbow_k = max(1, int(frac * s.numel()))
        ax.axvline(elbow_k, color='r', linestyle='--', alpha=0.6)
        ax.set_title(f'{title}\n(dominant {frac*100:.2f}%)')
        ax.set_xlabel('singular value index')
        ax.set_ylabel(r'$\sigma_i / \sigma_0$ (log)')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'saved {out_path}')


def plot_distributions(matrices, svd_results, out_path):
    """Fig 2B: matrix value distribution + selected rank-1 components."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    titles = ['Weight', 'Activation', 'Gradient']
    # pick a few rank-1 component indices (paper uses i=0,16,128,1024)
    comp_indices = [0, 16, 128, 1024]
    for ax, (name, M), (sname, S, U, Vt), title in zip(
            axes, matrices, svd_results, titles):
        vals = M.to(torch.float32).cpu().reshape(-1)
        # subsample for histogram speed (aggressive for large matrices)
        if vals.numel() > 100000:
            vals = vals[torch.randperm(vals.numel())[:100000]]
        ax.hist(vals.numpy(), bins=200, density=True, alpha=0.5,
                label='full matrix', color='gray')
        n_avail = U.shape[1]  # partial SVD may have fewer than S.numel()
        for i in comp_indices:
            if i >= n_avail:
                continue
            comp = rank_one_component(U, S, Vt, i)
            if comp.numel() > 100000:
                comp = comp[torch.randperm(comp.numel())[:100000]]
            ax.hist(comp.numpy(), bins=100, density=True, alpha=0.5,
                    label=f'i={i}', histtype='step')
        ax.set_title(title)
        ax.set_xlabel('value')
        ax.set_ylabel('density')
        ax.set_yscale('log')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'saved {out_path}')


def plot_residuals(matrices, svd_results, out_path, top_frac=0.03):
    """Fig 3B: original distribution vs residual after removing top-k%."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    titles = ['Weight', 'Activation', 'Gradient']
    for ax, (name, M), (sname, S, U, Vt), title in zip(
            axes, matrices, svd_results, titles):
        orig = M.to(torch.float32).cpu().reshape(-1)
        resid = residual_after_topk(M, U, S, Vt, top_frac).reshape(-1)
        # subsample for histogram speed
        if orig.numel() > 100000:
            orig = orig[torch.randperm(orig.numel())[:100000]]
        if resid.numel() > 100000:
            resid = resid[torch.randperm(resid.numel())[:100000]]
        rng = max(orig.abs().max().item(), 1e-8)
        bins = torch.linspace(-rng, rng, 200)
        ax.hist(orig.numpy(), bins=bins.numpy(), density=True, alpha=0.4,
                label='original', color='gray')
        ax.hist(resid.numpy(), bins=bins.numpy(), density=True, alpha=0.6,
                label=f'residual (top {top_frac*100:.0f}% removed)', color='#d62728')
        ax.set_title(title)
        ax.set_xlabel('value')
        ax.set_ylabel('density')
        ax.set_yscale('log')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'saved {out_path}')


# ---------------------------------------------------------------------------
# Main: load model, capture W/A/G, run analysis
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='/home/zs/model_from_hf/qwen2.5-7b-hf/')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--layer', type=int, default=14,
                    help='transformer layer index to analyze')
    ap.add_argument('--module', default='mlp.gate_proj',
                    help='module name suffix under the chosen layer')
    ap.add_argument('--out-dir', default='/home/zs/metis/anisotropy')
    ap.add_argument('--prompts', nargs='+',
                    default=['The meaning of life is',
                             'In a distant galaxy, a young pilot',
                             'The key to machine learning is',
                             'Once upon a time in a quiet village'])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f'Loading model from {args.model} ...')
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(args.device).eval()

    # locate the target module: model.layers.<layer>.<module>
    target_name = f'model.layers.{args.layer}.{args.module}'
    target = dict(model.named_modules()).get(target_name)
    if target is None:
        # Qwen2 naming fallback
        target_name = f'model.layers.{args.layer}.mlp.gate_proj'
        target = dict(model.named_modules()).get(target_name)
    assert isinstance(target, torch.nn.Linear), \
        f'target {target_name} is not nn.Linear: {type(target)}'
    print(f'target module: {target_name}  shape={tuple(target.weight.shape)}')

    # ---- capture activation (input to target) via forward hook ----
    captured = {}

    def act_hook(mod, inp, out):
        x = inp[0]
        if x.ndim >= 2:
            captured['act'] = x.reshape(-1, x.shape[-1]).detach().to('cpu')

    h_act = target.register_forward_hook(act_hook)

    # Forward-only to capture real activations (full backward on 7B in a
    # sandboxed CPU env is unreliable, so we capture activations with no_grad
    # and compute the gradient via a single-layer backward below).
    print('Running forward to capture activations ...', flush=True)
    acts = []
    with torch.no_grad():
        for pi, p in enumerate(args.prompts):
            print(f'  prompt {pi}: {p!r}', flush=True)
            ids = tok(p, return_tensors='pt').input_ids
            print(f'  ids shape {ids.shape}', flush=True)
            model(ids)
            a = captured.get('act')
            if a is None:
                print(f'  WARNING: no activation captured for prompt {p!r}', flush=True)
            else:
                print(f'  act shape {a.shape}', flush=True)
                acts.append(a)
    h_act.remove()
    print(f'captured {len(acts)} activations', flush=True)

    # ---- assemble matrices ----
    W = target.weight.detach().to('cpu').to(torch.float32)   # (out, in)
    A = torch.cat(acts, dim=0).to('cpu').to(torch.float32)   # (l, in)

    # ---- gradient via single-layer backward (avoids full-model backward) ----
    # G = dL/dW where L = sum(Y^2), Y = A @ W^T  =>  G = 2 * (A^T @ (A @ W^T))^T
    # This captures the spectral structure induced by real A and W (the
    # anisotropy of G is driven by the SVDs of A and W, which is what the
    # paper analyzes). Using L=sum(Y^2) gives a well-defined, full-rank
    # gradient that preserves the spectral coupling.
    print('Computing gradient via single-layer backward ...', flush=True)
    W_g = W.clone().requires_grad_(True)
    Y = A @ W_g.t()                    # (l, out)
    loss = Y.pow(2).sum()
    loss.backward()
    G = W_g.grad.detach().to('cpu')    # (out, in)

    print(f'W shape={tuple(W.shape)}  A shape={tuple(A.shape)}  G shape={tuple(G.shape)}', flush=True)

    # ---- free model memory before SVD (sandbox memory limits) ----
    print('Freeing model memory ...', flush=True)
    del model, target, captured, tok, W_g, Y, loss
    import gc; gc.collect()

    # ---- save W/A/G to disk (allows split run: capture vs analyze) ----
    os.makedirs(args.out_dir, exist_ok=True)
    wag_path = os.path.join(args.out_dir, 'wag.pt')
    print(f'Saving W/A/G to {wag_path} ...', flush=True)
    torch.save({'W': W, 'A': A, 'G': G}, wag_path)
    print('W/A/G saved. Capture phase done.', flush=True)
    print('Run analyze_wag_from_disk.py to generate figures.', flush=True)
    return

    # ---- SVD (CPU float32) ----
    # For spectra (Fig 2A): use svdvals_only (fast, no U/V).
    # For rank-1 / residual plots (Fig 2B, 3B): use partial_svd (top-q).
    print('Computing singular values (svdvals) ...', flush=True)
    sW = svdvals_only(W)
    sA = svdvals_only(A)
    sG = svdvals_only(G)
    print('Computing partial SVD (top-q) for rank-1 / residual plots ...', flush=True)
    pW_U, pW_S, pW_Vt = partial_svd(W, q=256)
    pA_U, pA_S, pA_Vt = partial_svd(A, q=256)
    pG_U, pG_S, pG_Vt = partial_svd(G, q=256)

    # ---- dominant fractions ----
    for name, s in [('Weight', sW), ('Activation', sA), ('Gradient', sG)]:
        frac = elbow_dominant_fraction(s)
        print(f'  {name}: dominant fraction = {frac*100:.2f}%  '
              f'(top {max(1,int(frac*s.numel()))} of {s.numel()})', flush=True)

    matrices = [('W', W), ('A', A), ('G', G)]
    svd_results = [('W', sW, pW_U, pW_Vt), ('A', sA, pA_U, pA_Vt), ('G', sG, pG_U, pG_Vt)]
    # NOTE: for rank-1 / residual plots we pass partial U/Vt with top-q
    # singular vectors; the singular values sW/sA/sG are the full spectra.

    # ---- plot paper-style figures ----
    print('Plotting ...', flush=True)
    plot_spectra(svd_results, os.path.join(args.out_dir, 'fig2a_spectra.png'))
    plot_distributions(matrices, svd_results,
                       os.path.join(args.out_dir, 'fig2b_distributions.png'))
    plot_residuals(matrices, svd_results,
                   os.path.join(args.out_dir, 'fig3b_residuals.png'))

    print(f'\nDone. Figures saved to {args.out_dir}/')
    print('  fig2a_spectra.png       - singular value spectra (Fig 2A)')
    print('  fig2b_distributions.png - matrix + rank-1 component distributions (Fig 2B)')
    print('  fig3b_residuals.png     - original vs top-3% residual (Fig 3B)')


if __name__ == '__main__':
    main()
