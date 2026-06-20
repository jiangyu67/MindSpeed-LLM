"""Standalone SVD + plotting for W/A/G anisotropy analysis.

Loads W/A/G from wag.pt (saved by analyze_anisotropy.py) and generates the
paper-style figures. Split into a separate script to avoid sandbox time
limits when running the full capture + analysis in one process.

Usage:
    python analyze_wag_from_disk.py --in-dir /home/zs/ckpt/qwen25-7b/anisotropy
"""
import argparse
import os

import torch

# reuse helpers from analyze_anisotropy
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_anisotropy import (
    svdvals_only, partial_svd, elbow_dominant_fraction,
    plot_spectra, plot_distributions, plot_residuals,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-dir', default='/home/zs/ckpt/qwen25-7b/anisotropy')
    args = ap.parse_args()

    wag_path = os.path.join(args.in_dir, 'wag.pt')
    print(f'Loading W/A/G from {wag_path} ...', flush=True)
    data = torch.load(wag_path, map_location='cpu')
    W = data['W'].to(torch.float32)
    A = data['A'].to(torch.float32)
    G = data['G'].to(torch.float32)
    print(f'W shape={tuple(W.shape)}  A shape={tuple(A.shape)}  G shape={tuple(G.shape)}', flush=True)

    # ---- SVD (CPU float32) ----
    # Use partial_svd once per matrix (gets S, U, Vt in one call).
    # For spectra we use the top-q singular values; for rank-1 / residual
    # plots we use the corresponding U/Vt.
    print('Computing partial SVD for W ...', flush=True)
    sW, pW_U, pW_Vt = partial_svd(W, q=256)
    print(f'  sW[:5]={sW[:5].tolist()}', flush=True)
    print('Computing partial SVD for A ...', flush=True)
    sA, pA_U, pA_Vt = partial_svd(A, q=256)
    print(f'  sA[:5]={sA[:5].tolist()}', flush=True)
    print('Computing partial SVD for G ...', flush=True)
    sG, pG_U, pG_Vt = partial_svd(G, q=256)
    print(f'  sG[:5]={sG[:5].tolist()}', flush=True)

    # ---- dominant fractions ----
    for name, s in [('Weight', sW), ('Activation', sA), ('Gradient', sG)]:
        frac = elbow_dominant_fraction(s)
        print(f'  {name}: dominant fraction = {frac*100:.2f}%  '
              f'(top {max(1,int(frac*s.numel()))} of {s.numel()})', flush=True)

    matrices = [('W', W), ('A', A), ('G', G)]
    svd_results = [('W', sW, pW_U, pW_Vt), ('A', sA, pA_U, pA_Vt), ('G', sG, pG_U, pG_Vt)]

    # ---- plot paper-style figures ----
    print('Plotting fig2a_spectra ...', flush=True)
    # plot_spectra expects [(name, S), ...]
    spectra_specs = [('W', sW), ('A', sA), ('G', sG)]
    plot_spectra(spectra_specs, os.path.join(args.in_dir, 'fig2a_spectra.png'))
    print('Plotting fig2b_distributions ...', flush=True)
    plot_distributions(matrices, svd_results,
                       os.path.join(args.in_dir, 'fig2b_distributions.png'))
    print('Plotting fig3b_residuals ...', flush=True)
    plot_residuals(matrices, svd_results,
                   os.path.join(args.in_dir, 'fig3b_residuals.png'))

    print(f'\nDone. Figures saved to {args.in_dir}/', flush=True)


if __name__ == '__main__':
    main()
