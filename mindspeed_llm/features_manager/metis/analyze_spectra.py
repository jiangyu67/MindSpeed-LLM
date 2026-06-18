"""Analyze Metis spectrum logs saved by the train_step hook.

Reads ``<metis_output_dir>/spectra/metis_spectrum_step_*.pt`` files and
produces a summary CSV + per-parameter singular-value plots comparing the
original weight spectrum against the Metis-quantized spectrum.

Usage:
    python mindspeed_llm/features_manager/metis/analyze_spectra.py \
        --spectra-dir /home/zs/metis/output/spectra \
        --output-dir /home/zs/metis/output/analysis
"""
import argparse
import csv
import glob
import json
import os
from pathlib import Path
from typing import Dict, List

import torch


def load_spectrum_files(spectra_dir: str) -> List[Dict]:
    """Load all metis_spectrum_step_*.pt files sorted by step."""
    pattern = os.path.join(spectra_dir, 'metis_spectrum_step_*.pt')
    files = sorted(glob.glob(pattern))
    records = []
    for f in files:
        try:
            data = torch.load(f, map_location='cpu', weights_only=False)
            records.append(data)
        except Exception as e:
            print(f'Warning: failed to load {f}: {e}')
    records.sort(key=lambda r: r.get('step', 0))
    return records


def build_summary_table(records: List[Dict]) -> List[Dict]:
    """Flatten records into one row per (step, parameter)."""
    rows = []
    for rec in records:
        step = rec.get('step', 0)
        results = rec.get('results', {})
        for name, info in results.items():
            w_spec = info.get('weight_spectrum', {}) or {}
            q_spec = info.get('quantized_spectrum', {}) or {}
            rows.append({
                'step': step,
                'param': name,
                'shape': '_'.join(str(s) for s in info.get('shape', [])),
                'subspace_updated': int(info.get('subspace_updated', False)),
                'subspace_rank': info.get('subspace_rank', ''),
                'weight_top1': w_spec.get('top1', ''),
                'weight_top5': w_spec.get('top5', ''),
                'weight_energy_k': w_spec.get('energy_k', ''),
                'quant_top1': q_spec.get('top1', ''),
                'quant_top5': q_spec.get('top5', ''),
                'quant_energy_k': q_spec.get('energy_k', ''),
                'quant_rel_error': info.get('quant_rel_error', ''),
            })
    return rows


def write_csv(rows: List[Dict], out_path: str) -> None:
    if not rows:
        print('No rows to write.')
        return
    fields = list(rows[0].keys())
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} rows to {out_path}')


def compute_spectral_distortion(records: List[Dict]) -> Dict:
    """Aggregate per-parameter spectral distortion metrics across steps.

    Spectral distortion = relative L2 distance between the top-k singular
    values of the original and quantized weight.
    """
    per_param: Dict[str, List[float]] = {}
    for rec in records:
        for name, info in rec.get('results', {}).items():
            w_spec = info.get('weight_spectrum', {}) or {}
            q_spec = info.get('quantized_spectrum', {}) or {}
            s_w = torch.tensor(w_spec.get('singular_values', []))
            s_q = torch.tensor(q_spec.get('singular_values', []))
            if s_w.numel() == 0 or s_q.numel() == 0:
                continue
            k = min(s_w.numel(), s_q.numel())
            dist = (s_w[:k] - s_q[:k]).norm().item() / max(s_w[:k].norm().item(), 1e-8)
            per_param.setdefault(name, []).append(dist)
    summary = {}
    for name, dists in per_param.items():
        t = torch.tensor(dists)
        summary[name] = {
            'mean': float(t.mean().item()),
            'std': float(t.std().item()) if t.numel() > 1 else 0.0,
            'min': float(t.min().item()),
            'max': float(t.max().item()),
            'n_samples': t.numel(),
        }
    return summary


def plot_top_params(records: List[Dict], out_dir: str, top_k: int = 5) -> None:
    """Save singular-value curves for the top-k params by mean error.

    Uses matplotlib if available; otherwise saves raw data as .pt.
    """
    if not records:
        return
    # Collect last-step spectrum per param.
    last = records[-1]
    results = last.get('results', {})
    # Rank by quant_rel_error.
    ranked = sorted(
        results.items(),
        key=lambda kv: kv[1].get('quant_rel_error', 0.0),
        reverse=True,
    )[:top_k]

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        has_mpl = True
    except Exception:
        has_mpl = False

    payload = {}
    for name, info in ranked:
        w_spec = info.get('weight_spectrum', {}) or {}
        q_spec = info.get('quantized_spectrum', {}) or {}
        s_w = w_spec.get('singular_values', [])
        s_q = q_spec.get('singular_values', [])
        payload[name] = {'weight': s_w, 'quantized': s_q}
        if has_mpl and s_w and s_q:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.semilogy(range(len(s_w)), s_w, 'o-', label='weight', markersize=3)
            ax.semilogy(range(len(s_q)), s_q, 's-', label='quantized', markersize=3)
            ax.set_xlabel('singular value index')
            ax.set_ylabel('magnitude')
            ax.set_title(f'{name} (step {last.get("step", 0)})')
            ax.legend()
            ax.grid(True, which='both', alpha=0.3)
            safe_name = name.replace('/', '_').replace('.', '_')
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f'spectrum_{safe_name}.png'), dpi=120)
            plt.close(fig)

    torch.save(payload, os.path.join(out_dir, 'topk_spectra.pt'))
    print(f'Saved top-{top_k} spectra to {out_dir}')


def main(spectra_dir: str, output_dir: str, top_k: int = 5) -> None:
    os.makedirs(output_dir, exist_ok=True)
    records = load_spectrum_files(spectra_dir)
    if not records:
        print(f'No spectrum files found in {spectra_dir}')
        return
    print(f'Loaded {len(records)} spectrum records (steps '
          f'{records[0].get("step", 0)}..{records[-1].get("step", 0)})')

    rows = build_summary_table(records)
    write_csv(rows, os.path.join(output_dir, 'spectrum_summary.csv'))

    distortion = compute_spectral_distortion(records)
    with open(os.path.join(output_dir, 'spectral_distortion.json'), 'w') as f:
        json.dump(distortion, f, indent=2)
    print(f'Wrote spectral distortion summary for {len(distortion)} params')

    plot_top_params(records, output_dir, top_k=top_k)

    # Print a quick console summary.
    print('\n=== Per-parameter spectral distortion (mean) ===')
    for name, m in sorted(distortion.items(), key=lambda kv: kv[1]['mean'], reverse=True):
        print(f'  {name:50s} mean={m["mean"]:.4f} max={m["max"]:.4f} n={m["n_samples"]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spectra-dir', type=str,
                        default='/home/zs/metis/output/spectra',
                        help='Directory containing metis_spectrum_step_*.pt files')
    parser.add_argument('--output-dir', type=str,
                        default='/home/zs/metis/output/analysis',
                        help='Directory to write analysis outputs')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Number of top-error params to plot')
    args = parser.parse_args()
    main(args.spectra_dir, args.output_dir, args.top_k)
