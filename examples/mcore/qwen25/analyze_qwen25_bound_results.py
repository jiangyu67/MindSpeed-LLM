#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from statistics import mean
from typing import Dict, List, Optional


@dataclass
class CaseMetrics:
    case_name: str
    seq_len: int
    mbs: int
    use_fused_swiglu: int
    exit_code: int
    tokens_s_avg: Optional[float]
    samples_s_avg: Optional[float]
    iter_ms_avg: Optional[float]
    aicore_avg: Optional[float]
    hbm_used_ratio_avg: Optional[float]


def _extract_float_matches(pattern: str, text: str) -> List[float]:
    values = []
    for m in re.finditer(pattern, text, flags=re.IGNORECASE):
        try:
            values.append(float(m.group(1)))
        except (ValueError, IndexError):
            continue
    return values


def _trim_warmup(values: List[float]) -> List[float]:
    if not values:
        return values
    if len(values) < 10:
        return values
    cut = int(len(values) * 0.4)
    return values[cut:]


def parse_train_log(log_file: str) -> Dict[str, Optional[float]]:
    if not os.path.exists(log_file):
        return {
            "tokens_s_avg": None,
            "samples_s_avg": None,
            "iter_ms_avg": None,
        }

    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # Compatible with common Megatron/MindSpeed log styles.
    tokens_values = _extract_float_matches(r"tokens/s[^0-9]*([0-9]+(?:\.[0-9]+)?)", text)
    samples_values = _extract_float_matches(r"samples/s[^0-9]*([0-9]+(?:\.[0-9]+)?)", text)
    iter_ms_values = _extract_float_matches(r"elapsed time per iteration \(ms\)[^0-9]*([0-9]+(?:\.[0-9]+)?)", text)

    tokens_values = _trim_warmup(tokens_values)
    samples_values = _trim_warmup(samples_values)
    iter_ms_values = _trim_warmup(iter_ms_values)

    return {
        "tokens_s_avg": mean(tokens_values) if tokens_values else None,
        "samples_s_avg": mean(samples_values) if samples_values else None,
        "iter_ms_avg": mean(iter_ms_values) if iter_ms_values else None,
    }


def parse_npu_log(npu_log_file: str) -> Dict[str, Optional[float]]:
    if not os.path.exists(npu_log_file):
        return {
            "aicore_avg": None,
            "hbm_used_ratio_avg": None,
        }

    aicore_values: List[float] = []
    hbm_ratio_values: List[float] = []

    # Typical row examples:
    # | 0 | ... | 86 | ... |
    # | 0 | ... | 40512 / 65536 |
    hbm_pattern = re.compile(r"([0-9]{3,6})\s*/\s*([0-9]{4,6})")
    percent_pattern = re.compile(r"(?<![0-9])([0-9]{1,3})(?:\.[0-9]+)?(?![0-9])")

    with open(npu_log_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            hbm_match = hbm_pattern.search(line)
            if hbm_match:
                used = float(hbm_match.group(1))
                total = float(hbm_match.group(2))
                if total > 0:
                    hbm_ratio_values.append(used / total * 100.0)

            if "AICore" in line or "aicore" in line:
                nums = [float(x) for x in percent_pattern.findall(line)]
                for v in nums:
                    if 0.0 <= v <= 100.0:
                        aicore_values.append(v)

    return {
        "aicore_avg": mean(_trim_warmup(aicore_values)) if aicore_values else None,
        "hbm_used_ratio_avg": mean(_trim_warmup(hbm_ratio_values)) if hbm_ratio_values else None,
    }


def safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return a / b


def verdict(metrics_map: Dict[str, CaseMetrics]) -> Dict[str, object]:
    baseline = metrics_map.get("baseline")
    low_seq = metrics_map.get("low_seq")
    high_mbs = metrics_map.get("high_mbs")

    if baseline is None:
        return {
            "verdict": "unknown",
            "reason": "missing baseline case",
            "confidence": "low",
        }

    baseline_tokens = baseline.tokens_s_avg
    low_seq_tokens = low_seq.tokens_s_avg if low_seq else None
    high_mbs_tokens = high_mbs.tokens_s_avg if high_mbs else None

    low_seq_speedup = safe_div(low_seq_tokens, baseline_tokens)
    high_mbs_speedup = safe_div(high_mbs_tokens, baseline_tokens)
    aicore_avg = baseline.aicore_avg
    hbm_ratio = baseline.hbm_used_ratio_avg

    # Conservative thresholds for first-pass diagnosis.
    memory_bias = 0.0
    compute_bias = 0.0

    if low_seq_speedup is not None and low_seq_speedup >= 1.10:
        memory_bias += 1.0
    if high_mbs_speedup is not None and high_mbs_speedup >= 1.10:
        compute_bias += 1.0

    if low_seq_speedup is not None and high_mbs_speedup is not None:
        if low_seq_speedup - high_mbs_speedup >= 0.08:
            memory_bias += 1.0
        if high_mbs_speedup - low_seq_speedup >= 0.08:
            compute_bias += 1.0

    if aicore_avg is not None:
        if aicore_avg >= 85:
            compute_bias += 1.0
        elif aicore_avg <= 60:
            memory_bias += 0.5

    if hbm_ratio is not None and hbm_ratio >= 85:
        memory_bias += 0.5

    if memory_bias >= compute_bias + 0.5:
        result = "memory-bound"
    elif compute_bias >= memory_bias + 0.5:
        result = "compute-bound"
    else:
        result = "mixed-or-uncertain"

    confidence = "medium"
    margin = abs(memory_bias - compute_bias)
    if margin >= 1.5:
        confidence = "high"
    elif margin < 0.5:
        confidence = "low"

    return {
        "verdict": result,
        "confidence": confidence,
        "scores": {
            "memory_bias": memory_bias,
            "compute_bias": compute_bias,
        },
        "signals": {
            "low_seq_speedup": low_seq_speedup,
            "high_mbs_speedup": high_mbs_speedup,
            "baseline_aicore_avg": aicore_avg,
            "baseline_hbm_used_ratio_avg": hbm_ratio,
        },
    }


def load_manifest(manifest_path: str) -> List[dict]:
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def write_metrics_csv(path: str, metrics: List[CaseMetrics]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "case_name",
            "seq_len",
            "mbs",
            "use_fused_swiglu",
            "exit_code",
            "tokens_s_avg",
            "samples_s_avg",
            "iter_ms_avg",
            "aicore_avg",
            "hbm_used_ratio_avg",
        ])
        for m in metrics:
            writer.writerow([
                m.case_name,
                m.seq_len,
                m.mbs,
                m.use_fused_swiglu,
                m.exit_code,
                m.tokens_s_avg,
                m.samples_s_avg,
                m.iter_ms_avg,
                m.aicore_avg,
                m.hbm_used_ratio_avg,
            ])


def write_summary_md(path: str, metrics: List[CaseMetrics], result: Dict[str, object]) -> None:
    lines = []
    lines.append("# Qwen25 Bound Analysis Summary")
    lines.append("")
    lines.append(f"- Verdict: {result.get('verdict')}")
    lines.append(f"- Confidence: {result.get('confidence')}")
    signals = result.get("signals", {})
    lines.append(f"- Low-seq speedup: {signals.get('low_seq_speedup')}")
    lines.append(f"- High-mbs speedup: {signals.get('high_mbs_speedup')}")
    lines.append(f"- Baseline AICore avg: {signals.get('baseline_aicore_avg')}")
    lines.append(f"- Baseline HBM used ratio avg: {signals.get('baseline_hbm_used_ratio_avg')}")
    lines.append("")
    lines.append("## Per-case Metrics")
    lines.append("")
    lines.append("| case | seq_len | mbs | fused_swiglu | exit | tokens/s | samples/s | iter_ms | aicore_avg | hbm_used_ratio_avg |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for m in metrics:
        lines.append(
            "| {case} | {seq} | {mbs} | {fused} | {exit_code} | {tokens} | {samples} | {iter_ms} | {aicore} | {hbm} |".format(
                case=m.case_name,
                seq=m.seq_len,
                mbs=m.mbs,
                fused=m.use_fused_swiglu,
                exit_code=m.exit_code,
                tokens=f"{m.tokens_s_avg:.4f}" if m.tokens_s_avg is not None and not math.isnan(m.tokens_s_avg) else "NA",
                samples=f"{m.samples_s_avg:.4f}" if m.samples_s_avg is not None and not math.isnan(m.samples_s_avg) else "NA",
                iter_ms=f"{m.iter_ms_avg:.4f}" if m.iter_ms_avg is not None and not math.isnan(m.iter_ms_avg) else "NA",
                aicore=f"{m.aicore_avg:.2f}" if m.aicore_avg is not None and not math.isnan(m.aicore_avg) else "NA",
                hbm=f"{m.hbm_used_ratio_avg:.2f}" if m.hbm_used_ratio_avg is not None and not math.isnan(m.hbm_used_ratio_avg) else "NA",
            )
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Qwen25 bound benchmark logs.")
    parser.add_argument("--manifest", required=True, help="Path to manifest.csv from benchmark script.")
    parser.add_argument("--output-dir", required=True, help="Output directory for reports.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = load_manifest(args.manifest)

    metrics: List[CaseMetrics] = []
    metrics_map: Dict[str, CaseMetrics] = {}

    for row in rows:
        case_name = row["case_name"]
        seq_len = int(row["seq_len"])
        mbs = int(row["mbs"])
        use_fused_swiglu = int(row["use_fused_swiglu"])
        exit_code = int(row["exit_code"])
        log_file = row["log_file"]
        npu_file = row["npu_file"]

        train_metrics = parse_train_log(log_file)
        npu_metrics = parse_npu_log(npu_file)

        item = CaseMetrics(
            case_name=case_name,
            seq_len=seq_len,
            mbs=mbs,
            use_fused_swiglu=use_fused_swiglu,
            exit_code=exit_code,
            tokens_s_avg=train_metrics["tokens_s_avg"],
            samples_s_avg=train_metrics["samples_s_avg"],
            iter_ms_avg=train_metrics["iter_ms_avg"],
            aicore_avg=npu_metrics["aicore_avg"],
            hbm_used_ratio_avg=npu_metrics["hbm_used_ratio_avg"],
        )
        metrics.append(item)
        metrics_map[case_name] = item

    result = verdict(metrics_map)

    metrics_csv = os.path.join(args.output_dir, "metrics_summary.csv")
    summary_md = os.path.join(args.output_dir, "summary.md")
    result_json = os.path.join(args.output_dir, "result.json")

    write_metrics_csv(metrics_csv, metrics)
    write_summary_md(summary_md, metrics, result)

    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"[OK] metrics csv: {metrics_csv}")
    print(f"[OK] summary md: {summary_md}")
    print(f"[OK] result json: {result_json}")


if __name__ == "__main__":
    main()
