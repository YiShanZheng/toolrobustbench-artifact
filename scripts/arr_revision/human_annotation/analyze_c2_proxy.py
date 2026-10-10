"""Recompute C2 machine-solvability proxies and sensitivity analyses."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
MODELS = (
    "claude-sonnet-4-6", "deepseek-v4-pro", "gpt-5.4-mini", "qwen-plus",
    "gemini-2.5-pro", "deepseek-v4-flash", "gemini-2.5-flash",
)
SEVERITIES = ("light", "medium", "heavy")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - m, c + m


def exact_binomial_two_sided(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / (2**n)
    return min(1.0, 2 * tail)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - position) + ordered[hi] * (position - lo)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "machine_proxy")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, str]] = []
    for model in MODELS:
        path = ROOT / "data" / "results" / "all-family" / model / "observation" / "records.csv"
        rows = [row for row in read_csv(path) if "C2" in row.get("subtype", "")]
        if len(rows) != 144:
            raise RuntimeError(f"Expected 144 C2 rows for {model}, found {len(rows)}")
        records.extend(rows)

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in records:
        grouped[row["sample_id"]].append(row)
    sample_rows: list[dict[str, Any]] = []
    for sample_id, group in grouped.items():
        successes = sum(int(float(row["overall_task_success"])) for row in group)
        first = group[0]
        sample_rows.append({
            "sample_id": sample_id,
            "source_sample_id": first["source_sample_id"],
            "severity": first["severity"],
            "tool": first["gold_canonical_tool_id"],
            "operator": first["operator"],
            "n_models_success": successes,
            "any_model_success": int(successes >= 1),
            "at_least_2": int(successes >= 2),
            "at_least_3": int(successes >= 3),
            "majority_4": int(successes >= 4),
            "all_7": int(successes == 7),
        })
    sample_rows.sort(key=lambda row: row["sample_id"])
    write_csv(output / "sample_level_proxy.csv", sample_rows, list(sample_rows[0]))

    threshold_rows = []
    for severity in (*SEVERITIES, "all"):
        subset = sample_rows if severity == "all" else [row for row in sample_rows if row["severity"] == severity]
        for threshold, field in ((1, "any_model_success"), (2, "at_least_2"), (3, "at_least_3"), (4, "majority_4"), (7, "all_7")):
            k = sum(row[field] for row in subset)
            lo, hi = wilson(k, len(subset))
            threshold_rows.append({"severity": severity, "success_threshold_models": threshold, "success": k, "n": len(subset), "rate": k / len(subset), "wilson95_low": lo, "wilson95_high": hi})
    write_csv(output / "threshold_sensitivity.csv", threshold_rows, list(threshold_rows[0]))

    model_rows = []
    for model in MODELS:
        model_records = [row for row in records if row["model_name"] == model]
        for severity in (*SEVERITIES, "all"):
            subset = model_records if severity == "all" else [row for row in model_records if row["severity"] == severity]
            k = sum(int(float(row["overall_task_success"])) for row in subset)
            lo, hi = wilson(k, len(subset))
            model_rows.append({"model": model, "severity": severity, "success": k, "n": len(subset), "rate": k / len(subset), "wilson95_low": lo, "wilson95_high": hi})
    write_csv(output / "per_model_c2.csv", model_rows, list(model_rows[0]))

    leave_one_out = []
    for excluded in MODELS:
        kept = [row for row in records if row["model_name"] != excluded]
        kept_grouped: dict[str, int] = defaultdict(int)
        for row in kept:
            kept_grouped[row["sample_id"]] = max(kept_grouped[row["sample_id"]], int(float(row["overall_task_success"])))
        for severity in (*SEVERITIES, "all"):
            ids = [row["sample_id"] for row in sample_rows if severity == "all" or row["severity"] == severity]
            k = sum(kept_grouped[item] for item in ids)
            leave_one_out.append({"excluded_model": excluded, "severity": severity, "success": k, "n": len(ids), "rate": k / len(ids)})
    write_csv(output / "leave_one_model_out.csv", leave_one_out, list(leave_one_out[0]))

    rng = random.Random(args.seed)
    anchors = sorted({row["source_sample_id"] for row in sample_rows})
    by_anchor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        by_anchor[row["source_sample_id"]].append(row)
    bootstrap_rows = []
    for severity in (*SEVERITIES, "all"):
        estimates = []
        for _ in range(args.bootstrap):
            draw = [rng.choice(anchors) for _ in anchors]
            drawn_rows = [row for anchor in draw for row in by_anchor[anchor] if severity == "all" or row["severity"] == severity]
            estimates.append(sum(row["any_model_success"] for row in drawn_rows) / len(drawn_rows))
        observed_rows = sample_rows if severity == "all" else [row for row in sample_rows if row["severity"] == severity]
        observed = sum(row["any_model_success"] for row in observed_rows) / len(observed_rows)
        bootstrap_rows.append({"severity": severity, "observed_rate": observed, "cluster_bootstrap95_low": percentile(estimates, 0.025), "cluster_bootstrap95_high": percentile(estimates, 0.975), "n_anchor_clusters": len(anchors), "iterations": args.bootstrap})
    write_csv(output / "cluster_bootstrap.csv", bootstrap_rows, list(bootstrap_rows[0]))

    paired = {row["source_sample_id"]: {} for row in sample_rows}
    for row in sample_rows:
        paired[row["source_sample_id"]][row["severity"]] = row["any_model_success"]
    pair_rows = []
    for left, right in (("light", "medium"), ("light", "heavy"), ("medium", "heavy")):
        b = sum(values[left] == 1 and values[right] == 0 for values in paired.values())
        c = sum(values[left] == 0 and values[right] == 1 for values in paired.values())
        pair_rows.append({"contrast": f"{left}_vs_{right}", "left_only_success": b, "right_only_success": c, "discordant": b + c, "exact_mcnemar_p": exact_binomial_two_sided(b, c)})
    write_csv(output / "paired_severity_mcnemar.csv", pair_rows, list(pair_rows[0]))

    by_tool_rows = []
    for tool in sorted({row["tool"] for row in sample_rows}):
        for severity in (*SEVERITIES, "all"):
            subset = [row for row in sample_rows if row["tool"] == tool and (severity == "all" or row["severity"] == severity)]
            k = sum(row["any_model_success"] for row in subset)
            by_tool_rows.append({"tool": tool, "severity": severity, "success": k, "n": len(subset), "rate": k / len(subset)})
    write_csv(output / "proxy_by_tool.csv", by_tool_rows, list(by_tool_rows[0]))

    key = {(row["severity"], row["success_threshold_models"]): row for row in threshold_rows}
    report = f"""# C2 machine-proxy and sensitivity analysis

## Primary conservative existence proxy

At least one of seven evaluated models exactly succeeds on **{key[("all", 1)]["success"]}/144 = {100*key[("all", 1)]["rate"]:.1f}%** of C2 instances (Wilson 95% CI {100*key[("all", 1)]["wilson95_low"]:.1f}–{100*key[("all", 1)]["wilson95_high"]:.1f}). By severity: light {key[("light", 1)]["success"]}/48 ({100*key[("light", 1)]["rate"]:.1f}%), medium {key[("medium", 1)]["success"]}/48 ({100*key[("medium", 1)]["rate"]:.1f}%), heavy {key[("heavy", 1)]["success"]}/48 ({100*key[("heavy", 1)]["rate"]:.1f}%).

This is a **lower-bound existence witness**, not an estimate of human recoverability: failure by all seven models does not imply unsolvability, and success by one model does not show that a typical human can recover the target.

## Robustness checks produced

- `threshold_sensitivity.csv`: requires ≥1, ≥2, ≥3, ≥4, or all 7 models to succeed.
- `leave_one_model_out.csv`: quantifies dependence on any single model.
- `cluster_bootstrap.csv`: resamples the 48 clean-anchor clusters, preserving the three repeated severities.
- `paired_severity_mcnemar.csv`: paired exact McNemar tests across severity on the same anchors.
- `proxy_by_tool.csv`: identifies tool strata that drive the aggregate.
- `per_model_c2.csv`: model-specific rates with Wilson intervals.

## Reporting rule

Keep these machine proxies separate from the blinded human audit. Do not replace missing human labels with model-derived labels, and do not call the any-model proxy “provably solvable” without the narrower phrase “witnessed solvable for at least one evaluated model under this scorer.”
"""
    (output / "MACHINE_PROXY_REPORT.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
