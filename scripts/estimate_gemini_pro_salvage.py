"""Estimate salvage bounds for gemini-2.5-pro without new API calls.

The script reads existing ``records.csv`` files and reports:
- observed: the original strict score in the CSV.
- stage1_ready: an upper-ish bound from correct tool selection and arguments.
- flash_calibrated: stage1_ready multiplied by the matched Gemini Flash
  conditional success rate among stage1-ready records.
- peer_calibrated: stage1_ready multiplied by the mean conditional success
  rate of non-Pro models for the same family and severity.

These are estimates for analysis notes, not replacement raw measurements.
"""

from __future__ import annotations

import argparse
import ast
import csv
import math
from pathlib import Path
from typing import Iterable


SEVERITIES = ("clean", "light", "medium", "heavy")
SALVAGE_FAMILIES = ("observation", "execution_environment")


def _read_records(results_root: Path, model: str, family: str) -> list[dict[str, str]]:
    path = results_root / model / family / "records.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_dict(value: str | None) -> tuple[bool, dict[str, object]]:
    if not value:
        return True, {}
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return False, {}
    return isinstance(parsed, dict), parsed if isinstance(parsed, dict) else {}


def _stage1_ready(record: dict[str, str]) -> bool:
    """Recover stage-1 readiness from persisted fields.

    For observation/execution families, the stored ``format_valid`` can be
    overwritten by stage-2 validity. The tool name and argument strings remain
    stage-1 fields, so we reconstruct readiness from those.
    """

    if record.get("call_present") != "1" or record.get("canonical_tool_match") != "1":
        return False
    args_ok, args = _parse_dict(record.get("predicted_arguments"))
    expected_ok, expected = _parse_dict(record.get("expected_arguments"))
    if not args_ok or not expected_ok:
        return False
    return all(args.get(key) == value for key, value in expected.items())


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else math.nan


def _rate(rows: list[dict[str, str]], predicate) -> float:
    return sum(1 for row in rows if predicate(row)) / len(rows) if rows else math.nan


def _conditional_success(rows: list[dict[str, str]]) -> float:
    ready_rows = [row for row in rows if _stage1_ready(row)]
    if not ready_rows:
        return math.nan
    return sum(int(row.get("overall_task_success", "0") or 0) for row in ready_rows) / len(ready_rows)


def _approx_ci(rate: float, n: int) -> tuple[float, float]:
    if math.isnan(rate) or n <= 0:
        return math.nan, math.nan
    se = math.sqrt(max(rate * (1.0 - rate), 0.0) / n)
    return max(0.0, rate - 1.96 * se), min(1.0, rate + 1.96 * se)


def build_estimates(results_root: Path, target_model: str, flash_model: str) -> list[dict[str, object]]:
    models = sorted(path.name for path in results_root.iterdir() if path.is_dir())
    peer_models = [model for model in models if model != target_model]
    rows_out: list[dict[str, object]] = []

    for family in SALVAGE_FAMILIES:
        target_records = _read_records(results_root, target_model, family)
        if not target_records:
            continue
        for severity in SEVERITIES:
            rows = [row for row in target_records if row.get("severity") == severity]
            if not rows:
                continue
            observed = _rate(rows, lambda row: row.get("overall_task_success") == "1")
            stage1 = _rate(rows, _stage1_ready)
            parse_failure = _rate(
                rows,
                lambda row: row.get("secondary_reason") in {"unparsable_tool_call", "invalid_json_arguments"},
            )
            no_call = _rate(rows, lambda row: row.get("secondary_reason") == "no_tool_call")

            flash_records = [
                row
                for row in _read_records(results_root, flash_model, family)
                if row.get("severity") == severity
            ]
            flash_cond = _conditional_success(flash_records)
            flash_estimate = stage1 * flash_cond if not math.isnan(flash_cond) else math.nan
            flash_low, flash_high = _approx_ci(flash_estimate, len(rows))

            peer_conds = [
                _conditional_success(
                    [
                        row
                        for row in _read_records(results_root, peer_model, family)
                        if row.get("severity") == severity
                    ]
                )
                for peer_model in peer_models
            ]
            peer_conds = [value for value in peer_conds if not math.isnan(value)]
            peer_mean = _mean(peer_conds)
            peer_min = min(peer_conds) if peer_conds else math.nan
            peer_max = max(peer_conds) if peer_conds else math.nan
            peer_estimate = stage1 * peer_mean if not math.isnan(peer_mean) else math.nan

            rows_out.append(
                {
                    "model": target_model,
                    "family": family,
                    "severity": severity,
                    "n": len(rows),
                    "observed_strict": observed,
                    "stage1_ready_upper": stage1,
                    "parse_failure_rate": parse_failure,
                    "no_call_rate": no_call,
                    "flash_conditional_success": flash_cond,
                    "flash_calibrated_estimate": flash_estimate,
                    "flash_estimate_ci95_low": flash_low,
                    "flash_estimate_ci95_high": flash_high,
                    "peer_conditional_mean": peer_mean,
                    "peer_conditional_min": peer_min,
                    "peer_conditional_max": peer_max,
                    "peer_calibrated_estimate": peer_estimate,
                    "peer_calibrated_min": stage1 * peer_min if not math.isnan(peer_min) else math.nan,
                    "peer_calibrated_max": stage1 * peer_max if not math.isnan(peer_max) else math.nan,
                }
            )
    return rows_out


def _fmt(value: object) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.3f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "family",
        "severity",
        "n",
        "observed_strict",
        "stage1_ready_upper",
        "parse_failure_rate",
        "flash_calibrated_estimate",
        "peer_calibrated_estimate",
        "peer_calibrated_min",
        "peer_calibrated_max",
    ]
    lines = [
        "# Gemini Pro Salvage Estimates",
        "",
        "Generated from existing `records.csv` files only. No API calls are made.",
        "",
        "`observed_strict` is the original benchmark score. `stage1_ready_upper` is not a final score; it is the fraction of records with a correct tool and arguments before the stage-2 parser failure. Calibrated estimates are sensitivity analyses, not replacement measurements.",
        "",
        "|" + "|".join(headers) + "|",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("|" + "|".join(_fmt(row[header]) for header in headers) + "|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="data/results/all-family")
    parser.add_argument("--target-model", default="gemini-2.5-pro")
    parser.add_argument("--flash-model", default="gemini-2.5-flash")
    parser.add_argument("--output-dir", default="data/results/salvage")
    args = parser.parse_args()

    rows = build_estimates(Path(args.results_root), args.target_model, args.flash_model)
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "gemini_2.5_pro_salvage_estimates.csv", rows)
    write_markdown(output_dir / "gemini_2.5_pro_salvage_estimates.md", rows)
    print(f"Wrote {len(rows)} rows to {output_dir}")


if __name__ == "__main__":
    main()
