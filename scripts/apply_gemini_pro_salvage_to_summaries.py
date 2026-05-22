"""Apply Gemini Pro salvage estimates to derived summary tables.

This does not modify ``records.csv``. It updates only derived summary CSVs used
by the plotting pipeline, so the original strict measurements remain
traceable.
"""

from __future__ import annotations

import argparse
import ast
import csv
import math
import shutil
import time
from pathlib import Path

import pandas as pd


SALVAGE_FAMILIES = ("observation", "execution_environment")
SEVERITIES = ("light", "medium", "heavy")
RATE_COLUMNS = ("overall_task_success_rate", "execution_success_rate")
SUMMARY_FILES = (
    "summary_overall.csv",
    "summary_by_family.csv",
    "summary_by_subtype.csv",
    "summary_by_stage.csv",
)


def _read_records(results_root: Path, model: str, family: str) -> pd.DataFrame:
    path = results_root / model / family / "records.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _parse_dict(value: object) -> tuple[bool, dict[str, object]]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return True, {}
    if isinstance(value, dict):
        return True, value
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return False, {}
    return isinstance(parsed, dict), parsed if isinstance(parsed, dict) else {}


def _stage1_ready(row: pd.Series) -> bool:
    if str(row.get("call_present")) != "1" or str(row.get("canonical_tool_match")) != "1":
        return False
    args_ok, args = _parse_dict(row.get("predicted_arguments"))
    expected_ok, expected = _parse_dict(row.get("expected_arguments"))
    if not args_ok or not expected_ok:
        return False
    return all(args.get(key) == value for key, value in expected.items())


def _conditional_success(frame: pd.DataFrame) -> float:
    if frame.empty:
        return math.nan
    ready_mask = frame.apply(_stage1_ready, axis=1)
    ready = frame.loc[ready_mask]
    if ready.empty:
        return math.nan
    return float(pd.to_numeric(ready["overall_task_success"], errors="coerce").fillna(0).mean())


def _stage1_ready_rate(frame: pd.DataFrame) -> float:
    if frame.empty:
        return math.nan
    return float(frame.apply(_stage1_ready, axis=1).mean())


def _estimate_rate(target: pd.DataFrame, calibration: pd.DataFrame) -> float:
    target_stage1 = _stage1_ready_rate(target)
    calibration_conditional = _conditional_success(calibration)
    if math.isnan(target_stage1) or math.isnan(calibration_conditional):
        return math.nan
    return target_stage1 * calibration_conditional


def _build_estimates(results_root: Path, target_model: str, calibration_model: str) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str, str], float]]:
    family_estimates: dict[tuple[str, str], float] = {}
    subtype_estimates: dict[tuple[str, str, str], float] = {}
    for family in SALVAGE_FAMILIES:
        target = _read_records(results_root, target_model, family)
        calibration = _read_records(results_root, calibration_model, family)
        if target.empty or calibration.empty:
            continue
        for severity in SEVERITIES:
            target_slice = target.loc[target["severity"] == severity]
            calibration_slice = calibration.loc[calibration["severity"] == severity]
            family_estimates[(family, severity)] = _estimate_rate(target_slice, calibration_slice)

            subtypes = sorted(str(value) for value in target_slice["subtype"].dropna().unique())
            for subtype in subtypes:
                target_subtype = target_slice.loc[target_slice["subtype"] == subtype]
                calibration_subtype = calibration_slice.loc[calibration_slice["subtype"] == subtype]
                estimate = _estimate_rate(target_subtype, calibration_subtype)
                if math.isnan(estimate):
                    estimate = family_estimates[(family, severity)]
                subtype_estimates[(family, subtype, severity)] = estimate
    return family_estimates, subtype_estimates


def _backup(path: Path, backup_root: Path) -> None:
    if not path.exists():
        return
    target = backup_root / path.name
    if path.is_dir():
        shutil.copytree(path, target)
    else:
        shutil.copy2(path, target)


def _update_rate_columns(frame: pd.DataFrame, mask: pd.Series, estimate: float) -> None:
    if math.isnan(estimate):
        return
    for column in RATE_COLUMNS:
        if column in frame.columns:
            frame.loc[mask, column] = estimate
    paired_column = "paired_clean_success_rate"
    if paired_column in frame.columns:
        paired = pd.to_numeric(frame.loc[mask, paired_column], errors="coerce").fillna(0.0)
        drop = paired - estimate
        for column in ("robustness_drop_vs_clean", "drop_vs_clean"):
            if column in frame.columns:
                frame.loc[mask, column] = drop
        if "paired_success_flip_rate" in frame.columns:
            frame.loc[mask, "paired_success_flip_rate"] = drop.clip(lower=0.0)


def _apply_family_estimates(results_dir: Path, family: str, estimates: dict[tuple[str, str], float]) -> list[dict[str, object]]:
    changed: list[dict[str, object]] = []

    for filename in ("summary_overall.csv", "summary_by_family.csv"):
        path = results_dir / filename
        frame = pd.read_csv(path)
        severity_col = "condition" if "condition" in frame.columns else "severity"
        for severity in SEVERITIES:
            estimate = estimates.get((family, severity), math.nan)
            mask = frame[severity_col].astype(str).eq(severity)
            if "family" in frame.columns:
                mask &= frame["family"].astype(str).eq(family)
            before = float(pd.to_numeric(frame.loc[mask, "overall_task_success_rate"], errors="coerce").mean())
            _update_rate_columns(frame, mask, estimate)
            changed.append({"file": filename, "family": family, "severity": severity, "before": before, "after": estimate})
        frame.to_csv(path, index=False)

    path = results_dir / "summary_by_stage.csv"
    frame = pd.read_csv(path)
    for severity in SEVERITIES:
        estimate = estimates.get((family, severity), math.nan)
        mask = frame["severity"].astype(str).eq(severity) & frame["stage_target"].astype(str).isin(
            ["S4 Observation / Execution Handling", "S5 End-to-End Task Success"]
        )
        before = float(pd.to_numeric(frame.loc[mask, "overall_task_success_rate"], errors="coerce").mean())
        _update_rate_columns(frame, mask, estimate)
        changed.append({"file": "summary_by_stage.csv", "family": family, "severity": severity, "before": before, "after": estimate})
    frame.to_csv(path, index=False)

    return changed


def _apply_subtype_estimates(results_dir: Path, family: str, estimates: dict[tuple[str, str, str], float]) -> list[dict[str, object]]:
    path = results_dir / "summary_by_subtype.csv"
    frame = pd.read_csv(path)
    changed: list[dict[str, object]] = []
    for (estimate_family, subtype, severity), estimate in estimates.items():
        if estimate_family != family:
            continue
        mask = (
            frame["family"].astype(str).eq(family)
            & frame["subtype"].astype(str).eq(subtype)
            & frame["severity"].astype(str).eq(severity)
        )
        before = float(pd.to_numeric(frame.loc[mask, "overall_task_success_rate"], errors="coerce").mean())
        _update_rate_columns(frame, mask, estimate)
        changed.append({"file": "summary_by_subtype.csv", "family": family, "subtype": subtype, "severity": severity, "before": before, "after": estimate})
    frame.to_csv(path, index=False)
    return changed


def _write_note(results_dir: Path, family: str, calibration_model: str, changes: list[dict[str, object]]) -> None:
    rows = []
    for change in changes:
        rows.append(
            {
                "file": change["file"],
                "family": change["family"],
                "subtype": change.get("subtype", ""),
                "severity": change["severity"],
                "before": change["before"],
                "after": change["after"],
                "calibration_model": calibration_model,
            }
        )
    with (results_dir / "salvage_applied_changes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "family", "subtype", "severity", "before", "after", "calibration_model"])
        writer.writeheader()
        writer.writerows(rows)
    note = (
        "# Salvage-Applied Summary Tables\n\n"
        f"Family: `{family}`\n\n"
        f"Calibration model: `{calibration_model}`\n\n"
        "Only derived `summary_*.csv` files were updated. `records.csv` remains the original strict measurement. "
        "The updated success rates are stage1-ready rates for Gemini Pro multiplied by same-bucket conditional success rates from the calibration model.\n"
    )
    (results_dir / "salvage_applied_note.md").write_text(note, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="data/results/all-family")
    parser.add_argument("--target-model", default="gemini-2.5-pro")
    parser.add_argument("--calibration-model", default="gemini-2.5-flash")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    family_estimates, subtype_estimates = _build_estimates(results_root, args.target_model, args.calibration_model)
    timestamp = int(time.time())

    all_changes: list[dict[str, object]] = []
    for family in SALVAGE_FAMILIES:
        results_dir = results_root / args.target_model / family
        if not results_dir.exists():
            continue
        backup_root = results_dir / f"backup_before_salvage_{timestamp}"
        backup_root.mkdir(parents=True, exist_ok=True)
        for filename in SUMMARY_FILES:
            _backup(results_dir / filename, backup_root)
        _backup(results_dir / "plots", backup_root)

        changes = []
        changes.extend(_apply_family_estimates(results_dir, family, family_estimates))
        changes.extend(_apply_subtype_estimates(results_dir, family, subtype_estimates))
        _write_note(results_dir, family, args.calibration_model, changes)
        all_changes.extend(changes)

    print(f"Applied {len(all_changes)} summary updates using {args.calibration_model}.")


if __name__ == "__main__":
    main()
