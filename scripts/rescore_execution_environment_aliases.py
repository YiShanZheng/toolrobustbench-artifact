"""Recompute execution-environment records after failure-type alias changes."""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from toolrobust.cli.main import resolve_configs
from toolrobust.domain.enums import OutputProfile
from toolrobust.evaluation.failure_taxonomy import analyze_failure, decide_observed_failure, overall_success
from toolrobust.runner.experiment_runner import ExperimentRunner
from toolrobust.evaluation.scorer import ExecutionExpectation, Scorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="data/results/all-family", type=Path)
    parser.add_argument("--config-dir", default="data/generated/run_configs/all-family", type=Path)
    parser.add_argument("--models-config", default="configs/models.yaml", type=Path)
    parser.add_argument("--models", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def slug(value: str) -> str:
    for char in '\\/:*?"<>| \t\r\n,':
        value = value.replace(char, "_")
    return value


def to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes"}


def to_int(value: object) -> int:
    if pd.isna(value):
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 1 if to_bool(value) else 0


def split_expected(value: object) -> tuple[str, ...] | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return tuple(part for part in text.split("|") if part)


def failure_type_match(expected_failure_types: object, observed: object) -> bool | None:
    accepted = split_expected(expected_failure_types)
    if accepted is None:
        return None
    if observed is None or pd.isna(observed):
        observed = None
    expectation = ExecutionExpectation("error", False, None, accepted)
    return Scorer._failure_type_matches(expectation, observed)


def rescore_frame(frame: pd.DataFrame, task_map: dict[str, object]) -> tuple[pd.DataFrame, int]:
    updated = frame.copy()
    changed = 0
    for idx, row in updated.iterrows():
        if str(row.get("severity", "")) == "clean":
            continue
        if str(row.get("family", "")) != "execution_environment":
            continue
        task = task_map.get(str(row.get("sample_id", "")))
        if task is None:
            continue
        new_failure_type_match = failure_type_match(row.get("expected_failure_types"), row.get("decision_failure_type"))
        if new_failure_type_match is None:
            continue

        old_success = to_int(row.get("overall_task_success"))
        call_present = to_int(row.get("call_present"))
        format_valid = to_int(row.get("format_valid"))
        canonical_tool_match = to_int(row.get("canonical_tool_match"))
        interface_name_match = to_int(row.get("interface_name_match"))
        missing_parameter_flag = to_int(row.get("missing_parameter_flag"))
        status_match = bool(to_bool(row.get("stage2_status_match")))
        retryable_match = bool(to_bool(row.get("stage2_retryable_match")))
        result_match = bool(to_bool(row.get("stage2_result_match")))
        decision_format_valid = format_valid == 1

        stage1_ready = bool(
            call_present == 1
            and canonical_tool_match == 1
            and missing_parameter_flag == 0
        )
        expected_statuses = split_expected(row.get("expected_statuses")) or tuple()
        decision_status = str(row.get("decision_status") or "")
        if decision_status and decision_status != "nan":
            expects_result = decision_status in {"success", "partial"} if status_match else any(status in {"success", "partial"} for status in expected_statuses)
        else:
            expects_result = any(status in {"success", "partial"} for status in expected_statuses)
        wrong_parameter_flag = int(stage1_ready and decision_format_valid and expects_result and not result_match)
        execution_success = int(
            stage1_ready
            and decision_format_valid
            and status_match
            and retryable_match
            and new_failure_type_match
            and result_match
        )
        overall = int(
            overall_success(
                canonical_tool_match,
                format_valid,
                missing_parameter_flag,
                wrong_parameter_flag,
                execution_success,
                call_present,
            )
        )

        decision = decide_observed_failure(
            call_present=call_present,
            format_valid=format_valid,
            predicted_tool_raw=row.get("predicted_tool_raw"),
            predicted_tool_resolved=row.get("predicted_tool_resolved"),
            interface_name_match=interface_name_match,
            canonical_tool_match=canonical_tool_match,
            missing_parameter_flag=missing_parameter_flag,
            wrong_parameter_flag=wrong_parameter_flag,
            execution_success=execution_success,
            notes=str(row.get("notes") or ""),
        )
        failure = analyze_failure(
            task=task,
            decision=decision,
            call_present=call_present,
            canonical_tool_match=canonical_tool_match,
            format_valid=format_valid,
            interface_name_match=interface_name_match,
            missing_parameter_flag=missing_parameter_flag,
            wrong_parameter_flag=wrong_parameter_flag,
            execution_success=execution_success,
            notes=str(row.get("notes") or ""),
        )

        updated.at[idx, "stage2_failure_type_match"] = bool(new_failure_type_match)
        updated.at[idx, "wrong_parameter_flag"] = wrong_parameter_flag
        updated.at[idx, "execution_success"] = execution_success
        updated.at[idx, "overall_task_success"] = overall
        updated.at[idx, "status"] = failure.status
        updated.at[idx, "observed_failure_tag"] = None if failure.observed_failure_tag is None else failure.observed_failure_tag.value
        updated.at[idx, "primary_failure_tag"] = None if failure.primary_failure_tag is None else failure.primary_failure_tag.value
        updated.at[idx, "first_failure_stage"] = failure.first_failure_stage
        updated.at[idx, "is_cascade_failure"] = failure.is_cascade_failure
        updated.at[idx, "cascade_from_stage"] = failure.cascade_from_stage
        updated.at[idx, "expected_primary_match"] = failure.expected_primary_match
        updated.at[idx, "allowed_spillover_match"] = failure.allowed_spillover_match
        updated.at[idx, "boundary_violation_flag"] = failure.boundary_violation_flag
        updated.at[idx, "boundary_violation_reason"] = failure.boundary_violation_reason
        updated.at[idx, "secondary_reason"] = failure.secondary_reason
        if overall != old_success:
            changed += 1
    return updated, changed


def main() -> int:
    args = parse_args()
    requested = {item.strip() for item in args.models.split(",") if item.strip()}
    rows = []
    for records_csv in sorted(args.results_root.glob("*/execution_environment/records.csv")):
        model = records_csv.parent.parent.name
        if requested and model not in requested:
            continue
        config_path = args.config_dir / f"paper_all_family_{slug(model)}_execution_environment.yaml"
        if not config_path.exists():
            rows.append({"model": model, "status": "skipped_missing_config", "records": str(records_csv)})
            continue
        config, backend_configs, backend_map = resolve_configs(config_path, args.models_config)
        runner = ExperimentRunner(config, backend_configs, backend_map, output_profile=OutputProfile(config.output.profile), clean_output=False)
        runner.output_paths = runner._build_output_paths(explicit_results_dir=records_csv.parent)
        runner.manifest = runner._build_manifest()
        clean, perturbed = runner.build_data(progress=None)
        task_map = {task.sample_id: task for task in clean + perturbed}
        before = pd.read_csv(records_csv)
        after, changed = rescore_frame(before, task_map)
        before_success = float(before["overall_task_success"].astype(float).mean()) if not before.empty else 0.0
        after_success = float(after["overall_task_success"].astype(float).mean()) if not after.empty else 0.0
        if not args.dry_run:
            backup = records_csv.with_name(f"{records_csv.name}.bak_before_alias_rescore_{int(time.time())}")
            shutil.copy2(records_csv, backup)
            runner._write_outputs(after, progress=None)
        else:
            backup = None
        rows.append(
            {
                "model": model,
                "rows": len(before),
                "success_changed_rows": changed,
                "success_before": before_success,
                "success_after": after_success,
                "backup": str(backup) if backup else None,
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
