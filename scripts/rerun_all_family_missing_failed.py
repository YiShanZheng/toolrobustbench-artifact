"""Repair and resume all-family records for selected model/family runs.

For each requested family, this script:
1. backs up records.csv when it exists;
2. removes rows whose notes contain "backend request failed";
3. rewrites records.csv with normal UTF-8 so resume counting can read sample_id;
4. runs only missing sample_ids;
5. appends only successful backend responses.

If a backend failure occurs during rerun, the script stops before writing the
failed row, leaving the sample missing rather than polluting records.csv.
"""

from __future__ import annotations

import argparse
import json
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
from toolrobust.runner.experiment_runner import ExperimentRunner


DEFAULT_FAMILIES = [
    "interface_exposure",
    "user_intent",
    "observation",
    "execution_environment",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--families", default=",".join(DEFAULT_FAMILIES))
    parser.add_argument("--config-dir", default="data/generated/run_configs/all-family", type=Path)
    parser.add_argument("--models-config", default="configs/models.yaml", type=Path)
    parser.add_argument("--results-root", default="data/results/all-family", type=Path)
    parser.add_argument("--cleanup-only", action="store_true")
    parser.add_argument("--refresh-outputs", action="store_true")
    return parser.parse_args()


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def slug(value: str) -> str:
    for char in '\\/:*?"<>| \t\r\n':
        value = value.replace(char, "_")
    return value


def backend_failed_mask(frame: pd.DataFrame) -> pd.Series:
    if frame.empty or "notes" not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame["notes"].fillna("").astype(str).str.contains("backend request failed", regex=False)


def run_family(args: argparse.Namespace, family: str) -> int:
    model_slug = slug(args.model)
    config_path = args.config_dir / f"paper_all_family_{model_slug}_{family}.yaml"
    results_dir = args.results_root / model_slug / family
    records_csv = results_dir / "records.csv"
    if not config_path.exists():
        raise FileNotFoundError(f"missing generated config: {config_path}")

    config, backend_configs, backend_map = resolve_configs(config_path, args.models_config)
    if args.model not in backend_map:
        raise ValueError(f"model {args.model!r} is not included by {config_path}")

    runner = ExperimentRunner(
        config,
        backend_configs,
        backend_map,
        output_profile=OutputProfile(config.output.profile),
        clean_output=False,
    )
    runner.output_paths = runner._build_output_paths(explicit_results_dir=results_dir)
    runner.manifest = runner._build_manifest()
    results_dir.mkdir(parents=True, exist_ok=True)

    clean_tasks, perturbed_tasks = runner.build_data(progress=None)
    tasks = clean_tasks + perturbed_tasks
    task_ids = {task.sample_id for task in tasks}

    if records_csv.exists():
        existing = pd.read_csv(records_csv)
        backup = records_csv.with_name(f"{records_csv.name}.bak_before_repair_{int(time.time())}")
        shutil.copy2(records_csv, backup)
    else:
        existing = pd.DataFrame()
        backup = None

    if not existing.empty:
        relevant = (
            (existing.get("model_name", pd.Series("", index=existing.index)).astype(str) == args.model)
            & existing.get("sample_id", pd.Series("", index=existing.index)).astype(str).isin(task_ids)
        )
        failed = relevant & backend_failed_mask(existing)
        cleaned = existing.loc[~failed].copy()
    else:
        failed = pd.Series(dtype=bool)
        cleaned = existing

    if records_csv.exists() or not cleaned.empty:
        cleaned.to_csv(records_csv, index=False)

    if not cleaned.empty and "model_name" in cleaned.columns and "sample_id" in cleaned.columns:
        completed_ids = set(
            cleaned.loc[
                cleaned["model_name"].astype(str).eq(args.model)
                & cleaned["sample_id"].astype(str).isin(task_ids),
                "sample_id",
            ].astype(str)
        )
    else:
        completed_ids = set()
    tasks_to_run = [task for task in tasks if task.sample_id not in completed_ids]

    print(
        json.dumps(
            {
                "family": family,
                "records_before": int(len(existing)),
                "failed_rows_removed": int(failed.sum()) if len(failed) else 0,
                "records_after_cleanup": int(len(cleaned)),
                "total_tasks": len(tasks),
                "already_completed": len(completed_ids),
                "to_run": len(tasks_to_run),
                "backup": str(backup) if backup else None,
            },
            ensure_ascii=False,
        )
    )

    if args.cleanup_only or not tasks_to_run:
        if args.refresh_outputs and not cleaned.empty:
            runner._write_outputs(pd.read_csv(records_csv), progress=None)
        return 0

    registry_index = runner.pipeline.registry_index()
    backend = runner._build_backend(args.model, backend_map[args.model])
    header_written = records_csv.exists() and records_csv.stat().st_size > 0
    appended = 0
    for task in tasks_to_run:
        registry = registry_index[runner.pipeline.registry_key_for_sample(task)]
        response = runner._predict_with_retries(backend, task, registry.list_tools())
        record = runner.scorer.score(task, response, registry)
        if "backend request failed" in str(record.notes):
            print(
                json.dumps(
                    {
                        "family": family,
                        "stopped_on_backend_failure": True,
                        "sample_id": task.sample_id,
                        "notes": record.notes,
                        "appended": appended,
                    },
                    ensure_ascii=False,
                ),
                file=sys.stderr,
            )
            return 2
        runner._append_record(record, records_csv, write_header=not header_written)
        header_written = True
        appended += 1
        print(json.dumps({"family": family, "appended": appended, "sample_id": task.sample_id}, ensure_ascii=False))

    if args.refresh_outputs:
        runner._write_outputs(pd.read_csv(records_csv), progress=None)

    print(json.dumps({"family": family, "done": True, "appended": appended}, ensure_ascii=False))
    return 0


def main() -> int:
    args = parse_args()
    for family in split_csv(args.families):
        code = run_family(args, family)
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
