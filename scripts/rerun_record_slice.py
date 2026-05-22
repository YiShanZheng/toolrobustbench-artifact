"""Rerun a narrow records.csv slice without replaying a whole family.

This is intentionally surgical: it removes failed rows for the requested
model/subtype/severity, runs only missing rows in that slice, and appends only
successful backend responses. If the backend returns a request failure, the
script stops before writing that failed record.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--models-config", default="configs/models.yaml", type=Path)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--subtype", required=True)
    parser.add_argument("--severity", required=True)
    parser.add_argument("--cleanup-only", action="store_true")
    parser.add_argument("--refresh-outputs", action="store_true")
    return parser.parse_args()


def is_backend_failed(row: pd.Series) -> bool:
    notes = str(row.get("notes", ""))
    return "backend request failed" in notes


def main() -> int:
    args = parse_args()
    records_csv = args.results_dir / "records.csv"
    if not records_csv.exists():
        raise FileNotFoundError(records_csv)

    config, backend_configs, backend_map = resolve_configs(args.config, args.models_config)
    if args.model not in backend_map:
        raise ValueError(f"model {args.model!r} is not present in {args.config}")

    runner = ExperimentRunner(
        config,
        backend_configs,
        backend_map,
        output_profile=OutputProfile(config.output.profile),
        clean_output=False,
    )
    runner.output_paths = runner._build_output_paths(explicit_results_dir=args.results_dir)
    runner.manifest = runner._build_manifest()

    clean_tasks, perturbed_tasks = runner.build_data(progress=None)
    all_tasks = clean_tasks + perturbed_tasks
    selected = [
        task
        for task in all_tasks
        if task.subtype == args.subtype and task.severity.value == args.severity
    ]
    if not selected:
        raise ValueError(f"no tasks matched subtype={args.subtype!r} severity={args.severity!r}")
    selected_ids = {task.sample_id for task in selected}

    existing = pd.read_csv(records_csv)
    backup = records_csv.with_name(f"{records_csv.name}.bak_before_slice_rerun_{int(time.time())}")
    shutil.copy2(records_csv, backup)

    def matching_slice(frame: pd.DataFrame) -> pd.Series:
        return (
            (frame["model_name"].astype(str) == args.model)
            & (frame["sample_id"].astype(str).isin(selected_ids))
            & (frame["subtype"].astype(str) == args.subtype)
            & (frame["severity"].astype(str) == args.severity)
        )

    slice_mask = matching_slice(existing)
    failed_mask = slice_mask & existing.apply(is_backend_failed, axis=1)
    cleaned = existing.loc[~failed_mask].copy()
    # Rewrite with Python UTF-8 (no PowerShell BOM issue) even if there were no
    # failed rows. This keeps resume/count tooling able to read sample_id.
    cleaned.to_csv(records_csv, index=False)

    completed_ids = set(
        cleaned.loc[
            (cleaned["model_name"].astype(str) == args.model)
            & (cleaned["sample_id"].astype(str).isin(selected_ids)),
            "sample_id",
        ].astype(str)
    )
    tasks_to_run = [task for task in selected if task.sample_id not in completed_ids]

    print(
        json.dumps(
            {
                "records_before": int(len(existing)),
                "failed_rows_removed": int(failed_mask.sum()),
                "records_after_cleanup": int(len(cleaned)),
                "selected_tasks": len(selected),
                "already_completed": len(completed_ids),
                "to_run": len(tasks_to_run),
                "backup": str(backup),
            },
            ensure_ascii=False,
        )
    )

    if not tasks_to_run:
        return 0
    if args.cleanup_only:
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
        print(json.dumps({"appended": appended, "sample_id": task.sample_id}, ensure_ascii=False))

    if args.refresh_outputs:
        records_frame = pd.read_csv(records_csv)
        runner._write_outputs(records_frame, progress=None)

    print(json.dumps({"done": True, "appended": appended}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
