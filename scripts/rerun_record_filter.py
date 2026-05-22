"""Remove and rerun records matching a simple row filter.

This is for surgical reproducibility repairs, for example rerunning all
``format_valid=0`` rows for one model/family without replaying the whole
benchmark. By default the script is a dry run. Add ``--apply`` to rewrite
``records.csv`` and ``--rerun`` to execute the removed sample ids again.

Supported filters are simple column comparisons:
- ``--where format_valid=0``
- ``--where secondary_reason=no_tool_call``
- ``--where severity in medium,heavy``
- repeat ``--where`` to AND conditions together.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from toolrobust.cli.main import resolve_configs
from toolrobust.domain.enums import OutputProfile
from toolrobust.runner.experiment_runner import ExperimentRunner


BACKEND_FAILURE_RE = re.compile(
    r"backend request failed|RateLimitError|429|rate limit|too many requests|quota|"
    r"insufficient balance|resource has been exhausted|APIConnectionError|"
    r"APITimeoutError|InternalServerError|ServiceUnavailable|BadGateway|GatewayTimeout",
    re.I,
)


@dataclass(frozen=True, slots=True)
class RowCondition:
    column: str
    operator: str
    values: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="data/results/all-family", type=Path)
    parser.add_argument("--config-dir", default="data/generated/run_configs/all-family", type=Path)
    parser.add_argument("--base-config", default="configs/paper_all_family.yaml", type=Path)
    parser.add_argument("--models-config", default="configs/models.yaml", type=Path)
    parser.add_argument("--models", required=True, help="Comma-separated model names.")
    parser.add_argument("--families", required=True, help="Comma-separated family names.")
    parser.add_argument(
        "--where",
        action="append",
        default=[],
        help="Row filter. Examples: 'format_valid=0', 'secondary_reason=no_tool_call', 'severity in medium,heavy'.",
    )
    parser.add_argument("--records-name", default="records.csv")
    parser.add_argument("--apply", action="store_true", help="Rewrite records.csv after backing it up.")
    parser.add_argument("--rerun", action="store_true", help="Rerun removed sample ids and append new records.")
    parser.add_argument("--refresh-outputs", action="store_true", help="Regenerate summaries/plots after cleanup or rerun.")
    parser.add_argument("--generate-missing-config", action="store_true", default=True)
    parser.add_argument(
        "--stop-on-backend-failure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop before appending a new row if the rerun hits API/backend failure.",
    )
    return parser.parse_args()


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def slug(value: str) -> str:
    for char in '\\/:*?"<>| \t\r\n':
        value = value.replace(char, "_")
    return value


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)


def parse_condition(raw: str) -> RowCondition:
    text = raw.strip()
    in_match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\s+in\s+(.+)", text)
    if in_match:
        values = tuple(split_csv(in_match.group(2)))
        if not values:
            raise ValueError(f"empty value list in --where {raw!r}")
        return RowCondition(in_match.group(1), "in", values)
    for operator in ("!=", "="):
        if operator in text:
            column, value = text.split(operator, 1)
            column = column.strip()
            value = value.strip()
            if not column:
                raise ValueError(f"missing column in --where {raw!r}")
            return RowCondition(column, operator, (value,))
    raise ValueError(f"unsupported --where expression {raw!r}")


def normalize_value(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def condition_mask(frame: pd.DataFrame, condition: RowCondition) -> pd.Series:
    if condition.column not in frame.columns:
        raise ValueError(f"column {condition.column!r} not found in records.csv")
    values = frame[condition.column].map(normalize_value)
    expected = tuple(str(value).strip() for value in condition.values)
    if condition.operator == "=":
        return values.eq(expected[0])
    if condition.operator == "!=":
        return ~values.eq(expected[0])
    if condition.operator == "in":
        return values.isin(expected)
    raise ValueError(f"unsupported operator {condition.operator!r}")


def combined_mask(frame: pd.DataFrame, conditions: list[RowCondition]) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for condition in conditions:
        mask &= condition_mask(frame, condition)
    return mask


def ensure_run_config(args: argparse.Namespace, model: str, family: str) -> Path:
    config_path = args.config_dir / f"paper_all_family_{slug(model)}_{family}.yaml"
    if config_path.exists():
        return config_path
    if not args.generate_missing_config:
        raise FileNotFoundError(f"missing generated config: {config_path}")
    payload = load_yaml(args.base_config)
    payload.setdefault("experiment", {})["name"] = f"paper_all_family_{slug(model)}_{family}"
    payload.setdefault("models", {})["include"] = [model]
    payload.setdefault("noise", {})["enabled_families"] = [family]
    payload.setdefault("cross_family", {})["enabled"] = False
    write_yaml(config_path, payload)
    return config_path


def build_runner(config_path: Path, models_config: Path, model: str, results_dir: Path) -> tuple[ExperimentRunner, dict[str, str]]:
    config, backend_configs, backend_map = resolve_configs(config_path, models_config)
    if model not in backend_map:
        raise ValueError(f"model {model!r} is not included by {config_path}")
    runner = ExperimentRunner(
        config,
        backend_configs,
        backend_map,
        output_profile=OutputProfile(config.output.profile),
        clean_output=False,
    )
    runner.output_paths = runner._build_output_paths(explicit_results_dir=results_dir)
    runner.manifest = runner._build_manifest()
    return runner, backend_map


def is_backend_failure(notes: object) -> bool:
    return bool(BACKEND_FAILURE_RE.search(str(notes or "")))


def write_removed_rows(results_dir: Path, removed: pd.DataFrame) -> Path:
    path = results_dir / f"removed_rows_before_filter_rerun_{int(time.time())}.csv"
    removed.to_csv(path, index=False)
    return path


@contextmanager
def record_lock(results_dir: Path):
    lock_path = results_dir / "records.csv.filter_rerun.lock"
    fd: int | None = None
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        payload = json.dumps({"pid": os.getpid(), "created_at": time.time()}, ensure_ascii=False)
        os.write(fd, payload.encode("utf-8"))
        os.close(fd)
        fd = None
        yield
    except FileExistsError as exc:
        raise RuntimeError(
            f"lock exists: {lock_path}. Another filter rerun may be active. "
            "If you are certain it is stale, remove the lock file manually."
        ) from exc
    finally:
        if fd is not None:
            os.close(fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _repair_one_unlocked(args: argparse.Namespace, model: str, family: str, conditions: list[RowCondition]) -> int:
    results_dir = args.results_root / model / family
    records_csv = results_dir / args.records_name
    frame = pd.read_csv(records_csv)
    if "model_name" not in frame.columns or "sample_id" not in frame.columns:
        raise ValueError(f"{records_csv} is missing model_name or sample_id")
    model_mask = frame["model_name"].map(normalize_value).eq(model)
    remove_mask = model_mask & combined_mask(frame, conditions)
    removed = frame.loc[remove_mask].copy()
    removed_ids = list(dict.fromkeys(removed["sample_id"].map(normalize_value).tolist()))
    cleaned = frame.loc[~remove_mask].copy()

    config_path = ensure_run_config(args, model, family)
    runner, backend_map = build_runner(config_path, args.models_config, model, results_dir)
    clean_tasks, perturbed_tasks = runner.build_data(progress=None)
    task_by_id = {task.sample_id: task for task in clean_tasks + perturbed_tasks}
    tasks_to_run = [task_by_id[sample_id] for sample_id in removed_ids if sample_id in task_by_id]
    missing_task_ids = [sample_id for sample_id in removed_ids if sample_id not in task_by_id]

    report = {
        "model": model,
        "family": family,
        "records": str(records_csv),
        "config": str(config_path),
        "records_before": int(len(frame)),
        "rows_matched_for_removal": int(len(removed)),
        "unique_sample_ids_matched": len(removed_ids),
        "records_after_cleanup": int(len(cleaned)),
        "tasks_to_rerun": len(tasks_to_run),
        "missing_task_ids": missing_task_ids,
        "dry_run": not args.apply,
    }
    print(json.dumps(report, ensure_ascii=False))
    if not args.apply:
        return 0

    backup = records_csv.with_name(f"{records_csv.name}.bak_before_filter_rerun_{int(time.time())}")
    shutil.copy2(records_csv, backup)
    removed_rows_path = write_removed_rows(results_dir, removed)
    cleaned.to_csv(records_csv, index=False)
    print(
        json.dumps(
            {
                "model": model,
                "family": family,
                "backup": str(backup),
                "removed_rows_csv": str(removed_rows_path),
                "removed": int(len(removed)),
            },
            ensure_ascii=False,
        )
    )

    if not args.rerun or not tasks_to_run:
        if args.refresh_outputs:
            runner._write_outputs(pd.read_csv(records_csv), progress=None)
        return 0

    registry_index = runner.pipeline.registry_index()
    backend = runner._build_backend(model, backend_map[model])
    header_written = records_csv.exists() and records_csv.stat().st_size > 0
    appended = 0
    for task in tasks_to_run:
        registry = registry_index[runner.pipeline.registry_key_for_sample(task)]
        response = runner._predict_with_retries(backend, task, registry.list_tools())
        record = runner.scorer.score(task, response, registry)
        if args.stop_on_backend_failure and is_backend_failure(record.notes):
            print(
                json.dumps(
                    {
                        "stopped_on_backend_failure": True,
                        "model": model,
                        "family": family,
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
        print(json.dumps({"model": model, "family": family, "appended": appended, "sample_id": task.sample_id}, ensure_ascii=False))

    if args.refresh_outputs:
        runner._write_outputs(pd.read_csv(records_csv), progress=None)

    print(json.dumps({"model": model, "family": family, "done": True, "appended": appended}, ensure_ascii=False))
    return 0


def repair_one(args: argparse.Namespace, model: str, family: str, conditions: list[RowCondition]) -> int:
    results_dir = args.results_root / model / family
    records_csv = results_dir / args.records_name
    if not records_csv.exists():
        raise FileNotFoundError(records_csv)
    if not args.apply:
        return _repair_one_unlocked(args, model, family, conditions)
    with record_lock(results_dir):
        return _repair_one_unlocked(args, model, family, conditions)


def main() -> int:
    args = parse_args()
    if not args.where:
        raise ValueError("at least one --where condition is required")
    conditions = [parse_condition(raw) for raw in args.where]
    max_code = 0
    for model in split_csv(args.models):
        for family in split_csv(args.families):
            code = repair_one(args, model, family, conditions)
            max_code = max(max_code, code)
            if code != 0:
                return code
    return max_code


if __name__ == "__main__":
    raise SystemExit(main())
