"""Clean contaminated records and optionally resume the affected runs.

This is a reproducible repair tool for benchmark result folders. It removes
non-capability failures from ``records.csv`` files and then reruns the missing
sample ids with the same generated run config.

Contamination classes:
- ``api``: the model call did not complete successfully.
- ``json``: C/D stage-2 output was empty, truncated, or not valid JSON.

For JSON compatibility repairs, the script raises ``stage2_max_tokens`` only in
memory during the repair run. No YAML/config file is modified, so the default
token limit is restored automatically when the process exits.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
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


DEFAULT_FAMILIES = [
    "interface_exposure",
    "user_intent",
    "observation",
    "execution_environment",
]

JSON_COMPAT_PATTERNS = {
    "empty observation result content": re.compile(r"empty observation result content", re.I),
    "observation content is not valid json object": re.compile(r"observation content is not valid json object", re.I),
    "empty execution decision content": re.compile(r"empty execution decision content", re.I),
    "execution decision is not valid json object": re.compile(r"execution decision is not valid json object", re.I),
    "finish_reason=length": re.compile(r"finish_reason\s*=\s*length", re.I),
}

API_FAILURE_PATTERNS = {
    "backend request failed": re.compile(r"backend request failed", re.I),
    "rate limit / quota": re.compile(
        r"RateLimitError|429|rate limit|too many requests|quota|insufficient balance|resource has been exhausted",
        re.I,
    ),
    "connection / server error": re.compile(
        r"APIConnectionError|APITimeoutError|InternalServerError|ServiceUnavailable|BadGateway|GatewayTimeout",
        re.I,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="data/results/all-family", type=Path)
    parser.add_argument("--config-dir", default="data/generated/run_configs/all-family", type=Path)
    parser.add_argument("--base-config", default="configs/paper_all_family.yaml", type=Path)
    parser.add_argument("--models-config", default="configs/models.yaml", type=Path)
    parser.add_argument("--models", default="", help="Comma-separated model names. Empty means all discovered models.")
    parser.add_argument("--families", default="", help="Comma-separated families. Empty means all discovered families.")
    parser.add_argument("--issues", choices=("all", "api", "json"), default="all")
    parser.add_argument("--records-name", default="records.csv")
    parser.add_argument("--json-stage2-max-tokens", type=int, default=4096)
    parser.add_argument("--apply", action="store_true", help="Actually rewrite records.csv. Without this, dry-run only.")
    parser.add_argument("--rerun", action="store_true", help="Rerun missing sample ids after cleanup.")
    parser.add_argument("--resume-incomplete", action="store_true", help="Also run pre-existing missing sample ids when no contaminated row was removed.")
    parser.add_argument("--cleanup-only", action="store_true", help="Alias for --apply without rerun; kept for clarity.")
    parser.add_argument("--refresh-outputs", action="store_true", help="Regenerate summaries/plots after cleanup or rerun.")
    parser.add_argument("--generate-missing-config", action="store_true", default=True)
    parser.add_argument("--stop-on-new-contamination", action="store_true", default=True)
    return parser.parse_args()


def split_csv(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


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


def infer_model_family(records_csv: Path, root: Path) -> tuple[str, str]:
    full_parts = records_csv.parts
    for marker in ("all-family", "cross-family"):
        if marker in full_parts:
            index = full_parts.index(marker)
            if index + 2 < len(full_parts):
                return full_parts[index + 1], full_parts[index + 2]
    try:
        relative = records_csv.relative_to(root)
    except ValueError:
        relative = records_csv
    parts = relative.parts
    if len(parts) >= 3:
        return parts[-3], parts[-2]
    return records_csv.parent.parent.name, records_csv.parent.name


def discover_records(args: argparse.Namespace) -> list[tuple[str, str, Path]]:
    selected_models = split_csv(args.models)
    selected_families = split_csv(args.families)
    records_files = sorted(args.results_root.rglob(args.records_name))
    records: list[tuple[str, str, Path]] = []
    for path in records_files:
        model, family = infer_model_family(path, args.results_root)
        if selected_models and model not in selected_models:
            continue
        if selected_families and family not in selected_families:
            continue
        records.append((model, family, path))
    return records


def issue_patterns(issue_type: str) -> dict[str, re.Pattern[str]]:
    if issue_type == "api":
        return API_FAILURE_PATTERNS
    if issue_type == "json":
        return JSON_COMPAT_PATTERNS
    return {**API_FAILURE_PATTERNS, **JSON_COMPAT_PATTERNS}


def classify_notes(notes: str) -> tuple[list[str], list[str]]:
    api = [name for name, pattern in API_FAILURE_PATTERNS.items() if pattern.search(notes)]
    json_compat = [name for name, pattern in JSON_COMPAT_PATTERNS.items() if pattern.search(notes)]
    return api, json_compat


def contamination_mask(frame: pd.DataFrame, issue_type: str) -> tuple[pd.Series, dict[str, int], dict[str, int]]:
    if frame.empty or "notes" not in frame.columns:
        return pd.Series(False, index=frame.index), {}, {}
    enabled = issue_patterns(issue_type)
    mask = pd.Series(False, index=frame.index)
    api_counts: dict[str, int] = {}
    json_counts: dict[str, int] = {}
    for index, notes_value in frame["notes"].fillna("").astype(str).items():
        notes = str(notes_value)
        matched = False
        for reason, pattern in enabled.items():
            if not pattern.search(notes):
                continue
            matched = True
            if reason in API_FAILURE_PATTERNS:
                api_counts[reason] = api_counts.get(reason, 0) + 1
            if reason in JSON_COMPAT_PATTERNS:
                json_counts[reason] = json_counts.get(reason, 0) + 1
        if matched:
            mask.loc[index] = True
    return mask, api_counts, json_counts


def ensure_run_config(args: argparse.Namespace, model: str, family: str) -> Path:
    model_slug = slug(model)
    config_path = args.config_dir / f"paper_all_family_{model_slug}_{family}.yaml"
    if config_path.exists():
        return config_path
    if not args.generate_missing_config:
        raise FileNotFoundError(f"missing generated config: {config_path}")
    payload = load_yaml(args.base_config)
    payload.setdefault("experiment", {})["name"] = f"paper_all_family_{model_slug}_{family}"
    payload.setdefault("models", {})["include"] = [model]
    payload.setdefault("noise", {})["enabled_families"] = [family]
    payload.setdefault("cross_family", {})["enabled"] = False
    write_yaml(config_path, payload)
    return config_path


def format_counts(counts: dict[str, int]) -> str:
    return "; ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def build_runner(config_path: Path, models_config: Path, model: str, results_dir: Path) -> tuple[ExperimentRunner, dict[str, Any], dict[str, str]]:
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
    return runner, backend_configs, backend_map


def completed_ids(frame: pd.DataFrame, model: str, task_ids: set[str]) -> set[str]:
    if frame.empty or "model_name" not in frame.columns or "sample_id" not in frame.columns:
        return set()
    return set(
        frame.loc[
            frame["model_name"].astype(str).eq(model)
            & frame["sample_id"].astype(str).isin(task_ids),
            "sample_id",
        ].astype(str)
    )


def rerun_missing(
    *,
    runner: ExperimentRunner,
    backend_configs: dict[str, Any],
    backend_map: dict[str, str],
    model: str,
    records_csv: Path,
    tasks_to_run: list[Any],
    use_json_token_override: bool,
    json_stage2_max_tokens: int,
    stop_on_new_contamination: bool,
    issue_type: str,
) -> tuple[int, int]:
    config = backend_configs[model]
    original_stage2_max_tokens = config.stage2_max_tokens
    if use_json_token_override:
        configured_max = original_stage2_max_tokens if original_stage2_max_tokens is not None else config.max_tokens
        config.stage2_max_tokens = max(int(configured_max), int(json_stage2_max_tokens))
    appended = 0
    try:
        registry_index = runner.pipeline.registry_index()
        backend = runner._build_backend(model, backend_map[model])
        header_written = records_csv.exists() and records_csv.stat().st_size > 0
        for task in tasks_to_run:
            registry = registry_index[runner.pipeline.registry_key_for_sample(task)]
            response = runner._predict_with_retries(backend, task, registry.list_tools())
            record = runner.scorer.score(task, response, registry)
            new_frame = pd.DataFrame([record.model_dump(mode="json")])
            new_mask, new_api, new_json = contamination_mask(new_frame, issue_type)
            if stop_on_new_contamination and bool(new_mask.any()):
                print(
                    json.dumps(
                        {
                            "stopped_on_new_contamination": True,
                            "model": model,
                            "sample_id": task.sample_id,
                            "api_reasons": new_api,
                            "json_reasons": new_json,
                            "notes": record.notes,
                            "appended": appended,
                        },
                        ensure_ascii=False,
                    ),
                    file=sys.stderr,
                )
                return appended, 2
            runner._append_record(record, records_csv, write_header=not header_written)
            header_written = True
            appended += 1
            print(json.dumps({"model": model, "appended": appended, "sample_id": task.sample_id}, ensure_ascii=False))
    finally:
        config.stage2_max_tokens = original_stage2_max_tokens
    return appended, 0


def repair_one(args: argparse.Namespace, model: str, family: str, records_csv: Path) -> int:
    results_dir = records_csv.parent
    config_path = ensure_run_config(args, model, family)
    runner, backend_configs, backend_map = build_runner(config_path, args.models_config, model, results_dir)
    clean_tasks, perturbed_tasks = runner.build_data(progress=None)
    tasks = clean_tasks + perturbed_tasks
    task_ids = {task.sample_id for task in tasks}

    existing = pd.read_csv(records_csv) if records_csv.exists() else pd.DataFrame()
    if not existing.empty:
        relevant = (
            existing.get("model_name", pd.Series("", index=existing.index)).astype(str).eq(model)
            & existing.get("sample_id", pd.Series("", index=existing.index)).astype(str).isin(task_ids)
        )
        issue_mask, _, _ = contamination_mask(existing, args.issues)
        remove_mask = relevant & issue_mask
        _, api_counts, json_counts = contamination_mask(existing.loc[remove_mask], args.issues)
    else:
        remove_mask = pd.Series(False, index=existing.index)
        api_counts = {}
        json_counts = {}
    removed = int(remove_mask.sum()) if len(remove_mask) else 0
    cleaned = existing.loc[~remove_mask].copy() if not existing.empty else existing
    completed = completed_ids(cleaned, model, task_ids)
    missing_tasks = [task for task in tasks if task.sample_id not in completed]
    tasks_to_run = missing_tasks if (removed > 0 or args.resume_incomplete) else []
    report = {
        "model": model,
        "family": family,
        "records": str(records_csv),
        "config": str(config_path),
        "records_before": int(len(existing)),
        "contaminated_rows_to_remove": removed,
        "api_reasons": api_counts,
        "json_reasons": json_counts,
        "records_after_cleanup": int(len(cleaned)),
        "total_tasks": len(tasks),
        "already_completed_after_cleanup": len(completed),
        "missing_after_cleanup": len(missing_tasks),
        "to_run": len(tasks_to_run),
        "json_stage2_token_override": bool(json_counts and removed),
        "json_stage2_max_tokens": args.json_stage2_max_tokens if json_counts and removed else None,
        "dry_run": not args.apply,
    }
    print(json.dumps(report, ensure_ascii=False))
    if not args.apply:
        return 0
    if removed == 0 and not args.resume_incomplete:
        print(json.dumps({"model": model, "family": family, "skipped": "no_contaminated_rows"}, ensure_ascii=False))
        return 0

    results_dir.mkdir(parents=True, exist_ok=True)
    backup = None
    if removed > 0 and records_csv.exists():
        backup = records_csv.with_name(f"{records_csv.name}.bak_before_contamination_repair_{int(time.time())}")
        shutil.copy2(records_csv, backup)
    if removed > 0:
        cleaned.to_csv(records_csv, index=False)
    print(json.dumps({"model": model, "family": family, "backup": str(backup) if backup else None, "removed": removed}, ensure_ascii=False))

    if args.cleanup_only or not args.rerun or not tasks_to_run:
        if args.refresh_outputs and records_csv.exists() and records_csv.stat().st_size > 0:
            runner._write_outputs(pd.read_csv(records_csv), progress=None)
        return 0

    appended, code = rerun_missing(
        runner=runner,
        backend_configs=backend_configs,
        backend_map=backend_map,
        model=model,
        records_csv=records_csv,
        tasks_to_run=tasks_to_run,
        use_json_token_override=bool(json_counts and removed),
        json_stage2_max_tokens=args.json_stage2_max_tokens,
        stop_on_new_contamination=args.stop_on_new_contamination,
        issue_type=args.issues,
    )
    if args.refresh_outputs and records_csv.exists() and records_csv.stat().st_size > 0:
        runner._write_outputs(pd.read_csv(records_csv), progress=None)
    print(json.dumps({"model": model, "family": family, "done": code == 0, "appended": appended}, ensure_ascii=False))
    return code


def main() -> int:
    args = parse_args()
    if args.cleanup_only:
        args.apply = True
        args.rerun = False
    records = discover_records(args)
    if not records:
        print(json.dumps({"found": 0, "message": "no records.csv files matched the selection"}, ensure_ascii=False))
        return 0
    max_code = 0
    for model, family, records_csv in records:
        code = repair_one(args, model, family, records_csv)
        max_code = max(max_code, code)
        if code != 0:
            return code
    return max_code


if __name__ == "__main__":
    raise SystemExit(main())
