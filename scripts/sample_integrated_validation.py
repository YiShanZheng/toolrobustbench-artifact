"""Sample a blind integrated validation set from ToolRobustBench raw records."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toolrobust.datasets.builders import DatasetBuilder  # noqa: E402
from toolrobust.domain.schemas import BenchmarkV3Config, TaskSample  # noqa: E402
from toolrobust.execution.runtime_feedback import run_perturbed_execution  # noqa: E402
from toolrobust.perturbation.pipeline import PerturbationPipeline  # noqa: E402
from toolrobust.tools.executor import ToolExecutor  # noqa: E402

SUBTYPE_ORDER = [
    "A1 Name Surface Shift",
    "A2 Description Abstraction",
    "A3 Family Similarity Pressure",
    "B1 Indirectness",
    "B2 Underspecification",
    "B3 Distraction",
    "B4 Conflict",
    "C1 Return Structure Noise",
    "C2 Return Evidence Loss",
    "C3 Return Evidence Corruption",
    "C4 Return Evidence Competition",
    "D1 Availability Failure",
    "D2 Runtime Variability",
    "D3 Failure Feedback Quality",
]
HUMAN_FIELDS = [
    "human_gold_tool_correct",
    "human_gold_arguments_correct",
    "human_expected_result_correct",
    "human_subtype_correct",
    "human_severity_reasonable",
    "human_solvable",
    "human_ambiguous",
    "human_success",
    "human_observed_error",
    "human_primary_error",
    "human_earliest_stage",
    "human_boundary_violation",
    "human_status",
    "human_retryability",
    "human_failure_type",
    "human_final_result_correct",
    "human_notes",
]
PUBLIC_FIELDS = [
    "record_id",
    "instance_id",
    "model",
    "family",
    "subtype",
    "severity",
    "operators",
    "user_request",
    "tool_registry_or_schema",
    "model_output",
    "tool_output_or_runtime_feedback",
    "gold_tool",
    "gold_arguments",
    "expected_result",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-root", type=Path, default=ROOT / "data/results/all-family")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/paper_all_family.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/validation")
    parser.add_argument("--per-subtype", type=int, default=10)
    return parser.parse_args()


def parse_object(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (dict, list, bool, int, float)):
        return value
    text = str(value).strip()
    if not text:
        return None
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(text)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            pass
    return text


def dumped(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def load_config(path: Path) -> BenchmarkV3Config:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return BenchmarkV3Config.model_validate(payload)


def build_task_and_registry_indexes(config: BenchmarkV3Config) -> tuple[dict[str, TaskSample], dict[str, Any]]:
    builder = DatasetBuilder(config)
    clean = builder._build_clean_samples(builder._build_template_pool())
    pipeline = PerturbationPipeline(config)
    perturbed = builder._build_perturbed_samples(clean, pipeline)
    tasks = {task.sample_id: task for task in clean + perturbed}
    registries = pipeline.registry_index()
    return tasks, registries


def load_single_family_records(records_root: Path) -> tuple[pd.DataFrame, list[str]]:
    paths = sorted(records_root.glob("*/*/records.csv"))
    if not paths:
        raise FileNotFoundError(f"No single-family records found under {records_root}")
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path, keep_default_na=False)
        frame["source_records_file"] = str(path.relative_to(ROOT))
        frames.append(frame)
    records = pd.concat(frames, ignore_index=True)
    records = records.loc[
        (records["severity"] != "clean") & records["subtype"].isin(SUBTYPE_ORDER)
    ].copy()
    records["model"] = records["model_name"]
    records["failure_record"] = records["overall_task_success"].astype(str).eq("0")
    return records, [str(path.relative_to(ROOT)) for path in paths]


def select_subtype_sample(group: pd.DataFrame, size: int) -> pd.DataFrame:
    selected: list[int] = []
    uncovered_severity = {"light", "medium", "heavy"}
    uncovered_outcome = {True, False}
    uncovered_models = set(group["model"].unique())
    uncovered_errors = {item for item in group["primary_failure_tag"].unique() if str(item).strip()}
    d_combos = set(
        zip(group["decision_status"], group["decision_retryable"], group["decision_failure_type"])
    ) if str(group.iloc[0]["subtype"]).startswith("D") else set()

    candidates = group.copy()
    candidates["_stable_key"] = (
        candidates["model"].astype(str) + "|" + candidates["sample_id"].astype(str)
    )
    while len(selected) < min(size, len(candidates)):
        best_idx: int | None = None
        best_score: tuple[int, str] | None = None
        failures_selected = sum(bool(candidates.loc[idx, "failure_record"]) for idx in selected)
        for idx, row in candidates.loc[~candidates.index.isin(selected)].iterrows():
            combo = (row["decision_status"], row["decision_retryable"], row["decision_failure_type"])
            score = 0
            score += 9 if row["severity"] in uncovered_severity else 0
            score += 5 if bool(row["failure_record"]) in uncovered_outcome else 0
            score += 4 if row["model"] in uncovered_models else 0
            score += 3 if row["primary_failure_tag"] in uncovered_errors else 0
            score += 3 if combo in d_combos else 0
            score += 4 if bool(row["failure_record"]) and failures_selected < max(1, size - 3) else 0
            candidate_score = (score, str(row["_stable_key"]))
            if best_score is None or candidate_score[0] > best_score[0] or (
                candidate_score[0] == best_score[0] and candidate_score[1] < best_score[1]
            ):
                best_score = candidate_score
                best_idx = idx
        assert best_idx is not None
        selected.append(best_idx)
        chosen = candidates.loc[best_idx]
        uncovered_severity.discard(chosen["severity"])
        uncovered_outcome.discard(bool(chosen["failure_record"]))
        uncovered_models.discard(chosen["model"])
        uncovered_errors.discard(chosen["primary_failure_tag"])
        d_combos.discard((chosen["decision_status"], chosen["decision_retryable"], chosen["decision_failure_type"]))
    return candidates.loc[selected].drop(columns=["_stable_key"])


def registry_payload(registry: Any) -> list[dict[str, Any]]:
    return [
        {
            "name": tool.exposed_name,
            "description": tool.description,
            "schema": tool.parameters.model_dump(mode="json"),
        }
        for tool in registry.list_tools()
    ]


def model_output_payload(row: pd.Series) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "tool_name": row["predicted_tool_raw"] or None,
        "arguments": parse_object(row["predicted_arguments"]) or {},
    }
    if str(row["family"]) == "execution_environment":
        payload["post_execution_decision"] = {
            "status": row["decision_status"] or None,
            "retryable": parse_object(row["decision_retryable"]),
            "failure_type": row["decision_failure_type"] or None,
            "final_result_summary": row["decision_final_result_summary"] or None,
        }
    if str(row["family"]) == "observation":
        payload["post_execution_result"] = (
            "not persisted in raw records; human/LLM should use unclear where this prevents judgment"
        )
    return payload


def execution_feedback(task: TaskSample, row: pd.Series, registry: Any) -> tuple[Any, list[str]]:
    warnings: list[str] = []
    predicted_tool = row["predicted_tool_resolved"] or None
    args = parse_object(row["predicted_arguments"]) or {}
    normalized = registry.normalize_arguments(predicted_tool, args) if predicted_tool else {}
    if task.family in {"observation", "execution_environment"}:
        feedback = run_perturbed_execution(task, predicted_tool, normalized).as_public_dict()
        if task.family == "observation":
            warnings.append("Observation records do not persist the raw post-execution model response.")
        return feedback, warnings
    if not predicted_tool:
        return {"execution_skipped": "no resolved tool call"}, warnings
    try:
        result = ToolExecutor().execute(predicted_tool, normalized)
        return {"deterministic_execution_result_for_model_call": result}, warnings
    except Exception as exc:  # a displayed failed execution is evidence, not a script failure.
        return {"execution_error_for_model_call": str(exc)}, warnings


def enrich_sample(sample: pd.DataFrame, tasks: dict[str, TaskSample], registries: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    enriched: list[dict[str, Any]] = []
    warnings: list[str] = []
    for n, (_, row) in enumerate(sample.iterrows(), start=1):
        instance_id = str(row["sample_id"])
        task = tasks.get(instance_id)
        if task is None:
            warnings.append(f"No reconstructed instance found for {instance_id}.")
        registry = registries.get(str(row["registry_variant_id"])) or registries.get("clean")
        record = row.to_dict()
        record["record_id"] = f"iv_{n:03d}"
        record["instance_id"] = instance_id
        record["model"] = row["model_name"]
        record["user_request"] = task.user_instruction if task else ""
        record["tool_registry_or_schema"] = dumped(registry_payload(registry)) if registry else ""
        record["model_output"] = dumped(model_output_payload(row))
        if task and registry:
            feedback, local_warnings = execution_feedback(task, row, registry)
            record["tool_output_or_runtime_feedback"] = dumped(feedback)
            warnings.extend(local_warnings)
        else:
            record["tool_output_or_runtime_feedback"] = ""
        record["gold_tool"] = row["gold_canonical_tool_id"]
        record["gold_arguments"] = dumped(parse_object(row["expected_arguments"]))
        record["expected_result"] = dumped(parse_object(row["expected_result"]))
        record["operators"] = row["operator"]
        enriched.append(record)
    return pd.DataFrame(enriched), sorted(set(warnings))


def write_report(
    path: Path,
    source_files: list[str],
    source: pd.DataFrame,
    sampled: pd.DataFrame,
    warnings: list[str],
    missing_columns: list[str],
) -> None:
    lines = [
        "# Integrated Validation Sampling Report",
        "",
        "## Source",
        "",
        f"- Single-family record files read: {len(source_files)}",
        f"- Eligible perturbed raw model-output records: {len(source):,}",
        f"- Selected validation records: {len(sampled):,}",
        "- Cross-family records were located but intentionally excluded from this A1--D3 stratified sample.",
        "",
        "## Selected Distribution",
        "",
        "### Family",
        "",
    ]
    for key, value in sampled["family"].value_counts().sort_index().items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "### Subtype", ""])
    for subtype in SUBTYPE_ORDER:
        lines.append(f"- `{subtype}`: {int((sampled['subtype'] == subtype).sum())}")
    lines.extend(["", "### Severity", ""])
    for key, value in sampled["severity"].value_counts().sort_index().items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "### Outcome (used only for sampling coverage; hidden from annotator inputs)", ""])
    outcome = sampled["overall_task_success"].astype(str).map({"1": "success", "0": "failure"}).value_counts()
    for key, value in outcome.items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Field Audit", ""])
    if missing_columns:
        lines.append("- Missing required raw fields: " + ", ".join(f"`{col}`" for col in missing_columns))
    else:
        lines.append("- Required raw scorer, gold-target, and grouping fields were found.")
    lines.extend(
        [
            "- Raw records contain parsed tool calls and selected post-execution decision fields, not original completion text.",
            "- For C/D records, public observation/runtime feedback is deterministically reconstructed from the task template, predicted call, and runtime perturbation implementation.",
            "- For family C, the raw post-execution model recovery JSON is not persisted; annotations may use `unclear` when that missing response prevents a judgment.",
            "",
            "## Warnings",
            "",
        ]
    )
    lines.extend([f"- {item}" for item in warnings] or ["- None."])
    lines.extend(["", "## Source Files", ""])
    lines.extend(f"- `{path_item}`" for path_item in source_files)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source, source_files = load_single_family_records(args.records_root)
    required_raw = [
        "sample_id", "model_name", "family", "subtype", "severity", "registry_variant_id",
        "gold_canonical_tool_id", "expected_arguments", "expected_result", "predicted_tool_raw",
        "predicted_arguments", "overall_task_success", "observed_failure_tag",
        "primary_failure_tag", "first_failure_stage", "boundary_violation_flag",
    ]
    missing_columns = [column for column in required_raw if column not in source.columns]
    if missing_columns:
        raise ValueError("Cannot sample: required raw columns missing: " + ", ".join(missing_columns))
    config = load_config(args.config)
    tasks, registries = build_task_and_registry_indexes(config)
    selections: list[pd.DataFrame] = []
    warnings: list[str] = []
    for subtype in SUBTYPE_ORDER:
        group = source.loc[source["subtype"] == subtype].copy()
        if len(group) < args.per_subtype:
            warnings.append(f"{subtype} has only {len(group)} eligible records; sampled all available.")
        selections.append(select_subtype_sample(group, args.per_subtype))
    sampled = pd.concat(selections, ignore_index=True)
    enriched, reconstruction_warnings = enrich_sample(sampled, tasks, registries)
    warnings.extend(reconstruction_warnings)

    full_path = args.output_dir / "integrated_validation_sample_full.csv"
    human_path = args.output_dir / "integrated_validation_annotation_human.csv"
    blind_path = args.output_dir / "integrated_validation_llm_input.jsonl"
    report_path = args.output_dir / "validation_sampling_report.md"
    enriched.to_csv(full_path, index=False, encoding="utf-8-sig")
    human = enriched[PUBLIC_FIELDS].copy()
    for field in HUMAN_FIELDS:
        human[field] = ""
    human.to_csv(human_path, index=False, encoding="utf-8-sig")
    with blind_path.open("w", encoding="utf-8") as handle:
        for record in enriched[PUBLIC_FIELDS].to_dict(orient="records"):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_report(report_path, source_files, source, enriched, sorted(set(warnings)), missing_columns)
    print(f"Selected {len(enriched)} records from {len(source)} eligible single-family perturbed records.")
    print(f"Wrote {full_path}")
    print(f"Wrote {human_path}")
    print(f"Wrote {blind_path}")
    print(f"Wrote {report_path}")
    for warning in sorted(set(warnings)):
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
