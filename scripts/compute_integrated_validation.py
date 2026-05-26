"""Compute integrated validation agreement only after human annotation is complete."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HUMAN_REQUIRED = [
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
]
LLM_REQUIRED = [field.replace("human_", "llm_") for field in HUMAN_REQUIRED]
COMPARISON_FIELDS = [
    ("success", "Task success"),
    ("observed_error", "Observed error"),
    ("primary_error", "Primary error"),
    ("earliest_stage", "Earliest stage"),
    ("boundary_violation", "Boundary violation"),
    ("status", "Status"),
    ("retryability", "Retryability"),
    ("failure_type", "Failure type"),
    ("final_result_correct", "Final result correctness"),
]
KAPPA_FIELDS = {
    "success",
    "observed_error",
    "primary_error",
    "earliest_stage",
    "boundary_violation",
    "status",
    "retryability",
    "failure_type",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-dir", type=Path, default=ROOT / "results/validation")
    return parser.parse_args()


def clean(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def yes_no_flag(value: object) -> str:
    text = clean(value).lower()
    if text in {"1", "true", "yes"}:
        return "yes"
    if text in {"0", "false", "no"}:
        return "no"
    return "unclear"


def normalized_failure(value: object) -> str:
    mapping = {
        "": "none",
        "no-call": "no_call",
        "wrong-tool": "tool_misselection",
        "missing-parameter": "missing_argument",
        "wrong-parameter": "incorrect_argument",
        "invalid-format": "schema_format_violation",
        "execution-fail": "runtime_failure",
    }
    return mapping.get(clean(value).lower(), "other")


def normalized_stage(value: object) -> str:
    text = clean(value)
    if not text:
        return "none"
    if text[:2] in {"S1", "S2", "S3", "S4", "S5"}:
        return text[:2]
    return "unclear"


def normalized_status(value: object) -> str:
    text = clean(value).lower()
    return {
        "": "not_applicable",
        "partial": "partial_success",
        "partial_success": "partial_success",
        "success": "success",
        "error": "error",
    }.get(text, "unclear")


def normalized_retryability(value: object) -> str:
    text = clean(value).lower()
    if not text:
        return "not_applicable"
    if text in {"true", "1", "yes", "retryable"}:
        return "retryable"
    if text in {"false", "0", "no", "non_retryable"}:
        return "non_retryable"
    return "unclear"


def normalized_failure_type(value: object, status: object) -> str:
    text = clean(value).lower()
    if not text:
        return "none" if normalized_status(status) == "success" else "not_applicable"
    if "timeout" in text or "tail_missing" in text or "truncated" in text:
        return "timeout"
    if "permission" in text or "auth" in text or "credential" in text:
        return "permission"
    if "unavailable" in text or "service_unavailable" in text or "dependency" in text:
        return "unavailable"
    if "argument" in text or "parameter" in text:
        return "invalid_argument"
    if "schema" in text or "parse" in text:
        return "schema_error"
    if text in {"none", "null"}:
        return "none"
    return "other"


def add_auto_fields(full: pd.DataFrame) -> pd.DataFrame:
    full = full.copy()
    full["auto_success"] = full["overall_task_success"].map(yes_no_flag)
    full["auto_observed_error"] = full["observed_failure_tag"].map(normalized_failure)
    full["auto_primary_error"] = full["primary_failure_tag"].map(normalized_failure)
    full["auto_earliest_stage"] = full["first_failure_stage"].map(normalized_stage)
    full["auto_boundary_violation"] = full["boundary_violation_flag"].map(yes_no_flag)
    full["auto_status"] = full["decision_status"].map(normalized_status)
    full["auto_retryability"] = full["decision_retryable"].map(normalized_retryability)
    full["auto_failure_type"] = [
        normalized_failure_type(value, status)
        for value, status in zip(full["decision_failure_type"], full["decision_status"], strict=False)
    ]
    full["auto_final_result_correct"] = [
        yes_no_flag(match) if clean(family) in {"observation", "execution_environment"} and clean(match) else "not_applicable"
        for family, match in zip(full["family"], full["stage2_result_match"], strict=False)
    ]
    return full


def missing_cells(frame: pd.DataFrame, fields: list[str]) -> dict[str, int]:
    return {
        field: int(frame[field].map(clean).eq("").sum()) if field in frame.columns else len(frame)
        for field in fields
    }


def write_missing_report(path: Path, human: pd.DataFrame, llm: pd.DataFrame, human_missing: dict[str, int], llm_missing: dict[str, int]) -> None:
    lines = [
        "# Integrated Validation Missing Annotation Report",
        "",
        f"- Human annotation rows: {len(human)}",
        f"- LLM annotation rows: {len(llm)}",
        "",
        "## Human Fields Requiring Completion",
        "",
    ]
    for field, count in human_missing.items():
        lines.append(f"- `{field}`: {count} blank row(s)")
    lines.extend(["", "## LLM Fields Missing", ""])
    for field, count in llm_missing.items():
        lines.append(f"- `{field}`: {count} blank row(s)")
    lines.extend(
        [
            "",
            "## Metric Generation Status",
            "",
            "- Metrics and LaTeX result tables are not generated while required human annotation cells are blank.",
            "- After completing the human annotation CSV, run `python scripts/compute_integrated_validation.py`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def kappa(left: Iterable[str], right: Iterable[str]) -> float | None:
    pairs = [(a, b) for a, b in zip(left, right, strict=False) if a and b and a != "unclear" and b != "unclear"]
    if not pairs:
        return None
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    left_counts = Counter(a for a, _ in pairs)
    right_counts = Counter(b for _, b in pairs)
    labels = set(left_counts) | set(right_counts)
    expected = sum(left_counts[label] * right_counts[label] for label in labels) / (n * n)
    if expected == 1:
        return 1.0 if observed == 1 else None
    return (observed - expected) / (1 - expected)


def agreement(left: pd.Series, right: pd.Series) -> tuple[int, float | None]:
    pairs = [(clean(a), clean(b)) for a, b in zip(left, right, strict=False)]
    valid = [(a, b) for a, b in pairs if a and b and a != "unclear" and b != "unclear"]
    if not valid:
        return 0, None
    return len(valid), sum(a == b for a, b in valid) / len(valid)


def display(value: float | None) -> str:
    return "--" if value is None else f"{value:.3f}"


def latex_escape(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%")


def write_confusion(path: Path, title: str, human: pd.Series, auto: pd.Series) -> None:
    pairs = [(clean(a), clean(b)) for a, b in zip(human, auto, strict=False) if clean(a) and clean(b) and clean(a) != "unclear"]
    labels = sorted({item for pair in pairs for item in pair})
    matrix = pd.crosstab(
        pd.Series([pair[0] for pair in pairs], name="Human"),
        pd.Series([pair[1] for pair in pairs], name="Scorer"),
    ).reindex(index=labels, columns=labels, fill_value=0)
    columns = "l" + "r" * len(labels)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        "Human / scorer & " + " & ".join(latex_escape(label) for label in labels) + r" \\",
        r"\midrule",
    ]
    for label in labels:
        lines.append(latex_escape(label) + " & " + " & ".join(str(int(matrix.loc[label, col])) for col in labels) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{Human--automatic scorer confusion matrix for {latex_escape(title)}.}}",
            r"\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    full_path = args.validation_dir / "integrated_validation_sample_full.csv"
    human_path = args.validation_dir / "integrated_validation_annotation_human.csv"
    llm_path = args.validation_dir / "integrated_validation_annotation_llm.csv"
    missing_path = args.validation_dir / "validation_missing_report.md"
    full = add_auto_fields(pd.read_csv(full_path, keep_default_na=False))
    human = pd.read_csv(human_path, keep_default_na=False)
    llm = pd.read_csv(llm_path, keep_default_na=False) if llm_path.exists() else pd.DataFrame({"record_id": full["record_id"]})
    human_missing = missing_cells(human, HUMAN_REQUIRED)
    llm_missing = missing_cells(llm, LLM_REQUIRED)
    write_missing_report(missing_path, human, llm, human_missing, llm_missing)
    if any(human_missing.values()):
        print(f"Human annotation incomplete; wrote missing report only: {missing_path}")
        for field, count in human_missing.items():
            if count:
                print(f"MISSING {field}: {count}")
        return

    joined = full.merge(human, on="record_id", how="inner").merge(llm, on="record_id", how="left")
    metric_rows: list[dict[str, object]] = []
    validity = [
        ("gold_tool_correct", "Gold tool correctness"),
        ("gold_arguments_correct", "Gold argument correctness"),
        ("expected_result_correct", "Expected result correctness"),
        ("subtype_correct", "Subtype validity"),
        ("severity_reasonable", "Severity reasonableness"),
        ("solvable", "Human solvability"),
        ("ambiguous", "Human ambiguity"),
    ]
    for field, label in validity:
        values = joined[f"human_{field}"].map(clean)
        valid = values.loc[values.isin(["yes", "no"])]
        metric_rows.append(
            {"comparison": "data_construction", "metric": label, "n": len(valid), "agreement_or_rate": (valid == "yes").mean() if len(valid) else None, "kappa": None}
        )
    pairs = [("human", "auto", "Human--scorer"), ("llm", "auto", "LLM--scorer"), ("human", "llm", "Human--LLM")]
    for left_prefix, right_prefix, comparison in pairs:
        for field, label in COMPARISON_FIELDS:
            left = joined[f"{left_prefix}_{field}"] if f"{left_prefix}_{field}" in joined else pd.Series(dtype=str)
            right = joined[f"{right_prefix}_{field}"] if f"{right_prefix}_{field}" in joined else pd.Series(dtype=str)
            n, value = agreement(left, right)
            metric_rows.append(
                {
                    "comparison": comparison,
                    "metric": label,
                    "n": n,
                    "agreement_or_rate": value,
                    "kappa": kappa(left.map(clean), right.map(clean)) if field in KAPPA_FIELDS else None,
                }
            )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.validation_dir / "integrated_validation_metrics.csv", index=False, encoding="utf-8-sig")
    table_rows = [
        ("Human--scorer", "Task success"),
        ("Human--scorer", "Observed error"),
        ("Human--scorer", "Primary error"),
        ("Human--scorer", "Status"),
        ("Human--scorer", "Retryability"),
        ("Human--scorer", "Failure type"),
        ("Human--LLM", "Task success"),
        ("Human--LLM", "Status"),
        ("Human--LLM", "Retryability"),
        ("Human--LLM", "Failure type"),
    ]
    table_lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Validation comparison & Label & Valid $n$ & Agreement / $\kappa$ \\",
        r"\midrule",
    ]
    for comparison, metric in table_rows:
        row = metrics.loc[(metrics["comparison"] == comparison) & (metrics["metric"] == metric)].iloc[0]
        table_lines.append(
            f"{comparison} & {metric} & {int(row['n'])} & "
            f"{display(row['agreement_or_rate'])} / {display(row['kappa'])} \\\\"
        )
        if comparison == "Human--scorer" and metric == "Failure type":
            table_lines.append(r"\midrule")
    table_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Integrated validation on 140 stratified human-audited records. Valid $n$ denotes the number of comparable records for each label after excluding \texttt{unclear} judgments and applying the prespecified \texttt{not\_applicable} convention for non-runtime decision fields.}",
            r"\label{tab:integrated-validation}",
            r"\end{table*}",
        ]
    )
    (args.validation_dir / "integrated_validation_table.tex").write_text("\n".join(table_lines) + "\n", encoding="utf-8")
    write_confusion(args.validation_dir / "confusion_task_success.tex", "task success", joined["human_success"], joined["auto_success"])
    write_confusion(args.validation_dir / "confusion_primary_error.tex", "primary error", joined["human_primary_error"], joined["auto_primary_error"])
    write_confusion(args.validation_dir / "confusion_earliest_stage.tex", "earliest failed stage", joined["human_earliest_stage"], joined["auto_earliest_stage"])
    print(f"Wrote completed integrated validation metrics for {len(joined)} annotated records.")


if __name__ == "__main__":
    main()
