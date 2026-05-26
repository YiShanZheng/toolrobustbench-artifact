"""Run blind LLM-as-annotator labeling for the integrated validation sample."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
LLM_FIELDS = [
    "llm_gold_tool_correct",
    "llm_gold_arguments_correct",
    "llm_expected_result_correct",
    "llm_subtype_correct",
    "llm_severity_reasonable",
    "llm_solvable",
    "llm_ambiguous",
    "llm_success",
    "llm_observed_error",
    "llm_primary_error",
    "llm_earliest_stage",
    "llm_boundary_violation",
    "llm_status",
    "llm_retryability",
    "llm_failure_type",
    "llm_final_result_correct",
    "llm_notes",
]
PROMPT = """You are independently annotating a ToolRobustBench validation record in a blind setting.
Do not infer or request automatic scorer labels or human labels. Use only the supplied record evidence
and definitions. Return one JSON object with exactly the requested annotation fields.

Data construction validity labels:
- gold_tool_correct: yes/no/unclear. Is the gold tool the tool that should satisfy the user request?
- gold_arguments_correct: yes/no/unclear. Do gold arguments match the user request and tool schema?
- expected_result_correct: yes/no/unclear. Can the expected result be supported by the tool output or runtime feedback?
- subtype_correct: yes/no/unclear. Does the perturbation fit the specified family/subtype definition?
- severity_reasonable: yes/no/unclear. Is the light/medium/heavy level reasonable for the visible perturbation?
- solvable: yes/no/unclear. Does an ideal tool-using agent have enough information to complete or judge the task?
- ambiguous: yes/no/unclear. Are there multiple reasonable gold tool, argument, or result targets?

Scorer-validity labels:
- success: yes/no/unclear. Did the model complete the task?
- observed_error: none/no_call/tool_misselection/missing_argument/incorrect_argument/schema_format_violation/runtime_failure/other/unclear.
- primary_error: none/no_call/tool_misselection/missing_argument/incorrect_argument/schema_format_violation/runtime_failure/other/unclear.
- earliest_stage: none/S1/S2/S3/S4/S5/unclear.
- boundary_violation: yes/no/unclear. Did the principal failure originate outside the stage boundary intended by the stated perturbation family?
- status: success/partial_success/error/not_applicable/unclear.
- retryability: retryable/non_retryable/not_applicable/unclear.
- failure_type: timeout/permission/unavailable/invalid_argument/schema_error/tool_error/other/none/not_applicable/unclear.
- final_result_correct: yes/no/not_applicable/unclear.

Capability stages:
- S1 Tool Selection: whether the correct tool is selected.
- S2 Schema Grounding: whether tool names, descriptions, and schemas are understood.
- S3 Argument Binding: whether user intent is bound to correct structured arguments.
- S4 Tool Output / Runtime Feedback Handling: whether returned evidence or runtime feedback is interpreted correctly.
- S5 End-to-End Task Success: whether the full task succeeds.

Subtype definitions:
- A1 Name Surface Shift: modifies exposed tool names/identifiers while preserving canonical function.
- A2 Description Abstraction: makes descriptions less directly matched while retaining coarse function.
- A3 Family Similarity Pressure: increases similarity among sibling tool names or descriptions.
- B1 Indirectness: expresses the same task indirectly or distributes intent cues.
- B2 Underspecification: weakens explicit slots or constraints while retaining a designated recoverable target.
- B3 Distraction: inserts irrelevant or competing contextual cues.
- B4 Conflict: inserts contradictory intent cues or a late revision with a designated primary target.
- C1 Return Structure Noise: changes returned-content format, keys, or nesting.
- C2 Return Evidence Loss: removes or hides supporting returned evidence.
- C3 Return Evidence Corruption: introduces stale, inconsistent, or misleading returned values.
- C4 Return Evidence Competition: supplies competing returned candidates or evidence fragments.
- D1 Availability Failure: changes availability or exposes fallback/unavailable/timeout execution feedback.
- D2 Runtime Variability: introduces stale/fresh conflicts, partial results, or multiple attempts.
- D3 Failure Feedback Quality: varies completeness or reliability of failure feedback.

Important evidence rule: when the record states that an original post-execution model result was not
persisted, use "unclear" for judgments that cannot be established from visible evidence. Do not fill in
missing behavior by assumption.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "results/validation/integrated_validation_llm_input.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/validation")
    parser.add_argument("--model", default=os.getenv("VALIDATION_LLM_MODEL", "gpt-5.4-mini"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url-env", default="OPENAI_BASE_URL")
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def output_schema() -> dict[str, Any]:
    enumerated = {
        "llm_gold_tool_correct": ["yes", "no", "unclear"],
        "llm_gold_arguments_correct": ["yes", "no", "unclear"],
        "llm_expected_result_correct": ["yes", "no", "unclear"],
        "llm_subtype_correct": ["yes", "no", "unclear"],
        "llm_severity_reasonable": ["yes", "no", "unclear"],
        "llm_solvable": ["yes", "no", "unclear"],
        "llm_ambiguous": ["yes", "no", "unclear"],
        "llm_success": ["yes", "no", "unclear"],
        "llm_observed_error": [
            "none", "no_call", "tool_misselection", "missing_argument", "incorrect_argument",
            "schema_format_violation", "runtime_failure", "other", "unclear",
        ],
        "llm_primary_error": [
            "none", "no_call", "tool_misselection", "missing_argument", "incorrect_argument",
            "schema_format_violation", "runtime_failure", "other", "unclear",
        ],
        "llm_earliest_stage": ["none", "S1", "S2", "S3", "S4", "S5", "unclear"],
        "llm_boundary_violation": ["yes", "no", "unclear"],
        "llm_status": ["success", "partial_success", "error", "not_applicable", "unclear"],
        "llm_retryability": ["retryable", "non_retryable", "not_applicable", "unclear"],
        "llm_failure_type": [
            "timeout", "permission", "unavailable", "invalid_argument", "schema_error",
            "tool_error", "other", "none", "not_applicable", "unclear",
        ],
        "llm_final_result_correct": ["yes", "no", "not_applicable", "unclear"],
    }
    properties: dict[str, Any] = {
        name: {"type": "string", "enum": allowed} for name, allowed in enumerated.items()
    }
    properties["llm_notes"] = {"type": "string"}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "integrated_validation_annotation",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": LLM_FIELDS,
                "additionalProperties": False,
            },
        },
    }


def load_input(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def blank_rows(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{"record_id": str(record["record_id"]), **{field: "" for field in LLM_FIELDS}} for record in records]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_input(args.input)
    output_path = args.output_dir / "integrated_validation_annotation_llm.csv"
    raw_path = args.output_dir / "llm_validation_raw_responses.jsonl"
    error_path = args.output_dir / "llm_validation_parse_errors.csv"
    rows = blank_rows(records)
    errors: list[dict[str, str]] = []
    api_key = os.getenv(args.api_key_env, "")
    base_url = os.getenv(args.base_url_env, "") or None

    if not api_key:
        pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
        raw_path.write_text("", encoding="utf-8")
        pd.DataFrame([{"record_id": "", "parse_error": f"{args.api_key_env} is not configured", "raw_response": ""}]).to_csv(
            error_path, index=False, encoding="utf-8-sig"
        )
        print(f"No API key in {args.api_key_env}; wrote empty LLM annotation template for {len(records)} records.")
        return

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=args.timeout, max_retries=1)
    with raw_path.open("w", encoding="utf-8") as raw_handle:
        for index, record in enumerate(records):
            record_id = str(record["record_id"])
            try:
                completion = client.chat.completions.create(
                    model=args.model,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": PROMPT},
                        {
                            "role": "user",
                            "content": "Blind validation record:\n" + json.dumps(record, ensure_ascii=False),
                        },
                    ],
                    response_format=output_schema(),
                )
                content = completion.choices[0].message.content or ""
                raw_handle.write(json.dumps({"record_id": record_id, "raw_response": content}, ensure_ascii=False) + "\n")
                parsed = json.loads(content)
                rows[index].update({field: str(parsed.get(field, "")) for field in LLM_FIELDS})
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                raw_handle.write(json.dumps({"record_id": record_id, "api_error": message}, ensure_ascii=False) + "\n")
                errors.append({"record_id": record_id, "parse_error": message, "raw_response": ""})
                print(f"WARNING: {record_id} annotation failed: {message}")
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(errors, columns=["record_id", "parse_error", "raw_response"]).to_csv(
        error_path, index=False, encoding="utf-8-sig"
    )
    completed = sum(bool(row["llm_success"]) for row in rows)
    print(f"LLM annotations completed: {completed}/{len(rows)} using {args.model}.")
    print(f"Wrote {output_path}")
    print(f"Wrote {raw_path}")
    print(f"Wrote {error_path}")


if __name__ == "__main__":
    main()
