"""Run a tiny Gemini stage-2 probe and save full raw responses.

This script is intentionally separate from the paper runner. It runs a small
slice, writes full BackendResponse dumps, and does not touch official
``records.csv`` files.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from toolrobust.cli.main import resolve_configs
from toolrobust.domain.enums import OutputProfile
from toolrobust.runner.experiment_runner import ExperimentRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("data/generated/run_configs/all-family/paper_all_family_gemini-2.5-pro_observation.yaml"),
    )
    parser.add_argument("--models-config", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument("--source-sample-id", default="route_01")
    parser.add_argument("--subtype", default="C1 Return Structure Noise")
    parser.add_argument("--severities", default="light,medium,heavy")
    parser.add_argument("--output-dir", type=Path, default=Path("data/results/debug/gemini-stage2-probe"))
    parser.add_argument(
        "--structured-stage2",
        choices=("auto", "on", "off"),
        default="auto",
        help="Force stage-2 response_format behavior for comparison.",
    )
    return parser.parse_args()


def _set_structured_mode(backend_configs: dict[str, Any], model: str, mode: str) -> None:
    if mode == "auto":
        return
    backend_configs[model].structured_stage2_response = mode == "on"


def _find_probe_tasks(runner: ExperimentRunner, model: str, args: argparse.Namespace) -> list[Any]:
    if model not in runner.backend_map:
        raise ValueError(f"Model {model!r} is not included by config {args.config}")
    clean, perturbed = runner.build_data(progress=None)
    severity_order = [item.strip() for item in args.severities.split(",") if item.strip()]
    selected = []
    for severity in severity_order:
        matches = [
            task
            for task in perturbed
            if task.subtype == args.subtype
            and task.severity.value == severity
            and str(task.metadata.get("source_sample_id")) == args.source_sample_id
        ]
        if not matches:
            raise ValueError(
                "No probe task found for "
                f"source={args.source_sample_id!r}, subtype={args.subtype!r}, severity={severity!r}"
            )
        selected.append(matches[0])
    return selected


def _extract_stage2_message(raw_response: dict[str, Any]) -> dict[str, Any]:
    stage2 = raw_response.get("stage2")
    if not isinstance(stage2, dict):
        return {"stage2_present": False}
    choices = stage2.get("choices")
    if not isinstance(choices, list) or not choices:
        return {"stage2_present": True, "choices_present": False}
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first, dict) else {}
    if not isinstance(message, dict):
        message = {}
    return {
        "stage2_present": True,
        "finish_reason": first.get("finish_reason") if isinstance(first, dict) else None,
        "message_keys": sorted(message),
        "content": message.get("content"),
        "tool_calls": message.get("tool_calls"),
        "refusal": message.get("refusal"),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config, backend_configs, backend_map = resolve_configs(args.config, args.models_config)
    _set_structured_mode(backend_configs, args.model, args.structured_stage2)
    runner = ExperimentRunner(
        config,
        backend_configs,
        backend_map,
        output_profile=OutputProfile(config.output.profile),
        clean_output=False,
    )
    tasks = _find_probe_tasks(runner, args.model, args)
    backend = runner._build_backend(args.model, runner.backend_map[args.model])
    registry_index = runner.pipeline.registry_index()
    output_dir = args.output_dir / f"{args.model}_{args.structured_stage2}"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for task in tasks:
        registry = registry_index[runner.pipeline.registry_key_for_sample(task)]
        response = runner._predict_with_retries(backend, task, registry.list_tools())
        record = runner.scorer.score(task, response, registry)
        stem = task.sample_id.replace("/", "_").replace("\\", "_")
        response_payload = response.model_dump(mode="json")
        record_payload = record.model_dump(mode="json")
        stage2_message = _extract_stage2_message(response_payload.get("raw_response", {}))
        _write_json(output_dir / f"{stem}.backend_response.json", response_payload)
        _write_json(output_dir / f"{stem}.evaluation_record.json", record_payload)
        _write_json(output_dir / f"{stem}.stage2_message.json", stage2_message)
        (output_dir / f"{stem}.stage2_content.txt").write_text(
            "" if stage2_message.get("content") is None else str(stage2_message.get("content")),
            encoding="utf-8",
        )
        rows.append(
            {
                "sample_id": task.sample_id,
                "severity": task.severity.value,
                "subtype": task.subtype,
                "call_present": int(record.call_present),
                "canonical_tool_match": int(record.canonical_tool_match),
                "format_valid": int(record.format_valid),
                "execution_success": int(record.execution_success),
                "overall_task_success": int(record.overall_task_success),
                "secondary_reason": record.secondary_reason or "",
                "notes": record.notes,
                "stage2_finish_reason": stage2_message.get("finish_reason"),
                "stage2_message_keys": ",".join(stage2_message.get("message_keys", [])),
                "stage2_content_length": len(str(stage2_message.get("content") or "")),
            }
        )

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output_dir": str(output_dir), "summary": str(summary_path), "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
