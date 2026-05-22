"""Export benchmark samples into ShareGPT-style SFT data for LLaMA-Factory."""

from __future__ import annotations

import json
from pathlib import Path

from toolrobust.domain.schemas import TaskSample

SYSTEM_PROMPT = "You are a helpful assistant that outputs one JSON tool invocation with keys tool_name and arguments."


def export_samples_to_sharegpt(samples: list[TaskSample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for sample in samples:
        payload.append(
            {
                "id": sample.sample_id,
                "conversations": [
                    {"from": "human", "value": f"{SYSTEM_PROMPT}\n\nTask: {sample.user_instruction}"},
                    {
                        "from": "gpt",
                        "value": json.dumps({"tool_name": sample.exposed_tool_name, "arguments": sample.expected_arguments}, ensure_ascii=False),
                    },
                ],
            }
        )
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
