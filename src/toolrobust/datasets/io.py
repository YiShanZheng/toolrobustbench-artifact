"""JSONL dataset IO utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from toolrobust.domain.schemas import TaskSample


def ensure_parent(path: Path) -> None:
    """Ensure a path parent exists."""

    path.parent.mkdir(parents=True, exist_ok=True)


def write_task_samples(path: Path, samples: Iterable[TaskSample]) -> None:
    """Write task samples to JSONL."""

    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(sample.model_dump_json())
            handle.write("\n")


def read_task_samples(path: Path) -> list[TaskSample]:
    """Read task samples from JSONL."""

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    samples: list[TaskSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            samples.append(TaskSample.model_validate(json.loads(line)))
    return samples
