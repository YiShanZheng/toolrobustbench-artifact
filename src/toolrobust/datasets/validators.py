"""Dataset validation helpers for benchmark v2."""

from __future__ import annotations

from collections import Counter

from toolrobust.domain.schemas import TaskSample


def validate_task_samples(samples: list[TaskSample]) -> None:
    sample_ids = [sample.sample_id for sample in samples]
    counts = Counter(sample_ids)
    duplicates = [sample_id for sample_id, count in counts.items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate sample ids found: {duplicates}")
    if not samples:
        raise ValueError("Dataset is empty")
