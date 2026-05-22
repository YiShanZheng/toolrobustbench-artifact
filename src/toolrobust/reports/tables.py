"""Tabular report generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class ReportGenerator(ABC):
    """Abstract report generator."""

    @abstractmethod
    def generate(self, *args: object, **kwargs: object) -> None:
        """Generate a report artifact."""


class CsvReportGenerator(ReportGenerator):
    """Write standard CSV report artifacts."""

    def generate(self, output_dir: Path, **tables: pd.DataFrame) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in tables.items():
            frame.to_csv(output_dir / f"{name}.csv", index=False)
