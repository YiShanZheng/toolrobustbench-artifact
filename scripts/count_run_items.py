"""Count model-task inference items for one concrete ToolRobust config."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from toolrobust.cli.main import resolve_configs
from toolrobust.datasets.builders import DatasetBuilder
from toolrobust.perturbation.pipeline import PerturbationPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--models-config", required=True, type=Path)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _completed_record_count(results_dir: Path | None) -> int:
    if results_dir is None:
        return 0
    records_csv = results_dir / "records.csv"
    if not records_csv.exists():
        return 0
    with records_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return len({(row.get("model_name", ""), row.get("sample_id", "")) for row in reader})


def main() -> None:
    args = parse_args()
    config, _, backend_map = resolve_configs(args.config, args.models_config)
    builder = DatasetBuilder(config)
    clean = builder._build_clean_samples(builder._build_template_pool(), progress=None)
    perturbed = builder._build_perturbed_samples(clean, PerturbationPipeline(config), progress=None)
    total = len(backend_map) * (len(clean) + len(perturbed))
    completed = _completed_record_count(args.results_dir) if args.resume else 0
    remaining = max(0, total - completed)
    print(json.dumps({"total": total, "completed": completed, "remaining": remaining}, ensure_ascii=False))


if __name__ == "__main__":
    main()
