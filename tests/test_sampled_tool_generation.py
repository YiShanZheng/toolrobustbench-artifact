from __future__ import annotations

from pathlib import Path

from toolrobust.cli.main import resolve_configs
from toolrobust.datasets.builders import DatasetBuilder
from toolrobust.perturbation.pipeline import PerturbationPipeline
from toolrobust.tools.local_tools import list_available_tool_ids


def test_sample_tool_count_limits_generated_tasks_but_not_runtime_tool_pool(tmp_path: Path) -> None:
    config, _, _ = resolve_configs(Path("configs/benchmark_v3.yaml"), Path("configs/models.yaml"))
    config.dataset.sample_tool_count = 5
    config.dataset.max_clean_per_tool = 1
    config.generated_dir = tmp_path / "generated"

    builder = DatasetBuilder(config)
    selected_tool_ids = builder.selected_tool_ids()

    assert len(selected_tool_ids) == 5

    clean, perturbed = builder.build()

    generated_tool_ids = {sample.canonical_tool_id for sample in clean + perturbed}
    assert generated_tool_ids == set(selected_tool_ids)

    clean_registry = PerturbationPipeline(config).registry_index()["clean"]
    assert len(clean_registry.list_tools()) == len(list_available_tool_ids())
