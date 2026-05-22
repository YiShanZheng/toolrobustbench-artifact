from __future__ import annotations

from pathlib import Path

from toolrobust.cli.main import resolve_configs
from toolrobust.domain.schemas import BackendConfig
from toolrobust.runner.experiment_runner import ExperimentRunner


def test_mock_pipeline_runs_lite(tmp_path: Path) -> None:
    config, backend_configs, backend_map = resolve_configs(Path("configs/benchmark_v3.yaml"), Path("configs/models.yaml"))
    config.models.include = ["mock-v1"]
    config.experiment.name = ""
    config.experiment.description = ""
    config.noise.enabled_families = ["interface_exposure"]
    config.dataset.sample_tool_count = 3
    config.dataset.max_clean_per_tool = 1
    config.noise.samples_per_bucket = 2
    backend_configs = {"mock-v1": BackendConfig(model_name="mock-v1", temperature=0.0, max_tokens=512)}
    backend_map = {"mock-v1": "mock"}
    config.output_root = tmp_path / "results"
    config.generated_dir = tmp_path / "generated"
    runner = ExperimentRunner(config, backend_configs, backend_map, clean_output=True)
    outputs = runner.run()
    assert outputs["results_dir"].name == "benchmark_v3_mock-v1_single_step_fam-a_n57-results"
    assert outputs["summary_overall"].exists()
    assert outputs["run_manifest"].exists()
    assert any(key.endswith("overall_robustness_curve") for key in outputs)


def test_mock_canary_run_limits_to_selected_d_subtypes(tmp_path: Path) -> None:
    config, _, _ = resolve_configs(Path("configs/benchmark_v3.yaml"), Path("configs/models.yaml"))
    config.models.include = ["mock-v1"]
    config.experiment.name = "d-canary"
    config.noise.enabled_families = ["execution_environment"]
    config.dataset.sample_tool_count = 5
    config.dataset.max_clean_per_tool = 3
    config.noise.samples_per_bucket = 2
    config.output.generate_plots = False
    backend_configs = {"mock-v1": BackendConfig(model_name="mock-v1", temperature=0.0, max_tokens=512)}
    backend_map = {"mock-v1": "mock"}
    config.output_root = tmp_path / "results"
    config.generated_dir = tmp_path / "generated"
    runner = ExperimentRunner(config, backend_configs, backend_map, clean_output=True)

    outputs = runner.run(
        results_dir=tmp_path / "results" / "d-canary-results",
        canary_subtypes={"D2 Runtime Variability", "D3 Failure Feedback Quality"},
        canary_samples_per_subtype_severity=2,
    )

    import pandas as pd

    records = pd.read_csv(outputs["records"])
    perturbed = records.loc[records["split"] == "perturbed"]
    assert len(records) == 13
    assert set(perturbed["subtype"]) == {"D2 Runtime Variability", "D3 Failure Feedback Quality"}
    assert perturbed.groupby(["subtype", "severity"]).size().eq(2).all()
    assert "decision_failure_type" in records.columns
    assert "expected_failure_types" in records.columns


def test_mock_cross_family_pairwise_run_exports_composition_tables(tmp_path: Path) -> None:
    config, _, _ = resolve_configs(Path("configs/benchmark_v3_cross_pairwise.yaml"), Path("configs/models.yaml"))
    config.models.include = ["mock-v1"]
    config.dataset.sample_tool_count = 2
    config.dataset.max_clean_per_tool = 1
    config.noise.samples_per_bucket = 1
    config.cross_family.samples_per_combo_bucket = 1
    config.output.generate_plots = False
    config.output_root = tmp_path / "results"
    config.generated_dir = tmp_path / "generated"
    backend_configs = {"mock-v1": BackendConfig(model_name="mock-v1", temperature=0.0, max_tokens=512)}
    backend_map = {"mock-v1": "mock"}
    runner = ExperimentRunner(config, backend_configs, backend_map, clean_output=True)

    outputs = runner.run(results_dir=tmp_path / "results" / "cross-results")

    import pandas as pd

    records = pd.read_csv(outputs["records"])
    summary = pd.read_csv(outputs["summary_by_composition"])
    interaction = pd.read_csv(outputs["interaction_effect"])
    boundary = pd.read_csv(outputs["family_boundary_report"])
    stage = pd.read_csv(outputs["summary_by_stage"])

    assert (records["family"] == "cross_family").any()
    assert not summary.empty
    assert not interaction.empty
    assert set(interaction["component_baseline_status"]) == {"missing_component_baseline"}
    assert "cross_family" not in set(boundary["family"])
    s5_medium = stage.loc[(stage["stage_target"] == "S5 End-to-End Task Success") & (stage["severity"] == "medium")].iloc[0]
    overall_medium = pd.read_csv(outputs["summary_overall"]).loc[lambda frame: frame["condition"] == "medium"].iloc[0]
    assert s5_medium["call_rate"] == overall_medium["call_rate"]
    assert s5_medium["canonical_tool_accuracy"] == overall_medium["canonical_tool_accuracy"]
