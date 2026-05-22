from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from toolrobust.cli.main import resolve_configs
from toolrobust.backends.openai_chat import OpenAIChatBackend
from toolrobust.datasets.builders import DatasetBuilder
from toolrobust.domain.enums import FailureTag, SeverityLevel, StageTarget
from toolrobust.domain.schemas import BackendConfig, BackendResponse, ParsedInvocation, TaskSample
from toolrobust.evaluation.failure_profiles import FAMILY_FAILURE_PROFILES, STAGE_FAILURE_PROFILES
from toolrobust.evaluation.failure_taxonomy import analyze_failure, decide_observed_failure
from toolrobust.evaluation.metrics import BenchmarkMetricCalculator
from toolrobust.evaluation.parser import parse_execution_decision_content, parse_observation_result_content
from toolrobust.evaluation.scorer import Scorer
from toolrobust.execution.runtime_feedback import RuntimeFeedback, build_observation_target, run_perturbed_execution
from toolrobust.perturbation.pipeline import OPAQUE_INTERNAL_NAMES, PerturbationPipeline
from toolrobust.reports.output_paths import build_run_output_paths
from toolrobust.reports.plotting import PlotContext, PlotManager
from toolrobust.runner.experiment_runner import ExperimentRunner
from toolrobust.tools.executor import ToolExecutor
from toolrobust.tools.registry import RegistryBuilder


def _base_runner(tmp_path: Path) -> ExperimentRunner:
    config, backend_configs, backend_map = resolve_configs(Path("configs/benchmark_v3.yaml"), Path("configs/models.yaml"))
    config.models.include = ["mock-v1"]
    config.noise.enabled_families = ["interface_exposure"]
    config.dataset.sample_tool_count = 3
    config.dataset.max_clean_per_tool = 1
    config.noise.samples_per_bucket = 2
    backend_configs = {"mock-v1": BackendConfig(model_name="mock-v1", temperature=0.0, max_tokens=512)}
    backend_map = {"mock-v1": "mock"}
    config.output_root = tmp_path / "results"
    config.generated_dir = tmp_path / "generated"
    return ExperimentRunner(config, backend_configs, backend_map, clean_output=True)


def _sample(*, family: str, stage_target: str, severity: SeverityLevel = SeverityLevel.LIGHT) -> TaskSample:
    return TaskSample(
        sample_id=f"{family}-sample",
        split="perturbed",
        family=family,
        subtype="demo-subtype",
        operator="demo-operator",
        severity=severity,
        stage_target=stage_target,
        registry_variant_id="demo-registry",
        canonical_tool_id="search_policy",
        exposed_tool_name="search_policy",
        clean_tool_name="search_policy",
        task_text="Find policy text",
        user_instruction="Find policy text",
        expected_arguments={"query": "budget", "scope": "finance"},
        expected_result={"matches": ["Budget revision approval matrix"]},
        operators_applied=["demo-operator"],
        metadata={"source_sample_id": "clean-source"},
    )


def _record_frame(*, boundary_violation_flag: int, expected_primary_match: int, allowed_spillover_match: int, drop_success: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": "clean-source",
                "model_name": "mock-v1",
                "backend_name": "mock",
                "split": "clean",
                "family": "clean",
                "subtype": "clean",
                "operator": "clean",
                "severity": "clean",
                "stage_target": "clean",
                "source_sample_id": "clean-source",
                "registry_variant_id": "clean",
                "gold_canonical_tool_id": "search_policy",
                "gold_exposed_tool_name": "search_policy",
                "operator_names": [],
                "expected_arguments": {"query": "budget", "scope": "finance"},
                "predicted_arguments": {"query": "budget", "scope": "finance"},
                "expected_result": {"matches": ["Budget revision approval matrix"]},
                "predicted_tool_raw": "search_policy",
                "predicted_tool_resolved": "search_policy",
                "call_present": 1,
                "interface_name_match": 1,
                "canonical_tool_match": 1,
                "format_valid": 1,
                "missing_parameter_flag": 0,
                "wrong_parameter_flag": 0,
                "execution_success": 1,
                "overall_task_success": 1,
                "status": "success",
                "observed_failure_tag": None,
                "primary_failure_tag": None,
                "first_failure_stage": None,
                "is_cascade_failure": False,
                "cascade_from_stage": None,
                "expected_primary_match": False,
                "allowed_spillover_match": False,
                "boundary_violation_flag": False,
                "boundary_violation_reason": None,
                "secondary_reason": None,
                "notes": "",
                "latency_ms": 1.0,
                "cost_usd": 0.0,
            },
            {
                "sample_id": "perturbed-1",
                "model_name": "mock-v1",
                "backend_name": "mock",
                "split": "perturbed",
                "family": "interface_exposure",
                "subtype": "A1 Name Surface Shift",
                "operator": "transparent_alias",
                "severity": "light",
                "stage_target": "S1 Tool Selection",
                "source_sample_id": "clean-source",
                "registry_variant_id": "demo",
                "gold_canonical_tool_id": "search_policy",
                "gold_exposed_tool_name": "policy_query",
                "operator_names": ["transparent_alias"],
                "expected_arguments": {"query": "budget", "scope": "finance"},
                "predicted_arguments": {"query": "budget"},
                "expected_result": {"matches": ["Budget revision approval matrix"]},
                "predicted_tool_raw": "policy_query",
                "predicted_tool_resolved": "search_policy",
                "call_present": 1,
                "interface_name_match": 1,
                "canonical_tool_match": 1,
                "format_valid": 1,
                "missing_parameter_flag": 1,
                "wrong_parameter_flag": 0,
                "execution_success": 0,
                "overall_task_success": drop_success,
                "status": "failure",
                "observed_failure_tag": "missing-parameter",
                "primary_failure_tag": "wrong-tool" if expected_primary_match else "invalid-format",
                "first_failure_stage": "S2 Interface Grounding",
                "is_cascade_failure": bool(allowed_spillover_match),
                "cascade_from_stage": "S2 Interface Grounding" if allowed_spillover_match else None,
                "expected_primary_match": bool(expected_primary_match),
                "allowed_spillover_match": bool(allowed_spillover_match),
                "boundary_violation_flag": bool(boundary_violation_flag),
                "boundary_violation_reason": "primary failure `invalid-format` does not match expected primary failures for family `interface_exposure`" if boundary_violation_flag else None,
                "secondary_reason": "missing_required_parameter",
                "notes": "",
                "latency_ms": 1.0,
                "cost_usd": 0.0,
            },
        ]
    )


def test_registry_maps_exposed_names_back_to_canonical() -> None:
    registry = RegistryBuilder().build_clean_registry()
    assert len(registry.list_tools()) >= 10
    for tool in registry.list_tools():
        assert registry.resolve_tool_name(tool.exposed_name) == tool.canonical_tool_id
        assert registry.resolve_tool_name(tool.clean_tool_name) == tool.canonical_tool_id


def test_family_a_opaque_names_are_blinded() -> None:
    for canonical_tool_id, opaque_name in OPAQUE_INTERNAL_NAMES.items():
        assert opaque_name != canonical_tool_id
        assert canonical_tool_id not in opaque_name
        assert "tool" not in opaque_name.lower()


def test_variant_specs_follow_v3_axes() -> None:
    config, _, _ = resolve_configs(Path("configs/benchmark_v3.yaml"), Path("configs/models.yaml"))
    config.noise.enabled_families = ["interface_exposure", "observation"]
    pipeline = PerturbationPipeline(config)
    specs = pipeline.variant_specs()
    assert any(spec.family == "interface_exposure" and spec.subtype == "A1 Name Surface Shift" for spec in specs)
    assert any(spec.family == "observation" and spec.stage_target == "S4 Observation / Execution Handling" for spec in specs)
    assert pipeline.composite_variant_specs() == []


def test_cross_family_pairwise_specs_are_additive_and_keep_single_specs() -> None:
    config, _, _ = resolve_configs(Path("configs/benchmark_v3_cross_pairwise.yaml"), Path("configs/models.yaml"))
    pipeline = PerturbationPipeline(config)
    single_specs = pipeline.variant_specs()
    composite_specs = pipeline.composite_variant_specs()

    assert len(single_specs) == 42
    assert len(composite_specs) == 122
    assert len(pipeline.composition_order()) == 61
    assert len(pipeline.composition_order()) == len(set(pipeline.composition_order()))
    assert {spec.severity for spec in composite_specs} == {"medium", "heavy"}
    assert all(len(spec.families_applied) == 2 for spec in composite_specs)
    assert all({"observation", "execution_environment"} != set(spec.families_applied) for spec in composite_specs)
    assert any(spec.families_applied == ("interface_exposure", "user_intent") for spec in composite_specs)


def test_cross_family_dataset_samples_carry_composition_fields(tmp_path: Path) -> None:
    config, _, _ = resolve_configs(Path("configs/benchmark_v3_cross_pairwise.yaml"), Path("configs/models.yaml"))
    config.models.include = ["mock-v1"]
    config.dataset.sample_tool_count = 2
    config.dataset.max_clean_per_tool = 1
    config.noise.samples_per_bucket = 1
    config.generated_dir = tmp_path / "generated"

    clean, perturbed = DatasetBuilder(config).build()
    cross_samples = [sample for sample in perturbed if sample.family == "cross_family"]

    assert len(clean) == 2
    assert cross_samples
    sample = cross_samples[0]
    assert sample.composition_id
    assert len(sample.families_applied) == 2
    assert len(sample.subtypes_applied) == 2
    assert sample.stage_target == "S5 End-to-End Task Success"
    assert sample.metadata["composition_id"] == sample.composition_id


def test_cd_preflight_rows_are_generated_and_monotonic() -> None:
    config, _, _ = resolve_configs(Path("configs/benchmark_v3.yaml"), Path("configs/models.yaml"))
    config.noise.enabled_families = ["observation", "execution_environment"]
    pipeline = PerturbationPipeline(config)
    frame = pipeline.preflight_gate_report_frame()
    assert {"observation", "execution_environment"}.issubset(set(frame["family"]))
    assert frame["monotonic_pass"].all()
    assert frame["subtype_purity_pass"].all()

    specs = {
        (spec.subtype, spec.severity): [operator.operator_name for operator in spec.operators]
        for spec in pipeline.variant_specs()
        if spec.family == "observation"
    }
    assert specs[("C1 Return Structure Noise", "light")] == ["key_rename_soft", "field_reorder"]
    assert specs[("C1 Return Structure Noise", "medium")] == ["nesting_shift", "key_rename_soft"]
    assert specs[("C1 Return Structure Noise", "heavy")] == ["wrapper_insertion", "key_rename"]
    assert specs[("C2 Return Evidence Loss", "light")] == ["dropped_metadata"]
    assert specs[("C2 Return Evidence Loss", "medium")] == ["sparse_result"]
    assert specs[("C3 Return Evidence Corruption", "medium")] == ["stale_value", "supporting_conflict"]
    assert specs[("C3 Return Evidence Corruption", "heavy")] == ["stale_value", "supporting_conflict", "unit_mismatch"]
    assert specs[("C4 Return Evidence Competition", "light")] == ["low_utility_redundancy_soft"]
    assert specs[("C4 Return Evidence Competition", "medium")] == ["distractor_candidate_soft"]
    assert specs[("C4 Return Evidence Competition", "heavy")] == ["distractor_candidate"]


def test_abd_template_mappings_remain_locked_while_tuning_c() -> None:
    config, _, _ = resolve_configs(Path("configs/benchmark_v3.yaml"), Path("configs/models.yaml"))
    config.noise.enabled_families = ["interface_exposure", "user_intent", "execution_environment"]
    pipeline = PerturbationPipeline(config)
    specs = {
        (spec.family, spec.subtype, spec.severity): [operator.operator_name for operator in spec.operators]
        for spec in pipeline.variant_specs()
    }

    assert specs[("interface_exposure", "A1 Name Surface Shift", "heavy")] == ["semi_opaque_name", "opaque_internal_name"]
    assert specs[("user_intent", "B1 Indirectness", "heavy")] == ["style_transfer", "cross_sentence_scatter", "implication_rewrite"]
    assert specs[("execution_environment", "D1 Availability Failure", "heavy")] == ["service_unavailable_with_stale_cache", "timeout_after_partial_stream"]
    assert specs[("execution_environment", "D2 Runtime Variability", "medium")] == ["delayed_fresh_partial_with_stale_full", "flaky_shape_conflicting_attempts"]
    assert specs[("execution_environment", "D2 Runtime Variability", "heavy")] == ["delayed_fresh_partial_with_stale_full", "contaminated_flaky_attempts"]
    assert specs[("execution_environment", "D3 Failure Feedback Quality", "medium")] == ["hidden_partial_wrong_label"]
    assert specs[("execution_environment", "D3 Failure Feedback Quality", "heavy")] == ["hidden_partial_wrong_label", "conflicting_diagnostic_decoy"]


def test_user_intent_preflight_rows_are_generated_and_monotonic() -> None:
    config, _, _ = resolve_configs(Path("configs/benchmark_v3.yaml"), Path("configs/models.yaml"))
    config.noise.enabled_families = ["user_intent"]
    pipeline = PerturbationPipeline(config)
    frame = pipeline.preflight_gate_report_frame()
    assert set(frame["family"]) == {"user_intent"}
    assert frame["monotonic_pass"].all()
    assert frame["subtype_purity_pass"].all()

    specs = {
        (spec.subtype, spec.severity): [operator.operator_name for operator in spec.operators]
        for spec in pipeline.variant_specs()
    }
    assert specs[("B1 Indirectness", "light")] == ["constraint_reposition"]
    assert specs[("B1 Indirectness", "medium")] == ["style_transfer", "constraint_reposition"]
    assert specs[("B3 Distraction", "medium")] == ["near_semantic_distractor", "side_goal_insertion"]
    assert specs[("B4 Conflict", "heavy")] == ["contradictory_constraint", "late_revision", "intra_prompt_conflict"]


def test_user_intent_heavy_prompts_do_not_use_recovery_framing() -> None:
    config, _, _ = resolve_configs(Path("configs/benchmark_v3.yaml"), Path("configs/models.yaml"))
    config.noise.enabled_families = ["user_intent"]
    config.generated_dir = Path("data/generated/_tmp_user_intent_prompt_test")
    _, perturbed = DatasetBuilder(config).build()
    by_id = {sample.sample_id: sample for sample in perturbed}

    b1_medium = by_id["searchdoc_02__b1_indirectness__medium__1"]
    b3_heavy = by_id["searchdoc_02__b3_distraction__heavy__1"]
    b4_heavy = by_id["searchdoc_02__b4_conflict__heavy__1"]

    assert b1_medium.operators_applied == ["style_transfer", "constraint_reposition"]
    assert "once the request settles" in b1_medium.user_instruction

    assert "Coming back to the real ask" not in b3_heavy.user_instruction
    assert "The main thing I need" not in b3_heavy.user_instruction
    assert "keep circling around" in b3_heavy.user_instruction

    assert "Start with" not in b4_heavy.user_instruction
    assert "as if that had been the scope all along" in b4_heavy.user_instruction


def test_models_include_uses_model_names_not_backend_names() -> None:
    config, backend_configs, backend_map = resolve_configs(Path("configs/benchmark_v3.yaml"), Path("configs/models.yaml"))
    assert "gpt-5.4-mini" in config.models.include
    assert backend_map["gpt-5.4-mini"] == "openai"


def test_deepseek_config_disables_thinking_for_tool_calls() -> None:
    config, backend_configs, backend_map = resolve_configs(Path("configs/paper_all_family.yaml"), Path("configs/models.yaml"))
    assert "deepseek-v4-flash" in config.models.include
    assert backend_map["deepseek-v4-flash"] == "deepseek-v4-flash"
    assert backend_configs["deepseek-v4-flash"].base_url == "https://api.deepseek.com"
    assert backend_configs["deepseek-v4-flash"].extra_body == {"thinking": {"type": "disabled"}}


def test_openai_backend_includes_configured_extra_body_in_tool_calls() -> None:
    backend = OpenAIChatBackend(
        BackendConfig(
            model_name="deepseek-v4-pro",
            temperature=0.0,
            max_tokens=256,
            extra_body={"thinking": {"type": "disabled"}},
        )
    )
    tool_registry = RegistryBuilder().build_clean_registry().list_tools()

    kwargs = backend._create_tool_call_kwargs(
        messages=[{"role": "user", "content": "route from hotel to expo by metro"}],
        tool_registry=tool_registry,
    )

    assert kwargs["tool_choice"] == "required"
    assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
    assert kwargs["tools"][0]["type"] == "function"


def test_openai_backend_falls_back_to_openai_named_env_for_compat_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "fallback-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepseek.com")
    backend = OpenAIChatBackend(
        BackendConfig(
            model_name="deepseek-v4-pro",
            api_key_env="DEEPSEEK_API_KEY",
            base_url_env="DEEPSEEK_BASE_URL",
            base_url="https://api.deepseek.com",
        )
    )

    assert backend._resolve_env(backend.config.api_key_env, "OPENAI_API_KEY") == "fallback-key"
    assert backend._resolve_base_url() == "https://api.deepseek.com"


def test_failure_taxonomy_primary_labels_remain_six_tags() -> None:
    assert {tag.value for tag in FailureTag} == {
        "no-call",
        "wrong-tool",
        "invalid-format",
        "missing-parameter",
        "wrong-parameter",
        "execution-fail",
    }


def test_family_and_stage_failure_profiles_use_expected_primary_and_spillover() -> None:
    assert FAMILY_FAILURE_PROFILES["interface_exposure"].expected_primary_failures == ("wrong-tool",)
    assert FAMILY_FAILURE_PROFILES["interface_exposure"].allowed_spillover_failures == ("missing-parameter", "wrong-parameter")
    assert STAGE_FAILURE_PROFILES[StageTarget.S3.value].expected_primary_failures == ("missing-parameter", "wrong-parameter")
    assert STAGE_FAILURE_PROFILES[StageTarget.S4.value].allowed_spillover_failures == ("no-call",)


def test_interface_exposure_wrong_parameter_can_be_allowed_spillover_from_s2() -> None:
    task = _sample(family="interface_exposure", stage_target=StageTarget.S2.value)
    decision = decide_observed_failure(
        call_present=1,
        format_valid=1,
        predicted_tool_raw=task.exposed_tool_name,
        predicted_tool_resolved=task.canonical_tool_id,
        interface_name_match=1,
        canonical_tool_match=1,
        missing_parameter_flag=0,
        wrong_parameter_flag=1,
        execution_success=0,
        notes="",
    )
    analysis = analyze_failure(
        task=task,
        decision=decision,
        call_present=1,
        canonical_tool_match=1,
        format_valid=1,
        interface_name_match=1,
        missing_parameter_flag=0,
        wrong_parameter_flag=1,
        execution_success=0,
        notes="",
    )
    assert analysis.observed_failure_tag == FailureTag.WRONG_PARAMETER
    assert analysis.primary_failure_tag == FailureTag.WRONG_TOOL
    assert analysis.first_failure_stage == StageTarget.S2.value
    assert analysis.is_cascade_failure is True
    assert analysis.allowed_spillover_match is True
    assert analysis.boundary_violation_flag is False


def test_interface_exposure_unexpected_primary_failure_is_boundary_violation() -> None:
    task = _sample(family="interface_exposure", stage_target=StageTarget.S2.value)
    decision = decide_observed_failure(
        call_present=1,
        format_valid=0,
        predicted_tool_raw=task.exposed_tool_name,
        predicted_tool_resolved=task.canonical_tool_id,
        interface_name_match=1,
        canonical_tool_match=1,
        missing_parameter_flag=0,
        wrong_parameter_flag=0,
        execution_success=0,
        notes="invalid json arguments",
    )
    analysis = analyze_failure(
        task=task,
        decision=decision,
        call_present=1,
        canonical_tool_match=1,
        format_valid=0,
        interface_name_match=1,
        missing_parameter_flag=0,
        wrong_parameter_flag=0,
        execution_success=0,
        notes="invalid json arguments",
    )
    assert analysis.primary_failure_tag == FailureTag.INVALID_FORMAT
    assert analysis.expected_primary_match is False
    assert analysis.allowed_spillover_match is False
    assert analysis.boundary_violation_flag is True


def test_alias_prediction_resolves_and_executes_successfully(tmp_path: Path) -> None:
    runner = _base_runner(tmp_path)
    _, perturbed = runner.build_data()
    sample = next(sample for sample in perturbed if sample.family == "interface_exposure" and sample.subtype == "A1 Name Surface Shift" and sample.severity.value == "light")
    registry = runner.pipeline.registry_index()[sample.registry_variant_id]
    response = BackendResponse(
        backend_name="mock",
        model_name="mock-v1",
        raw_response={},
        parsed=ParsedInvocation(predicted_tool_raw=sample.exposed_tool_name, predicted_args_raw=sample.expected_arguments, call_present=True, format_valid=True),
    )
    record = Scorer().score(sample, response, registry)
    assert record.canonical_tool_match == 1
    assert record.execution_success == 1
    assert record.overall_task_success == 1


def test_runtime_feedback_injection_distinguishes_observation_and_execution_cases() -> None:
    observation_task = _sample(family="observation", stage_target=StageTarget.S4.value)
    observation_task.subtype = "C3 Return Evidence Corruption"
    observation_task.operators_applied = ["stale_value"]
    execution_task = _sample(family="execution_environment", stage_target=StageTarget.S4.value)
    execution_task.subtype = "D1 Availability Failure"
    execution_task.operators_applied = ["timeout_after_partial_stream"]

    feedback_obs = run_perturbed_execution(observation_task, observation_task.canonical_tool_id, observation_task.expected_arguments, executor=ToolExecutor())
    feedback_exec = run_perturbed_execution(execution_task, execution_task.canonical_tool_id, execution_task.expected_arguments, executor=ToolExecutor())

    assert feedback_obs.execution_success == 1
    assert feedback_obs.content != observation_task.expected_result
    assert feedback_exec.execution_success == 0
    assert isinstance(feedback_exec.content, dict)
    assert feedback_exec.content["elapsed_ms"] == 60000
    assert feedback_exec.content["raw"]["fragment_hint"]["matches_hint"][0]["preview"] == "Budget r"
    assert feedback_exec.content["stream_frames"][1]["payload"] != {"matches": ["Budget revision approval matrix"]}


def test_runtime_feedback_public_view_hides_clean_result_and_rejects_leaks() -> None:
    feedback = RuntimeFeedback(
        execution_success=1,
        status="success",
        content={"matches": ["Budget revision approval matrix"]},
        note="clean observation payload",
        clean_result={"matches": ["Budget revision approval matrix"]},
    )

    public_payload = feedback.as_public_dict()
    assert "clean_result" not in public_payload
    assert public_payload["content"] == {"matches": ["Budget revision approval matrix"]}

    leaking_feedback = RuntimeFeedback(
        execution_success=1,
        status="success",
        content={"clean_result": {"matches": ["Budget revision approval matrix"]}},
        note="bad payload",
        clean_result={"matches": ["Budget revision approval matrix"]},
    )
    with pytest.raises(ValueError, match="must not expose clean_result"):
        leaking_feedback.as_public_dict()


def test_observation_competition_operators_preserve_top_level_payload_shape() -> None:
    task = _sample(family="observation", stage_target=StageTarget.S4.value)
    task.subtype = "C4 Return Evidence Competition"
    task.operators_applied = ["distractor_candidate", "misleading_note"]

    feedback = run_perturbed_execution(task, task.canonical_tool_id, task.expected_arguments, executor=ToolExecutor())

    assert isinstance(feedback.content, dict)
    assert "candidate_bundle" in feedback.content
    assert "comparison_axes" in feedback.content
    assert "adjudication_note" in feedback.content
    assert "matches" not in feedback.content
    assert "reported_result" not in feedback.content
    assert "channel_preference" not in feedback.content
    assert isinstance(feedback.content["candidate_bundle"], list)
    assert len(feedback.content["candidate_bundle"]) == 2
    assert isinstance(feedback.content["comparison_axes"], dict)


def test_observation_structure_noise_moves_payload_off_the_top_level_path() -> None:
    task = _sample(family="observation", stage_target=StageTarget.S4.value)
    task.subtype = "C1 Return Structure Noise"

    task.operators_applied = ["key_rename_soft", "field_reorder"]
    soft_feedback = run_perturbed_execution(task, task.canonical_tool_id, task.expected_arguments, executor=ToolExecutor())
    assert isinstance(soft_feedback.content, dict)
    assert "matches" in soft_feedback.content
    assert any(key in soft_feedback.content for key in ("result_items", "lookup_query", "lookup_scope"))

    task.operators_applied = ["nesting_shift", "camel_snake_shift"]
    nested_feedback = run_perturbed_execution(task, task.canonical_tool_id, task.expected_arguments, executor=ToolExecutor())
    assert isinstance(nested_feedback.content, dict)
    assert "runtimeFrame" in nested_feedback.content
    assert "matches" not in nested_feedback.content
    sections = nested_feedback.content["runtimeFrame"]["sections"]
    assert any(section["slot"] == "result" for section in sections)

    task.operators_applied = ["wrapper_insertion", "key_rename"]
    wrapped_feedback = run_perturbed_execution(task, task.canonical_tool_id, task.expected_arguments, executor=ToolExecutor())
    assert isinstance(wrapped_feedback.content, dict)
    assert "observed_runtime_packet" in wrapped_feedback.content
    assert "matches" not in wrapped_feedback.content
    payload_channel = next(packet for packet in wrapped_feedback.content["observed_runtime_packet"] if packet["observed_channel"] == "payload")
    assert "observed_payload_fragments" in payload_channel["observed_body"]


def test_observation_evidence_loss_preserves_anchor_fields_but_degrades_context_progressively() -> None:
    task = _sample(family="observation", stage_target=StageTarget.S4.value)
    task.expected_arguments = {"query": "budget", "scope": "finance"}
    task.expected_result = {
        "scope": "finance",
        "query": "budget",
        "matches": ["Budget revision approval matrix"],
    }
    task.canonical_tool_id = "search_policy"
    task.exposed_tool_name = "search_policy"
    task.clean_tool_name = "search_policy"
    task.subtype = "C2 Return Evidence Loss"

    task.operators_applied = ["sparse_result"]
    medium_feedback = run_perturbed_execution(task, task.canonical_tool_id, task.expected_arguments, executor=ToolExecutor())
    assert medium_feedback.content["matches"] == ["Budget revision approval matrix"]
    assert medium_feedback.content["query"] == "budget"
    assert "scope" not in medium_feedback.content
    assert medium_feedback.content["scope_hint"] == {"token": "fin", "completeness": "partial"}
    assert medium_feedback.content["evidence_status"] == "sparse_context"

    task.operators_applied = ["missing_field", "partial_truncation"]
    heavy_feedback = run_perturbed_execution(task, task.canonical_tool_id, task.expected_arguments, executor=ToolExecutor())
    assert "matches" not in heavy_feedback.content
    assert "query" not in heavy_feedback.content
    assert "scope" not in heavy_feedback.content
    assert heavy_feedback.content["matches_hint"] == [{"preview": "Budget r", "remaining": 0}]
    assert heavy_feedback.content["query_hint"] == {"token": "b", "completeness": "minimal"}
    assert heavy_feedback.content["scope_hint"] == {"token": "f", "completeness": "minimal"}
    assert heavy_feedback.content["evidence_status"] == "anchor_hint_only"


def test_observation_corruption_uses_supporting_vs_primary_paths_for_retrieval_tools() -> None:
    task = _sample(family="observation", stage_target=StageTarget.S4.value)
    task.expected_arguments = {"query": "budget", "scope": "finance"}
    task.expected_result = {
        "scope": "finance",
        "query": "budget",
        "matches": ["Budget revision approval matrix"],
    }
    task.canonical_tool_id = "search_policy"
    task.exposed_tool_name = "search_policy"
    task.clean_tool_name = "search_policy"
    task.subtype = "C3 Return Evidence Corruption"

    task.operators_applied = ["stale_value"]
    stale_feedback = run_perturbed_execution(task, task.canonical_tool_id, task.expected_arguments, executor=ToolExecutor())
    assert stale_feedback.content["matches"] == ["Budget revision approval matrix"]
    assert stale_feedback.content["query"] == "budget"
    assert stale_feedback.content["scope"] == "finance"
    assert stale_feedback.content["reported_query"] != "budget"
    assert stale_feedback.content["supporting_review"]["focus_key"] == "query"

    task.operators_applied = ["supporting_conflict"]
    supporting_feedback = run_perturbed_execution(task, task.canonical_tool_id, task.expected_arguments, executor=ToolExecutor())
    assert supporting_feedback.content["matches"] == ["Budget revision approval matrix"]
    assert supporting_feedback.content["query"] == "budget"
    assert supporting_feedback.content["scope"] == "finance"
    assert supporting_feedback.content["supporting_variants"]["query"]["stale_candidate"] != "budget"
    assert supporting_feedback.content["supporting_variants"]["scope"]["stale_candidate"] != "finance"
    assert supporting_feedback.content["support_conflict_state"] == "multiple_supporting_cues_disagree"

    task.operators_applied = ["stale_value", "supporting_conflict"]
    medium_feedback = run_perturbed_execution(task, task.canonical_tool_id, task.expected_arguments, executor=ToolExecutor())
    assert medium_feedback.content["matches"] == ["Budget revision approval matrix"]
    assert medium_feedback.content["query"] == "budget"
    assert medium_feedback.content["reported_query"] != "budget"
    assert medium_feedback.content["supporting_review"]["focus_key"] == "query"
    assert medium_feedback.content["supporting_variants"]["query"]["stale_candidate"] != "budget"
    assert medium_feedback.content["supporting_variants"]["scope"]["stale_candidate"] != "finance"

    task.operators_applied = ["wrong_fact_field"]
    wrong_feedback = run_perturbed_execution(task, task.canonical_tool_id, task.expected_arguments, executor=ToolExecutor())
    assert wrong_feedback.content["matches"] != ["Budget revision approval matrix"]


def test_observation_unit_mismatch_changes_generic_tool_outputs() -> None:
    task = _sample(family="observation", stage_target=StageTarget.S4.value)
    task.canonical_tool_id = "aggregate_table"
    task.exposed_tool_name = "aggregate_table"
    task.clean_tool_name = "aggregate_table"
    task.subtype = "C3 Return Evidence Corruption"
    task.expected_arguments = {"entity": "alpha", "scope": "ops", "attribute": "status"}
    task.operators_applied = ["unit_mismatch"]

    feedback = run_perturbed_execution(task, task.canonical_tool_id, task.expected_arguments, executor=ToolExecutor())

    assert feedback.clean_result == {
        "entity": "alpha",
        "scope": "ops",
        "attribute": "status",
        "result": "aggregate_table:ops:alpha:status",
    }
    assert feedback.content["entity"] == "alpha"
    assert feedback.content["scope"] == "ops"
    assert feedback.content["attribute"] == "status"
    assert feedback.content["result"] != feedback.clean_result["result"]
    assert feedback.content["result_unit"] == "mismatched_label"


def test_build_observation_target_keeps_evidence_derived_fields() -> None:
    route_target = build_observation_target(
        "calculate_route",
        {"origin": "hotel", "target": "expo", "profile": "metro", "distance_km": 9, "duration_min": 22},
    )
    search_target = build_observation_target(
        "search_policy",
        {"scope": "finance", "query": "budget", "matches": ["Budget revision approval matrix"]},
    )

    assert route_target == {"distance_km": 9, "duration_min": 22, "profile": "metro"}
    assert search_target == {"matches": ["Budget revision approval matrix"], "query": "budget", "scope": "finance"}


def test_runtime_prompts_do_not_expose_clean_result_key() -> None:
    public_runtime_feedback = {
        "execution_success": 1,
        "status": "success",
        "content": {"matches": ["Budget revision approval matrix"]},
        "note": "clean observation payload",
    }

    observation_prompt = OpenAIChatBackend._build_observation_prompt(
        "search_policy",
        {"query": "budget", "scope": "finance"},
        public_runtime_feedback,
        {"matches": ["Budget revision approval matrix"]},
    )
    execution_prompt = OpenAIChatBackend._build_execution_prompt(
        _sample(family="execution_environment", stage_target=StageTarget.S4.value),
        "search_policy",
        {"query": "budget", "scope": "finance"},
        public_runtime_feedback,
    )

    assert "clean_result" not in observation_prompt
    assert "clean_result" not in execution_prompt
    assert "clean observation payload" not in observation_prompt
    assert '"note"' not in observation_prompt
    assert '"note"' not in execution_prompt
    assert "Original user request" not in observation_prompt
    assert '"argument_keys"' in observation_prompt
    assert '"query": "budget"' not in observation_prompt
    assert '"target_complexity"' in observation_prompt
    assert "Do not switch tools" in execution_prompt
    assert "degraded transport metadata" in execution_prompt
    assert "empty_response" not in execution_prompt
    assert "wrong_failure_label" not in execution_prompt


def test_execution_backend_retries_stage1_tool_call_for_family_d() -> None:
    task = _sample(family="execution_environment", stage_target=StageTarget.S4.value)
    task.subtype = "D1 Availability Failure"
    task.operators_applied = ["timeout_after_partial_stream"]
    backend = OpenAIChatBackend(BackendConfig(model_name="gpt-5.4-mini", temperature=0.0, max_tokens=256))
    tool_registry = RegistryBuilder().build_clean_registry().list_tools()

    class _FakeToolCall:
        def __init__(self, tool_name: str, arguments: str) -> None:
            self.function = type("Function", (), {"name": tool_name, "arguments": arguments})()

    class _FakeMessage:
        def __init__(self, content: str, *, tool_name: str | None = None, arguments: str = "{}") -> None:
            self.content = content
            self.tool_calls = [] if tool_name is None else [_FakeToolCall(tool_name, arguments)]

    class _FakeCompletion:
        def __init__(self, content: str, *, tool_name: str | None = None, arguments: str = "{}") -> None:
            self.choices = [type("Choice", (), {"message": _FakeMessage(content, tool_name=tool_name, arguments=arguments)})()]

        def model_dump(self, mode: str = "json") -> dict[str, object]:
            return {"mode": mode, "content": self.choices[0].message.content}

    class _FakeCompletions:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return _FakeCompletion("I think this is probably a timeout.")
            return _FakeCompletion(
                '{"status":"partial","retryable":true,"failure_type":"timeout","final_result":{"matches":["Budget revision approval matrix"]}}',
                tool_name="search_policy",
                arguments='{"query":"budget","scope":"finance"}',
            )

    fake_completions = _FakeCompletions()
    fake_client = type("FakeClient", (), {"chat": type("FakeChat", (), {"completions": fake_completions})()})()

    response = backend._predict_execution_stage(task, tool_registry, fake_client)

    assert len(fake_completions.calls) == 3
    assert response.parsed.predicted_tool_raw == task.exposed_tool_name
    assert response.parsed.call_present is True
    assert "attempt_1" in response.raw_response["stage1"]
    assert "attempt_2" in response.raw_response["stage1"]
    assert response.auxiliary["stage1_predicted_tool_resolved"] == task.canonical_tool_id
    assert response.auxiliary["execution_stage2_format_valid"] == 1


def test_parse_observation_result_content_accepts_wrapped_and_bare_json() -> None:
    wrapped_valid, wrapped_result, _ = parse_observation_result_content('{"final_result": {"matches": ["Budget revision approval matrix"]}}')
    bare_valid, bare_result, _ = parse_observation_result_content('{"matches": ["Budget revision approval matrix"]}')

    assert wrapped_valid is True
    assert wrapped_result == {"matches": ["Budget revision approval matrix"]}
    assert bare_valid is True
    assert bare_result == {"matches": ["Budget revision approval matrix"]}


def test_parse_execution_decision_content_accepts_complete_runtime_schema() -> None:
    valid, decision, _ = parse_execution_decision_content(
        '{"status": "partial", "retryable": true, "failure_type": "partial_response", "final_result": {"matches": ["Budget"]}}'
    )

    assert valid is True
    assert decision == {
        "status": "partial",
        "retryable": True,
        "failure_type": "partial_response",
        "final_result": {"matches": ["Budget"]},
    }


def test_observation_scorer_uses_stage1_call_and_stage2_result_match() -> None:
    task = _sample(family="observation", stage_target=StageTarget.S4.value)
    task.subtype = "C3 Return Evidence Corruption"
    task.operators_applied = ["wrong_fact_field", "stale_value"]
    registry = RegistryBuilder().build_clean_registry()

    success_response = BackendResponse(
        backend_name="mock",
        model_name="mock-v1",
        raw_response={},
        parsed=ParsedInvocation(
            predicted_tool_raw=task.exposed_tool_name,
            predicted_args_raw=task.expected_arguments,
            call_present=True,
            format_valid=True,
        ),
        auxiliary={
            "stage1_predicted_tool_raw": task.exposed_tool_name,
            "stage1_predicted_args_raw": task.expected_arguments,
            "stage1_predicted_tool_resolved": task.canonical_tool_id,
            "stage1_normalized_arguments": task.expected_arguments,
            "runtime_execution_success": 1,
            "runtime_status": "success",
            "runtime_note": "observation corrupted",
            "observation_target": task.expected_result,
            "observation_stage2_format_valid": 1,
            "observation_stage2_result": task.expected_result,
            "observation_stage2_parser_message": "parsed observation final_result wrapper",
        },
    )
    success_record = Scorer().score(task, success_response, registry)
    assert success_record.call_present == 1
    assert success_record.canonical_tool_match == 1
    assert success_record.format_valid == 1
    assert success_record.wrong_parameter_flag == 0
    assert success_record.execution_success == 1
    assert success_record.overall_task_success == 1

    wrong_result_response = BackendResponse(
        backend_name="mock",
        model_name="mock-v1",
        raw_response={},
        parsed=ParsedInvocation(
            predicted_tool_raw=task.exposed_tool_name,
            predicted_args_raw=task.expected_arguments,
            call_present=True,
            format_valid=True,
        ),
        auxiliary={
            "stage1_predicted_tool_raw": task.exposed_tool_name,
            "stage1_predicted_args_raw": task.expected_arguments,
            "stage1_predicted_tool_resolved": task.canonical_tool_id,
            "stage1_normalized_arguments": task.expected_arguments,
            "runtime_execution_success": 1,
            "runtime_status": "success",
            "runtime_note": "observation corrupted",
            "observation_target": task.expected_result,
            "observation_stage2_format_valid": 1,
            "observation_stage2_result": {"matches": ["Wrong policy"]},
            "observation_stage2_parser_message": "parsed observation final_result wrapper",
        },
    )
    wrong_result_record = Scorer().score(task, wrong_result_response, registry)
    assert wrong_result_record.call_present == 1
    assert wrong_result_record.canonical_tool_match == 1
    assert wrong_result_record.format_valid == 1
    assert wrong_result_record.wrong_parameter_flag == 1
    assert wrong_result_record.execution_success == 1
    assert wrong_result_record.overall_task_success == 0
    assert wrong_result_record.observed_failure_tag == FailureTag.WRONG_PARAMETER

    invalid_result_response = BackendResponse(
        backend_name="mock",
        model_name="mock-v1",
        raw_response={},
        parsed=ParsedInvocation(
            predicted_tool_raw=task.exposed_tool_name,
            predicted_args_raw=task.expected_arguments,
            call_present=True,
            format_valid=True,
        ),
        auxiliary={
            "stage1_predicted_tool_raw": task.exposed_tool_name,
            "stage1_predicted_args_raw": task.expected_arguments,
            "stage1_predicted_tool_resolved": task.canonical_tool_id,
            "stage1_normalized_arguments": task.expected_arguments,
            "runtime_execution_success": 1,
            "runtime_status": "success",
            "runtime_note": "observation corrupted",
            "observation_target": task.expected_result,
            "observation_stage2_format_valid": 0,
            "observation_stage2_result": None,
            "observation_stage2_parser_message": "observation content is not valid json object",
        },
    )
    invalid_result_record = Scorer().score(task, invalid_result_response, registry)
    assert invalid_result_record.call_present == 1
    assert invalid_result_record.format_valid == 0
    assert invalid_result_record.execution_success == 0
    assert invalid_result_record.overall_task_success == 0
    assert invalid_result_record.observed_failure_tag == FailureTag.INVALID_FORMAT


def test_execution_scorer_uses_runtime_decision_instead_of_runtime_success_bit() -> None:
    task = _sample(family="execution_environment", stage_target=StageTarget.S4.value)
    task.subtype = "D1 Availability Failure"
    task.operators_applied = ["timeout_after_partial_stream"]
    task.expected_result = {"matches": ["Budget revision approval matrix"], "query": "budget", "scope": "finance"}
    registry = RegistryBuilder().build_clean_registry()
    partial_result = {"matches": ["Budget revision approval matrix"]}

    success_response = BackendResponse(
        backend_name="mock",
        model_name="mock-v1",
        raw_response={},
        parsed=ParsedInvocation(
            predicted_tool_raw=task.exposed_tool_name,
            predicted_args_raw=task.expected_arguments,
            call_present=True,
            format_valid=True,
        ),
        auxiliary={
            "stage1_predicted_tool_raw": task.exposed_tool_name,
            "stage1_predicted_args_raw": task.expected_arguments,
            "stage1_predicted_tool_resolved": task.canonical_tool_id,
            "stage1_normalized_arguments": task.expected_arguments,
            "runtime_execution_success": 0,
            "runtime_status": "error",
            "runtime_feedback": {"elapsed_ms": 60000, "message": "stream stalled after partial frames", "raw": {"fragment_hint": {"matches_hint": [{"preview": "Budget r", "remaining": 0}], "evidence_status": "lossy_hint_only"}}},
            "runtime_note": "simulated a timeout after partial frames",
            "runtime_clean_result": task.expected_result,
            "execution_stage2_format_valid": 1,
            "execution_stage2_decision": {
                "status": "error",
                "retryable": True,
                "failure_type": "timeout_stream_interrupted",
                "final_result": None,
            },
            "execution_stage2_parser_message": "parsed execution decision json",
        },
    )
    success_record = Scorer().score(task, success_response, registry)
    assert success_record.call_present == 1
    assert success_record.format_valid == 1
    assert success_record.execution_success == 1
    assert success_record.overall_task_success == 1
    assert success_record.observed_failure_tag is None
    assert success_record.decision_status == "error"
    assert success_record.decision_failure_type == "timeout_stream_interrupted"
    assert success_record.expected_statuses == "error"
    assert success_record.stage2_status_match is True
    assert success_record.stage2_failure_type_match is True

    wrong_partial_response = BackendResponse(
        backend_name="mock",
        model_name="mock-v1",
        raw_response={},
        parsed=ParsedInvocation(
            predicted_tool_raw=task.exposed_tool_name,
            predicted_args_raw=task.expected_arguments,
            call_present=True,
            format_valid=True,
        ),
        auxiliary={
            "stage1_predicted_tool_raw": task.exposed_tool_name,
            "stage1_predicted_args_raw": task.expected_arguments,
            "stage1_predicted_tool_resolved": task.canonical_tool_id,
            "stage1_normalized_arguments": task.expected_arguments,
            "runtime_execution_success": 0,
            "runtime_status": "error",
            "runtime_feedback": {"elapsed_ms": 60000, "message": "stream stalled after partial frames", "raw": {"fragment_hint": {"matches_hint": [{"preview": "Budget r", "remaining": 0}], "evidence_status": "lossy_hint_only"}}},
            "runtime_note": "simulated a timeout after partial frames",
            "runtime_clean_result": task.expected_result,
            "execution_stage2_format_valid": 1,
            "execution_stage2_decision": {
                "status": "partial",
                "retryable": True,
                "failure_type": "timeout",
                "final_result": partial_result,
            },
            "execution_stage2_parser_message": "parsed execution decision json",
        },
    )
    wrong_partial_record = Scorer().score(task, wrong_partial_response, registry)
    assert wrong_partial_record.execution_success == 0
    assert wrong_partial_record.overall_task_success == 0
    assert wrong_partial_record.stage2_status_match is False

    wrong_decision_response = BackendResponse(
        backend_name="mock",
        model_name="mock-v1",
        raw_response={},
        parsed=ParsedInvocation(
            predicted_tool_raw=task.exposed_tool_name,
            predicted_args_raw=task.expected_arguments,
            call_present=True,
            format_valid=True,
        ),
        auxiliary={
            "stage1_predicted_tool_raw": task.exposed_tool_name,
            "stage1_predicted_args_raw": task.expected_arguments,
            "stage1_predicted_tool_resolved": task.canonical_tool_id,
            "stage1_normalized_arguments": task.expected_arguments,
            "runtime_execution_success": 0,
            "runtime_status": "error",
            "runtime_feedback": {"elapsed_ms": 60000, "message": "stream stalled after partial frames", "raw": {"fragment_hint": {"matches_hint": [{"preview": "Budget r", "remaining": 0}], "evidence_status": "lossy_hint_only"}}},
            "runtime_note": "simulated a timeout after partial frames",
            "runtime_clean_result": task.expected_result,
            "execution_stage2_format_valid": 1,
            "execution_stage2_decision": {
                "status": "success",
                "retryable": False,
                "failure_type": None,
                "final_result": task.expected_result,
            },
            "execution_stage2_parser_message": "parsed execution decision json",
        },
    )
    wrong_decision_record = Scorer().score(task, wrong_decision_response, registry)
    assert wrong_decision_record.call_present == 1
    assert wrong_decision_record.format_valid == 1
    assert wrong_decision_record.execution_success == 0
    assert wrong_decision_record.overall_task_success == 0
    assert wrong_decision_record.observed_failure_tag == FailureTag.EXECUTION_FAIL


def test_execution_scorer_accepts_sparse_empty_response_payload() -> None:
    task = _sample(family="execution_environment", stage_target=StageTarget.S4.value)
    task.subtype = "D1 Availability Failure"
    task.operators_applied = ["empty_response"]
    registry = RegistryBuilder().build_clean_registry()

    response = BackendResponse(
        backend_name="mock",
        model_name="mock-v1",
        raw_response={},
        parsed=ParsedInvocation(
            predicted_tool_raw=task.exposed_tool_name,
            predicted_args_raw=task.expected_arguments,
            call_present=True,
            format_valid=True,
        ),
        auxiliary={
            "stage1_predicted_tool_raw": task.exposed_tool_name,
            "stage1_predicted_args_raw": task.expected_arguments,
            "stage1_predicted_tool_resolved": task.canonical_tool_id,
            "stage1_normalized_arguments": task.expected_arguments,
            "runtime_execution_success": 0,
            "runtime_status": "error",
            "runtime_feedback": {"response_meta": {"body_bytes": 0, "headers_only": True}},
            "runtime_note": "returned a near-empty runtime payload",
            "runtime_clean_result": task.expected_result,
            "execution_stage2_format_valid": 1,
            "execution_stage2_decision": {
                "status": "error",
                "retryable": True,
                "failure_type": "empty_response",
                "final_result": None,
            },
            "execution_stage2_parser_message": "parsed execution decision json",
        },
    )

    record = Scorer().score(task, response, registry)
    assert record.execution_success == 1
    assert record.overall_task_success == 1


def test_execution_scorer_accepts_retryable_partial_runtime_results() -> None:
    task = _sample(family="execution_environment", stage_target=StageTarget.S4.value)
    task.subtype = "D2 Runtime Variability"
    task.operators_applied = ["delayed_fresh_partial_with_stale_full"]
    registry = RegistryBuilder().build_clean_registry()
    partial_result = {"matches": ["Budget"]}

    response = BackendResponse(
        backend_name="mock",
        model_name="mock-v1",
        raw_response={},
        parsed=ParsedInvocation(
            predicted_tool_raw=task.exposed_tool_name,
            predicted_args_raw=task.expected_arguments,
            call_present=True,
            format_valid=True,
        ),
        auxiliary={
            "stage1_predicted_tool_raw": task.exposed_tool_name,
            "stage1_predicted_args_raw": task.expected_arguments,
            "stage1_predicted_tool_resolved": task.canonical_tool_id,
            "stage1_normalized_arguments": task.expected_arguments,
            "runtime_execution_success": 0,
            "runtime_status": "error",
            "runtime_feedback": {
                "attempts": [
                    {"attempt_id": "cached", "freshness": "stale", "result": {"matches": ["Archived note for budget"]}},
                    {"attempt_id": "live", "freshness": "fresh", "fragment": partial_result, "completion_state": "truncated"},
                ],
                "fragment": partial_result,
                "completion_state": "truncated",
            },
            "runtime_note": "returned a stale complete candidate beside a fresher partial fragment",
            "runtime_clean_result": task.expected_result,
            "execution_stage2_format_valid": 1,
            "execution_stage2_decision": {
                "status": "partial",
                "retryable": True,
                "failure_type": "partial_response",
                "final_result": partial_result,
            },
            "execution_stage2_parser_message": "parsed execution decision json",
        },
    )
    record = Scorer().score(task, response, registry)
    assert record.call_present == 1
    assert record.format_valid == 1
    assert record.execution_success == 1
    assert record.overall_task_success == 1
    assert record.observed_failure_tag is None


def test_execution_scorer_prefers_partial_ground_truth_for_heavy_d2_runtime_variability() -> None:
    task = _sample(family="execution_environment", stage_target=StageTarget.S4.value)
    task.subtype = "D2 Runtime Variability"
    task.operators_applied = ["delayed_fresh_partial_with_stale_full", "flaky_shape_conflicting_attempts"]
    registry = RegistryBuilder().build_clean_registry()
    partial_result = {"matches": ["Budget"]}

    response = BackendResponse(
        backend_name="mock",
        model_name="mock-v1",
        raw_response={},
        parsed=ParsedInvocation(
            predicted_tool_raw=task.exposed_tool_name,
            predicted_args_raw=task.expected_arguments,
            call_present=True,
            format_valid=True,
        ),
        auxiliary={
            "stage1_predicted_tool_raw": task.exposed_tool_name,
            "stage1_predicted_args_raw": task.expected_arguments,
            "stage1_predicted_tool_resolved": task.canonical_tool_id,
            "stage1_normalized_arguments": task.expected_arguments,
            "runtime_execution_success": 0,
            "runtime_status": "error",
            "runtime_feedback": {
                "attempts": [
                    {"attempt_id": "t1", "shape": "canonical", "freshness": "stale", "result": {"matches": ["Archived note for budget"]}},
                    {"attempt_id": "t2", "shape": "compat", "freshness": "fresh", "compat_envelope": {"data": partial_result}},
                ],
                "raw": {"fragment": partial_result, "completion_state": "truncated"},
                "schema_mode": "rotating",
                "shape_variant": "unstable",
            },
            "runtime_note": "returned conflicting runtime attempts inside changing envelopes",
            "runtime_clean_result": task.expected_result,
            "execution_stage2_format_valid": 1,
            "execution_stage2_decision": {
                "status": "partial",
                "retryable": True,
                "failure_type": "runtime_variability",
                "final_result": partial_result,
            },
            "execution_stage2_parser_message": "parsed execution decision json",
        },
    )

    record = Scorer().score(task, response, registry)
    assert record.call_present == 1
    assert record.format_valid == 1
    assert record.execution_success == 1
    assert record.overall_task_success == 1
    assert record.observed_failure_tag is None


def test_execution_scorer_accepts_semantic_failure_labels_for_d3_feedback_quality() -> None:
    task = _sample(family="execution_environment", stage_target=StageTarget.S4.value)
    task.subtype = "D3 Failure Feedback Quality"
    task.operators_applied = ["wrong_failure_label_with_evidence"]
    registry = RegistryBuilder().build_clean_registry()

    response = BackendResponse(
        backend_name="mock",
        model_name="mock-v1",
        raw_response={},
        parsed=ParsedInvocation(
            predicted_tool_raw=task.exposed_tool_name,
            predicted_args_raw=task.expected_arguments,
            call_present=True,
            format_valid=True,
        ),
        auxiliary={
            "stage1_predicted_tool_raw": task.exposed_tool_name,
            "stage1_predicted_args_raw": task.expected_arguments,
            "stage1_predicted_tool_resolved": task.canonical_tool_id,
            "stage1_normalized_arguments": task.expected_arguments,
            "runtime_execution_success": 0,
            "runtime_status": "error",
            "runtime_feedback": {
                "status_code": 400,
                "message": "runtime wrapper reported required field missing, but invocation audit received all required keys",
                "reported_issue": "missing_input",
                "diagnostic_panel": {
                    "received_invocation": "complete",
                    "missing_keys": [],
                    "transport_retryable": True,
                },
            },
            "runtime_note": "reported a validation-style label that conflicts with runtime evidence",
            "runtime_clean_result": task.expected_result,
            "execution_stage2_format_valid": 1,
            "execution_stage2_decision": {
                "status": "error",
                "retryable": True,
                "failure_type": "validation_error",
                "final_result": None,
            },
            "execution_stage2_parser_message": "parsed execution decision json",
        },
    )

    record = Scorer().score(task, response, registry)
    assert record.execution_success == 1
    assert record.overall_task_success == 1
    assert record.stage2_failure_type_match is True


def test_failure_type_aliases_cover_safe_diagnostic_labels_only() -> None:
    assert Scorer._canonical_failure_label("conflicting_diagnostics") == "wrong_failure_label"
    assert Scorer._canonical_failure_label("transport_tail_missing_with_conflicting_previews") == "wrong_failure_label"
    assert Scorer._canonical_failure_label("response_envelope_malformed") == "wrong_failure_label"
    assert Scorer._canonical_failure_label("label_conflicts_with_payload_evidence") == "wrong_failure_label"
    assert Scorer._canonical_failure_label("conflicting_partial_results") == "wrong_failure_label"
    assert Scorer._canonical_failure_label("contradictory_payloads") == "wrong_failure_label"
    assert Scorer._canonical_failure_label("corrupted_response") == "wrong_failure_label"
    assert Scorer._canonical_failure_label("stale_fallback") == "service_unavailable"
    assert Scorer._canonical_failure_label("stale_cache_fallback_due_to_upstream_unavailability") == "service_unavailable"
    assert Scorer._canonical_failure_label("temporary_error") == "runtime_error"

    assert Scorer._canonical_failure_label("transport_error") == "transport_error"
    assert Scorer._canonical_failure_label("tail_missing") == "tail_missing"


def test_execution_scorer_recovers_partial_payload_hidden_under_wrong_failure_label() -> None:
    task = _sample(family="execution_environment", stage_target=StageTarget.S4.value)
    task.subtype = "D3 Failure Feedback Quality"
    task.operators_applied = ["hidden_partial_wrong_label"]
    registry = RegistryBuilder().build_clean_registry()
    hidden_partial = {"matches": ["Budget"]}

    response = BackendResponse(
        backend_name="mock",
        model_name="mock-v1",
        raw_response={},
        parsed=ParsedInvocation(
            predicted_tool_raw=task.exposed_tool_name,
            predicted_args_raw=task.expected_arguments,
            call_present=True,
            format_valid=True,
        ),
        auxiliary={
            "stage1_predicted_tool_raw": task.exposed_tool_name,
            "stage1_predicted_args_raw": task.expected_arguments,
            "stage1_predicted_tool_resolved": task.canonical_tool_id,
            "stage1_normalized_arguments": task.expected_arguments,
            "runtime_execution_success": 0,
            "runtime_status": "error",
            "runtime_feedback": {
                "status_code": 502,
                "message": "response envelope malformed after partial data",
                "reported_issue": "schema_parse_failure",
                "raw": {
                    "shadow_fragment": {"matches": ["Archived note for budget"]},
                    "recovery_trace": {
                        "candidate_fragment": hidden_partial,
                        "candidate_state": "tail_missing",
                        "confidence": "low",
                    },
                    "completion_state": "truncated",
                    "transport_retryable": True,
                },
            },
            "runtime_note": "hid a usable partial fragment under an actively wrong failure label",
            "runtime_clean_result": task.expected_result,
            "execution_stage2_format_valid": 1,
            "execution_stage2_decision": {
                "status": "partial",
                "retryable": True,
                "failure_type": "wrong_failure_label",
                "final_result": hidden_partial,
            },
            "execution_stage2_parser_message": "parsed execution decision json",
        },
    )
    record = Scorer().score(task, response, registry)
    assert record.call_present == 1
    assert record.format_valid == 1
    assert record.execution_success == 1
    assert record.overall_task_success == 1
    assert record.observed_failure_tag is None


def test_execution_scorer_accepts_nearby_label_for_vague_d3_errors() -> None:
    task = _sample(family="execution_environment", stage_target=StageTarget.S4.value)
    task.subtype = "D3 Failure Feedback Quality"
    task.operators_applied = ["vague_retry_hint"]
    registry = RegistryBuilder().build_clean_registry()

    response = BackendResponse(
        backend_name="mock",
        model_name="mock-v1",
        raw_response={},
        parsed=ParsedInvocation(
            predicted_tool_raw=task.exposed_tool_name,
            predicted_args_raw=task.expected_arguments,
            call_present=True,
            format_valid=True,
        ),
        auxiliary={
            "stage1_predicted_tool_raw": task.exposed_tool_name,
            "stage1_predicted_args_raw": task.expected_arguments,
            "stage1_predicted_tool_resolved": task.canonical_tool_id,
            "stage1_normalized_arguments": task.expected_arguments,
            "runtime_execution_success": 0,
            "runtime_status": "error",
            "runtime_feedback": {"message": "request could not be completed", "detail": "try the same call again later", "reported_issue": "temporary_dependency_unclear", "retryable_hint": True, "diagnostic_code": "temporary_dependency_unclear"},
            "runtime_note": "returned a vague runtime error with a weak retryability hint",
            "runtime_clean_result": task.expected_result,
            "execution_stage2_format_valid": 1,
            "execution_stage2_decision": {
                "status": "error",
                "retryable": True,
                "failure_type": "service_unavailable",
                "final_result": None,
            },
            "execution_stage2_parser_message": "parsed execution decision json",
        },
    )

    record = Scorer().score(task, response, registry)
    assert record.execution_success == 1
    assert record.overall_task_success == 1
    assert record.observed_failure_tag is None


def test_runtime_feedback_wrong_failure_label_with_evidence_stays_error() -> None:
    task = _sample(family="execution_environment", stage_target=StageTarget.S4.value)
    task.subtype = "D3 Failure Feedback Quality"
    task.operators_applied = ["wrong_failure_label_with_evidence"]

    feedback = run_perturbed_execution(
        task,
        task.canonical_tool_id,
        task.expected_arguments,
        executor=ToolExecutor(),
    )

    assert feedback.execution_success == 0
    assert feedback.status == "error"
    assert feedback.content["status_code"] == 400
    assert feedback.content["diagnostic_panel"]["transport_retryable"] is True
    assert "raw" not in feedback.content


def test_runtime_feedback_hidden_partial_wrong_label_exposes_partial_without_status_key() -> None:
    task = _sample(family="execution_environment", stage_target=StageTarget.S4.value)
    task.subtype = "D3 Failure Feedback Quality"
    task.operators_applied = ["hidden_partial_wrong_label"]

    feedback = run_perturbed_execution(
        task,
        task.canonical_tool_id,
        task.expected_arguments,
        executor=ToolExecutor(),
    )

    assert feedback.execution_success == 0
    assert feedback.status == "error"
    assert isinstance(feedback.content, dict)
    assert "partial" not in feedback.content
    assert "status" not in feedback.content
    candidate = feedback.content["raw"]["recoverable_fragment"]
    assert candidate["matches"] == ["Budget revision approval matrix"]
    assert candidate["query"] == "budget"
    assert feedback.content["diagnostic_panel"]["usable_fragment_path"] == "raw.recoverable_fragment"
    assert feedback.content["raw"]["shadow_fragment"] != {"matches": ["Budget revision approval matrix"]}


def test_availability_light_marks_fallback_payload_complete() -> None:
    task = _sample(family="execution_environment", stage_target=StageTarget.S4.value)
    task.subtype = "D1 Availability Failure"
    task.operators_applied = ["cached_availability_fallback"]

    feedback = run_perturbed_execution(
        task,
        task.canonical_tool_id,
        task.expected_arguments,
        executor=ToolExecutor(),
    )

    assert feedback.execution_success == 1
    assert feedback.status == "success"
    assert feedback.content["result"] == {"matches": ["Budget revision approval matrix"], "query": "budget", "scope": "finance"}
    assert feedback.content["delivery_meta"]["payload_state"] == "complete"
    assert feedback.content["delivery_meta"]["result_integrity"] == "complete_payload"
    assert Scorer._observation_result_matches(
        {"matches": ["Budget revision approval matrix"], "query": "budget", "scope": "finance"},
        feedback.content,
    )


def test_availability_heavy_combines_stale_cache_and_timeout_stream() -> None:
    task = _sample(family="execution_environment", stage_target=StageTarget.S4.value)
    task.subtype = "D1 Availability Failure"

    task.operators_applied = ["service_unavailable_with_stale_cache"]
    medium_feedback = run_perturbed_execution(
        task,
        task.canonical_tool_id,
        task.expected_arguments,
        executor=ToolExecutor(),
    )

    task.operators_applied = ["service_unavailable_with_stale_cache", "timeout_after_partial_stream"]
    heavy_feedback = run_perturbed_execution(
        task,
        task.canonical_tool_id,
        task.expected_arguments,
        executor=ToolExecutor(),
    )

    assert heavy_feedback.content != medium_feedback.content
    assert heavy_feedback.content["error_type"] == "timeout_after_service_unavailable"
    assert heavy_feedback.content["status_code"] == 503
    assert heavy_feedback.content["cache_state"] == "stale_fallback"
    assert heavy_feedback.content["cache_snapshot"] == medium_feedback.content["cache_snapshot"]
    assert heavy_feedback.content["recoverable_fragment"] == medium_feedback.content["fragment"]
    assert "stream_frames" in heavy_feedback.content
    assert "stream_frames" not in medium_feedback.content


def test_runtime_variability_light_wraps_result_apart_from_metadata() -> None:
    task = _sample(family="execution_environment", stage_target=StageTarget.S4.value)
    task.subtype = "D2 Runtime Variability"
    task.operators_applied = ["jittered_metadata"]

    feedback = run_perturbed_execution(
        task,
        task.canonical_tool_id,
        task.expected_arguments,
        executor=ToolExecutor(),
    )

    assert feedback.execution_success == 1
    assert feedback.status == "success"
    assert feedback.content["result"] == {"matches": ["Budget revision approval matrix"], "query": "budget", "scope": "finance"}
    assert feedback.content["delivery_meta"]["jitter_scope"] == "delivery_headers_only"
    assert feedback.content["delivery_meta"]["result_integrity"] == "complete_payload"
    assert Scorer._observation_result_matches(
        {"matches": ["Budget revision approval matrix"], "query": "budget", "scope": "finance"},
        feedback.content,
    )


def test_runtime_variability_heavy_keeps_fresh_fragment_inside_rotating_envelope() -> None:
    task = _sample(family="execution_environment", stage_target=StageTarget.S4.value)
    task.subtype = "D2 Runtime Variability"
    task.operators_applied = ["delayed_fresh_partial_with_stale_full", "flaky_shape_conflicting_attempts"]

    feedback = run_perturbed_execution(
        task,
        task.canonical_tool_id,
        task.expected_arguments,
        executor=ToolExecutor(),
    )

    assert feedback.execution_success == 0
    assert feedback.status == "error"
    assert feedback.content["raw"]["envelope"]["data"] == {"matches": ["Budget revision approval matrix"]}
    assert "fragment" not in feedback.content["raw"]
    assert "resolution_hints" not in feedback.content
    assert feedback.content["error_type"] == "runtime_attempt_conflict"


def test_runtime_variability_heavy_contaminates_replay_without_losing_ground_truth_fragment() -> None:
    task = _sample(family="execution_environment", stage_target=StageTarget.S4.value)
    task.subtype = "D2 Runtime Variability"
    task.operators_applied = ["delayed_fresh_partial_with_stale_full", "contaminated_flaky_attempts"]

    feedback = run_perturbed_execution(
        task,
        task.canonical_tool_id,
        task.expected_arguments,
        executor=ToolExecutor(),
    )

    assert feedback.execution_success == 0
    assert feedback.status == "error"
    assert feedback.content["error_type"] == "attempt_resolution_conflict"
    assert feedback.content["raw"]["envelope"]["data"] == {"matches": ["Budget revision approval matrix"]}
    assert feedback.content["selection_bias"]["surface_preference"] == "complete_replay"


def test_failure_feedback_heavy_adds_conflicting_diagnostics_around_hidden_partial() -> None:
    task = _sample(family="execution_environment", stage_target=StageTarget.S4.value)
    task.subtype = "D3 Failure Feedback Quality"
    task.operators_applied = ["hidden_partial_wrong_label", "conflicting_diagnostic_decoy"]

    feedback = run_perturbed_execution(
        task,
        task.canonical_tool_id,
        task.expected_arguments,
        executor=ToolExecutor(),
    )

    assert feedback.execution_success == 0
    assert feedback.status == "error"
    assert feedback.content["raw"]["envelope"]["fragment_hint"]["matches_hint"][0]["preview"] == "Budget r"
    assert feedback.content["raw"]["envelope"]["payload_integrity"] == "conflicting_preview_only"
    assert feedback.content["raw"]["audit_tail"]["missing_tail_bytes"] is True
    assert len(feedback.content["raw"]["audit_tail"]["preview_bundle"]) == 3
    assert feedback.content["diagnostic_panel"]["decoy_frame"] == "f2"
    assert feedback.content["diagnostic_panel"]["surface_bias"] == "partial_preview_is_visually_salient_but_not_recoverable"
    assert feedback.content["raw"]["frames"][1]["fragment_hint"] != {"matches": ["Budget revision approval matrix"]}


def test_run_outputs_follow_new_protocol(tmp_path: Path) -> None:
    runner = _base_runner(tmp_path)
    outputs = runner.run()
    for key in [
        "summary_overall",
        "summary_by_family",
        "summary_by_subtype",
        "summary_by_stage",
        "summary_by_operator",
        "summary_by_composition",
        "interaction_effect",
        "preflight_severity_gate_report",
        "severity_calibration_report",
        "failure_taxonomy",
        "primary_failure_breakdown",
        "cascade_failure_breakdown",
        "failure_by_family",
        "failure_by_composition",
        "paired_case_dump",
        "family_boundary_report",
        "run_manifest",
    ]:
        assert outputs[key].exists()


def test_run_converts_backend_interruptions_into_failed_records_and_continues(tmp_path: Path) -> None:
    runner = _base_runner(tmp_path)
    runner.config.execution.retry_times = 0
    call_count = 0

    class FailingBackend:
        def predict(self, task, tool_registry):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise RuntimeError("simulated api interruption")
            return BackendResponse(
                backend_name="mock",
                model_name="mock-v1",
                raw_response={},
                parsed=ParsedInvocation(
                    predicted_tool_raw=task.exposed_tool_name,
                    predicted_args_raw=task.expected_arguments,
                    call_present=True,
                    format_valid=True,
                ),
            )

    runner._build_backend = lambda model_name, backend_name: FailingBackend()
    clean, perturbed = runner.build_data()
    outputs = runner.run()

    frame = pd.read_csv(outputs["records"])
    assert len(frame) == len(clean) + len(perturbed)
    assert (frame["overall_task_success"] == 1).sum() == 1
    failed_rows = frame.loc[frame["overall_task_success"] == 0]
    assert not failed_rows.empty
    assert failed_rows["notes"].str.contains("backend request failed after 1 attempt\\(s\\): RuntimeError: simulated api interruption").any()


def test_resume_skips_existing_records(tmp_path: Path) -> None:
    runner = _base_runner(tmp_path)
    clean, perturbed = runner.build_data()
    tasks = clean + perturbed
    first_task = tasks[0]
    registry = runner.pipeline.registry_index()[runner.pipeline.registry_key_for_sample(first_task)]
    first_record = runner.scorer.score(
        first_task,
        BackendResponse(
            backend_name="mock",
            model_name="mock-v1",
            raw_response={},
            parsed=ParsedInvocation(
                predicted_tool_raw=first_task.exposed_tool_name,
                predicted_args_raw=first_task.expected_arguments,
                call_present=True,
                format_valid=True,
            ),
        ),
        registry,
    )
    runner.output_paths.results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([first_record.model_dump(mode="json")]).to_csv(runner.output_paths.records_csv, index=False)

    predicted_sample_ids: list[str] = []

    class CountingBackend:
        def predict(self, task, tool_registry):
            predicted_sample_ids.append(task.sample_id)
            return BackendResponse(
                backend_name="mock",
                model_name="mock-v1",
                raw_response={},
                parsed=ParsedInvocation(
                    predicted_tool_raw=task.exposed_tool_name,
                    predicted_args_raw=task.expected_arguments,
                    call_present=True,
                    format_valid=True,
                ),
            )

    class InspectProgress:
        def __init__(self) -> None:
            self.starts: list[tuple[str, int | None, str | None]] = []

        def start(self, stage, total_items=None, current_item=None):
            self.starts.append((stage, total_items, current_item))

        def advance(self, step=1, current_item=None):
            return

        def finish_stage(self, message=None):
            return

    runner._build_backend = lambda model_name, backend_name: CountingBackend()
    progress = InspectProgress()
    outputs = runner.run(resume=True, progress=progress)

    assert first_task.sample_id not in predicted_sample_ids
    frame = pd.read_csv(outputs["records"])
    assert len(frame) == len(tasks)
    assert frame.loc[(frame["sample_id"] == first_task.sample_id) & (frame["model_name"] == "mock-v1")].shape[0] == 1
    running_stage = next((entry for entry in progress.starts if entry[0] == "running_inference"), None)
    assert running_stage is not None
    assert running_stage[1] == len(tasks) - 1
    assert "resume loaded 1 completed records; remaining" in str(running_stage[2])


def test_resolve_configs_propagates_execution_retry_and_timeout() -> None:
    config, backend_configs, _ = resolve_configs(Path("configs/benchmark_v3.yaml"), Path("configs/models.yaml"))
    backend = backend_configs["gpt-5.4-mini"]
    assert backend.request_timeout_sec == config.execution.timeout_sec
    assert backend.request_max_retries == config.execution.retry_times


def test_failure_taxonomy_distinguishes_observed_and_primary_failure_tags(tmp_path: Path) -> None:
    runner = _base_runner(tmp_path)
    outputs = runner.run()
    frame = pd.read_csv(outputs["failure_taxonomy"])
    assert {"observed_failure_tag", "primary_failure_tag", "secondary_reason", "is_cascade_failure"}.issubset(frame.columns)
    assert set(frame["observed_failure_tag"].dropna().unique()).issubset({tag.value for tag in FailureTag})


def test_zero_sample_stage_rows_show_na(tmp_path: Path) -> None:
    runner = _base_runner(tmp_path)
    outputs = runner.run()
    frame = pd.read_csv(outputs["summary_by_stage"], keep_default_na=False)
    zero_rows = frame.loc[(frame["sample_count"] == 0) & (frame["severity"] != "clean")]
    assert not zero_rows.empty
    assert set(zero_rows["drop_vs_clean"]) == {"N/A"}


def test_stage_breakdown_zero_sample_buckets_are_not_treated_as_zero_rates(tmp_path: Path) -> None:
    runner = _base_runner(tmp_path)
    outputs = runner.run()
    manager = PlotManager(outputs["results_dir"])
    tables = manager.load_result_tables(outputs["results_dir"])
    manifest = manager.resolve_run_manifest(outputs["results_dir"])
    plot_context = PlotContext(manifest=manifest, model_name="mock-v1")
    rate_grid, sample_grid = manager._build_stage_breakdown_grid(
        frame=tables["summary_by_stage"].loc[tables["summary_by_stage"]["model_name"] == "mock-v1"].copy(),
        context=plot_context,
    )
    assert sample_grid.loc["S4 Observation / Execution Handling", "light"] == 0
    assert pd.isna(rate_grid.loc["S4 Observation / Execution Handling", "light"])


def test_paired_case_dump_contains_new_failure_diagnostics(tmp_path: Path) -> None:
    runner = _base_runner(tmp_path)
    outputs = runner.run()
    frame = pd.read_csv(outputs["paired_case_dump"])
    assert not frame.empty
    assert (frame["source_sample_id"] == frame["clean_sample_id"]).all()
    assert {
        "clean_success",
        "perturbed_success",
        "observed_failure_tag",
        "primary_failure_tag",
        "first_failure_stage",
        "is_cascade_failure",
        "cascade_from_stage",
        "expected_primary_match",
        "allowed_spillover_match",
        "boundary_violation_flag",
        "boundary_violation_reason",
        "operator_names",
        "family",
        "subtype",
        "stage_target",
    }.issubset(frame.columns)


def test_family_boundary_report_exports_new_columns(tmp_path: Path) -> None:
    runner = _base_runner(tmp_path)
    outputs = runner.run()
    frame = pd.read_csv(outputs["family_boundary_report"])
    family_a = frame.loc[frame["family"] == "interface_exposure"]
    assert not family_a.empty
    assert {
        "primary_wrong_tool_failures",
        "cascade_wrong_parameter_failures",
        "expected_primary_match_rate",
        "allowed_spillover_match_rate",
        "boundary_violation_rate",
        "dominant_primary_failure_tag",
        "dominant_observed_failure_tag",
        "family_boundary_pass",
        "notes",
    }.issubset(frame.columns)


def test_summary_by_operator_recommended_status_rules() -> None:
    config, _, _ = resolve_configs(Path("configs/benchmark_v3.yaml"), Path("configs/models.yaml"))
    config.models.include = ["mock-v1"]
    config.noise.enabled_families = ["interface_exposure"]
    manifest = ExperimentRunner(
        config,
        {"mock-v1": BackendConfig(model_name="mock-v1", temperature=0.0, max_tokens=512)},
        {"mock-v1": "mock"},
        clean_output=True,
    ).manifest
    calculator = BenchmarkMetricCalculator(manifest)

    disable_frame = _record_frame(boundary_violation_flag=1, expected_primary_match=0, allowed_spillover_match=0, drop_success=0)
    disable_summary = calculator._summary_by_operator(disable_frame, calculator._build_paired_frame(disable_frame))
    assert disable_summary.loc[0, "recommended_status"] == "disable"

    inspect_frame = _record_frame(boundary_violation_flag=0, expected_primary_match=1, allowed_spillover_match=1, drop_success=0)
    inspect_summary = calculator._summary_by_operator(inspect_frame, calculator._build_paired_frame(inspect_frame))
    assert inspect_summary.loc[0, "recommended_status"] == "inspect"

    rewrite_frame = _record_frame(boundary_violation_flag=0, expected_primary_match=0, allowed_spillover_match=0, drop_success=0)
    rewrite_summary = calculator._summary_by_operator(rewrite_frame, calculator._build_paired_frame(rewrite_frame))
    assert rewrite_summary.loc[0, "recommended_status"] == "rewrite"
    assert {"severity_fit_match", "calibration_issue_flag", "calibration_issue_reason", "intrinsic_strength_score", "exposure_axis", "stack_role"}.issubset(rewrite_summary.columns)


def test_preflight_severity_gate_report_exports_expected_columns(tmp_path: Path) -> None:
    runner = _base_runner(tmp_path)
    runner.build_data()
    path = runner.config.generated_dir / "preflight_severity_gate_report.csv"
    frame = pd.read_csv(path)
    assert not frame.empty
    assert {
        "family",
        "subtype",
        "severity",
        "template_name",
        "primary_operator",
        "amplifier_operator",
        "exposure_axis",
        "static_strength_score",
        "monotonic_pass",
        "subtype_purity_pass",
        "light_nonzero_pass",
        "note",
    }.issubset(frame.columns)


def test_severity_calibration_report_exports_expected_columns(tmp_path: Path) -> None:
    runner = _base_runner(tmp_path)
    outputs = runner.run()
    frame = pd.read_csv(outputs["severity_calibration_report"])
    assert not frame.empty
    assert {
        "family",
        "subtype",
        "severity",
        "template_name",
        "exposure_axis",
        "static_strength_score",
        "monotonic_template_pass",
        "subtype_purity_pass",
        "light_nonzero_pass",
        "sample_count",
        "overall_task_success_rate",
        "paired_success_flip_rate",
        "dominant_primary_failure_tag",
        "dominant_observed_failure_tag",
        "operator_names",
        "posthoc_model_result",
        "calibration_status",
        "calibration_note",
    }.issubset(frame.columns)


def test_old_strict_boundary_logic_is_removed() -> None:
    targets = [
        Path("src/toolrobust/evaluation/metrics.py"),
        Path("src/toolrobust/reports/markdown.py"),
        Path("src/toolrobust/runner/experiment_runner.py"),
        Path("src/toolrobust/reports/output_paths.py"),
    ]
    removed_terms = ["strict_boundary_pass", "s3_drift_failure_share", "wrong_parameter_failure_share", "family_stage_drift_report"]
    for path in targets:
        text = path.read_text(encoding="utf-8")
        for term in removed_terms:
            assert term not in text


def test_output_paths_use_experiment_name(tmp_path: Path) -> None:
    paths = build_run_output_paths(tmp_path / "results", experiment_name="benchmark_v3_single_step_mock-v1")
    assert paths.results_dir.name == "benchmark_v3_single_step_mock-v1-results"


def test_plot_manager_reads_v3_manifest(tmp_path: Path) -> None:
    runner = _base_runner(tmp_path)
    outputs = runner.run()
    manifest = PlotManager(outputs["results_dir"]).resolve_run_manifest(outputs["results_dir"])
    assert manifest.benchmark_version == "v3"
    assert {"interface_exposure", "user_intent", "observation", "execution_environment"}.issuperset(manifest.family_failure_profiles.keys())


def _stage2_response(parser_message: str, *, args: dict[str, object] | None = None) -> BackendResponse:
    return BackendResponse(
        backend_name="openai",
        model_name="gemini-2.5-pro",
        raw_response={},
        parsed=ParsedInvocation(
            predicted_tool_raw="search_policy",
            predicted_args_raw=args or {"query": "budget", "scope": "finance"},
            call_present=True,
            format_valid=True,
            parser_message="parsed backend payload",
        ),
        auxiliary={
            "runtime_execution_success": 1,
            "stage1_predicted_tool_resolved": "search_policy",
            "stage1_normalized_arguments": args or {"query": "budget", "scope": "finance"},
            "observation_stage2_parser_message": parser_message,
        },
    )


def test_stage2_compatibility_guard_flags_stage2_protocol_issue_for_any_model() -> None:
    task = _sample(family="observation", stage_target=StageTarget.S4.value)
    bad_response = _stage2_response("empty observation result content")
    assert ExperimentRunner._is_stage2_compatibility_guard_candidate("gemini-2.5-pro", task, bad_response)
    assert ExperimentRunner._is_stage2_compatibility_guard_candidate("gpt-5.4-mini", task, bad_response)
    assert ExperimentRunner._is_stage2_compatibility_issue(bad_response)

    normal_stage2_failure = _stage2_response("observation content used tool-call schema instead of final_result")
    assert not ExperimentRunner._is_stage2_compatibility_issue(normal_stage2_failure)

    wrong_args_response = _stage2_response("empty observation result content", args={"query": "wrong", "scope": "finance"})
    assert not ExperimentRunner._is_stage2_compatibility_guard_candidate("gemini-2.5-pro", task, wrong_args_response)
    non_stage2_task = _sample(family="user_intent", stage_target=StageTarget.S3.value)
    assert not ExperimentRunner._is_stage2_compatibility_guard_candidate("gpt-5.4-mini", non_stage2_task, bad_response)


def test_stage2_compatibility_guard_trips_after_repeated_protocol_issue() -> None:
    task = _sample(family="observation", stage_target=StageTarget.S4.value)
    response = _stage2_response("observation content is not valid json object")
    state: dict[str, dict[str, int]] = {}

    for _ in range(2):
        ExperimentRunner._check_stage2_compatibility_guard(
            model_name="gemini-2.5-pro",
            task=task,
            response=response,
            state=state,
            min_issues=3,
            min_checked=8,
            failure_rate=0.5,
        )

    with pytest.raises(RuntimeError, match="Stage-2 compatibility guard tripped"):
        ExperimentRunner._check_stage2_compatibility_guard(
            model_name="gemini-2.5-pro",
            task=task,
            response=response,
            state=state,
            min_issues=3,
            min_checked=8,
            failure_rate=0.5,
        )
