from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import pytest

from toolrobust.cli.main import resolve_configs
from toolrobust.domain.enums import BenchmarkFamily, FailureTag, OperatorStrength, StageTarget
from toolrobust.domain.schemas import OperatorSpec
from toolrobust.perturbation.catalog import FAMILY_CATALOG, SUBTYPE_STAGE_TARGET, operator_index
from toolrobust.perturbation.pipeline import PerturbationPipeline


def _pipeline():
    config, _, _ = resolve_configs(Path("configs/benchmark_v3.yaml"), Path("configs/models.yaml"))
    config.noise.enabled_families = ["interface_exposure"]
    return PerturbationPipeline(config)


def test_family_a_contains_new_surface_only_operators() -> None:
    family_a = FAMILY_CATALOG["interface_exposure"]
    operators = {spec.operator_name for specs in family_a.values() for spec in specs}
    assert {
        "transparent_alias",
        "semi_opaque_name",
        "opaque_internal_name",
        "alias_pruning",
        "light_description_rewrite",
        "description_abstraction",
        "family_collision_name",
        "sibling_similarity_hint",
        "cross_tool_description_blend",
    }.issubset(operators)
    assert {
        "parameter_rename",
        "parameter_reorder",
        "required_hint_shift",
        "example_removal",
        "description_compression",
    }.isdisjoint(operators)


def test_family_a_operator_subtype_mapping_is_surface_only() -> None:
    index = operator_index()
    assert index["semi_opaque_name"].subtype == "A1 Name Surface Shift"
    assert index["opaque_internal_name"].subtype == "A1 Name Surface Shift"
    assert index["description_abstraction"].subtype == "A2 Description Abstraction"
    assert index["family_collision_name"].subtype == "A3 Family Similarity Pressure"


def test_family_a_stage_targets_are_locked_to_s1_s2() -> None:
    pipeline = _pipeline()
    specs = [spec for spec in pipeline.variant_specs() if spec.family == "interface_exposure"]
    assert specs
    assert all(spec.stage_target in {StageTarget.S1.value, StageTarget.S2.value} for spec in specs)
    assert all(operator.default_stage_target in {StageTarget.S1.value, StageTarget.S2.value} for spec in specs for operator in spec.operators)


def test_family_a_light_never_uses_strong_operator() -> None:
    specs = [spec for spec in _pipeline().variant_specs() if spec.family == "interface_exposure" and spec.severity == "light"]
    assert specs
    assert all(len(spec.operators) == 1 for spec in specs)
    assert all(operator.stack_role == "primary" for spec in specs for operator in spec.operators)
    assert all("light" in operator.severity_fit for spec in specs for operator in spec.operators)


def test_family_a_medium_rule_is_enforced() -> None:
    specs = [spec for spec in _pipeline().variant_specs() if spec.family == "interface_exposure" and spec.severity == "medium"]
    assert specs
    for spec in specs:
        assert len(spec.operators) == 1
        assert spec.operators[0].stack_role == "primary"
        assert all("medium" in operator.severity_fit for operator in spec.operators)


def test_family_a_heavy_prefers_same_family_and_multiple_strong_operators() -> None:
    pipeline = _pipeline()
    specs = [spec for spec in pipeline.variant_specs() if spec.family == "interface_exposure" and spec.severity == "heavy"]
    assert specs
    for spec in specs:
        assert all(operator.family == "interface_exposure" for operator in spec.operators)
        assert len(spec.operators) <= 2
        assert sum(operator.stack_role == "primary" for operator in spec.operators) == 1


def test_family_a_registry_variant_keeps_parameter_schema_unchanged() -> None:
    pipeline = _pipeline()
    registries = pipeline.registry_index()
    clean_registry = registries["clean"]
    interface_keys = [key for key in registries if key != "clean" and key.startswith("interface_exposure:")]
    assert interface_keys
    for registry_key in interface_keys:
        registry = registries[registry_key]
        for tool in registry.list_tools():
            clean_tool = clean_registry.get_tool(tool.canonical_tool_id)
            assert clean_tool is not None
            assert tool.parameters.model_dump() == clean_tool.parameters.model_dump()
            assert tool.parameter_name_map == clean_tool.parameter_name_map


def test_family_a_operator_failure_risks_do_not_include_parameter_failures() -> None:
    index = operator_index()
    interface_ops = [spec for spec in index.values() if spec.family == BenchmarkFamily.INTERFACE_EXPOSURE.value]
    assert interface_ops
    assert all(FailureTag.MISSING_PARAMETER.value not in spec.primary_failure_risks for spec in interface_ops)
    assert all(FailureTag.WRONG_PARAMETER.value not in spec.primary_failure_risks for spec in interface_ops)


def test_allow_cross_family_heavy_false_blocks_cross_family_mix() -> None:
    pipeline = _pipeline()
    assert pipeline.config.noise.allow_cross_family_heavy is False
    heavy_specs = [spec for spec in pipeline.variant_specs() if spec.severity == "heavy"]
    assert heavy_specs
    for spec in heavy_specs:
        assert all(operator.family == spec.family for operator in spec.operators)


def test_a1_heavy_combo_is_not_weaker_than_medium_design() -> None:
    specs = {
        (spec.subtype, spec.severity): [operator.operator_name for operator in spec.operators]
        for spec in _pipeline().variant_specs()
        if spec.family == "interface_exposure"
    }
    medium_ops = specs[("A1 Name Surface Shift", "medium")]
    heavy_ops = specs[("A1 Name Surface Shift", "heavy")]
    assert medium_ops == ["semi_opaque_name"]
    assert heavy_ops == [*medium_ops, "opaque_internal_name"]


def test_a1_heavy_changes_model_visible_name_beyond_medium() -> None:
    pipeline = _pipeline()
    registries = pipeline.registry_index()
    specs = {
        spec.severity: spec
        for spec in pipeline.variant_specs()
        if spec.family == "interface_exposure" and spec.subtype == "A1 Name Surface Shift"
    }
    medium_registry = registries[specs["medium"].registry_key]
    heavy_registry = registries[specs["heavy"].registry_key]
    for tool_id in pipeline.expanded_registry_tool_ids():
        medium_tool = medium_registry.get_tool(tool_id)
        heavy_tool = heavy_registry.get_tool(tool_id)
        assert medium_tool is not None
        assert heavy_tool is not None
        assert heavy_tool.exposed_name != medium_tool.exposed_name
        assert heavy_tool.exposed_name.startswith("svc_")
        assert medium_tool.description == heavy_tool.description
        assert medium_tool.parameters.model_dump() == heavy_tool.parameters.model_dump()


def test_a2_medium_uses_effective_description_abstraction_template() -> None:
    pipeline = _pipeline()
    spec = next(
        spec for spec in pipeline.variant_specs() if spec.subtype == "A2 Description Abstraction" and spec.severity == "medium"
    )
    assert [operator.operator_name for operator in spec.operators] == ["description_abstraction"]
    registry = pipeline.registry_index()[spec.registry_key]
    tool = registry.get_tool("search_policy")
    assert tool is not None
    clean_registry = pipeline.registry_index()["clean"]
    clean_tool = clean_registry.get_tool("search_policy")
    assert clean_tool is not None
    assert tool.description != clean_tool.description
    assert "selected" in tool.description.lower()


def test_a3_heavy_similarity_stack_is_stronger_than_medium_design() -> None:
    specs = {
        (spec.subtype, spec.severity): [operator.operator_name for operator in spec.operators]
        for spec in _pipeline().variant_specs()
        if spec.family == "interface_exposure"
    }
    assert specs[("A3 Family Similarity Pressure", "medium")] == ["family_collision_name"]
    assert specs[("A3 Family Similarity Pressure", "heavy")] == ["family_collision_name", "cross_tool_description_blend"]


def test_a3_heavy_description_blend_changes_all_expanded_tools() -> None:
    pipeline = _pipeline()
    registries = pipeline.registry_index()
    specs = {
        spec.severity: spec
        for spec in pipeline.variant_specs()
        if spec.family == "interface_exposure" and spec.subtype == "A3 Family Similarity Pressure"
    }
    medium_registry = registries[specs["medium"].registry_key]
    heavy_registry = registries[specs["heavy"].registry_key]
    for tool_id in pipeline.expanded_registry_tool_ids():
        medium_tool = medium_registry.get_tool(tool_id)
        heavy_tool = heavy_registry.get_tool(tool_id)
        assert medium_tool is not None
        assert heavy_tool is not None
        assert heavy_tool.exposed_name == medium_tool.exposed_name
        assert heavy_tool.description != medium_tool.description
        assert "neighboring interface wording" in heavy_tool.description
        assert heavy_tool.parameters.model_dump() == medium_tool.parameters.model_dump()


def test_family_a_preflight_gate_is_monotonic_and_pure() -> None:
    report = _pipeline().preflight_gate_report_frame()
    assert not report.empty
    assert set(report["monotonic_pass"].unique()).issubset({True, False})
    assert report["subtype_purity_pass"].all()
    assert report["light_nonzero_pass"].all()


def test_family_a_static_strengths_are_strictly_increasing() -> None:
    report = _pipeline().preflight_gate_report_frame()
    for subtype, group in report.groupby("subtype"):
        score_map = group.set_index("severity")["static_strength_score"].to_dict()
        assert score_map["light"] <= score_map["medium"] <= score_map["heavy"], subtype


def test_a3_medium_no_longer_uses_name_axis_operator() -> None:
    spec = next(spec for spec in _pipeline().variant_specs() if spec.subtype == "A3 Family Similarity Pressure" and spec.severity == "medium")
    assert spec.primary_operator == "family_collision_name"
    assert all(operator.operator_name != "semi_opaque_name" for operator in spec.operators)


def test_a1_heavy_static_strength_is_strictly_higher_than_medium() -> None:
    report = _pipeline().preflight_gate_report_frame()
    subtype = report.loc[report["subtype"] == "A1 Name Surface Shift"].set_index("severity")["static_strength_score"]
    assert subtype["heavy"] > subtype["medium"]


def test_interface_exposure_guard_rejects_s3_target(monkeypatch: pytest.MonkeyPatch) -> None:
    bad_catalog = OrderedDict(
        FAMILY_CATALOG | {
            "interface_exposure": OrderedDict(
                FAMILY_CATALOG["interface_exposure"]
                | {
                    "A3 Family Similarity Pressure": list(FAMILY_CATALOG["interface_exposure"]["A3 Family Similarity Pressure"])
                    + [
                        OperatorSpec(
                            family=BenchmarkFamily.INTERFACE_EXPOSURE.value,
                            subtype="A3 Family Similarity Pressure",
                            operator_name="illegal_s3_operator",
                            default_strength=OperatorStrength.STRONG,
                            default_stage_target=StageTarget.S3.value,
                            primary_failure_risks=[FailureTag.WRONG_TOOL.value],
                        )
                    ]
                }
            )
        }
    )
    monkeypatch.setattr("toolrobust.perturbation.pipeline.FAMILY_CATALOG", bad_catalog)
    monkeypatch.setattr(
        "toolrobust.perturbation.pipeline.SUBTYPE_STAGE_TARGET",
        dict(SUBTYPE_STAGE_TARGET) | {"A3 Family Similarity Pressure": StageTarget.S2.value},
    )
    with pytest.raises(ValueError, match="must not target S3/S4"):
        _pipeline()


def test_interface_exposure_guard_rejects_parameter_key_mutation(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _pipeline()
    original_apply = pipeline._apply_interface_exposure_operator

    def bad_apply(tool, operator_name, clean_tools):
        original_apply(tool, operator_name, clean_tools)
        first_key = next(iter(tool.parameters.properties))
        tool.parameters.properties[f"{first_key}_mutated"] = tool.parameters.properties.pop(first_key)

    monkeypatch.setattr(pipeline, "_apply_interface_exposure_operator", bad_apply)
    with pytest.raises(ValueError, match="illegally modified parameter schema"):
        pipeline.registry_index()
