"""ToolRobust v3 perturbation pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from itertools import combinations, product

import pandas as pd

from toolrobust.domain.enums import BenchmarkFamily, FailureTag, OperatorStrength, SeverityLevel, StageTarget
from toolrobust.domain.schemas import BenchmarkV3Config, OperatorSpec, TaskSample, ToolSpec
from toolrobust.perturbation.catalog import (
    FAMILY_CATALOG,
    OPERATOR_CATALOG_VERSION,
    SUBTYPE_STAGE_TARGET,
    family_order as catalog_family_order,
    operator_index,
    subtype_order as catalog_subtype_order,
)
from toolrobust.tools.local_tools import TOOL_GROUPS, expand_enabled_tool_ids
from toolrobust.tools.registry import RegistryBuilder, ToolRegistry
from toolrobust.tools.schema_utils import validate_tool_registry

TRANSPARENT_ALIAS = {
    "get_weather": "weather_query",
    "get_air_quality": "air_quality_query",
    "get_temperature": "temperature_query",
    "convert_unit": "unit_translate",
    "convert_currency": "currency_translate",
    "search_document": "document_query",
    "search_policy": "policy_query",
    "search_faq": "faq_query",
    "calculate_route": "route_query",
    "estimate_travel_time": "travel_time_query",
    "sort_items": "items_sort_query",
    "filter_items": "items_filter_query",
}

SEMI_OPAQUE_NAMES = {
    "get_weather": "env_signal_a",
    "get_air_quality": "env_signal_b",
    "get_temperature": "env_signal_c",
    "convert_unit": "convert_signal_a",
    "convert_currency": "convert_signal_b",
    "search_document": "lookup_signal_a",
    "search_policy": "lookup_signal_b",
    "search_faq": "lookup_signal_c",
    "calculate_route": "route_signal_a",
    "estimate_travel_time": "route_signal_b",
    "sort_items": "list_signal_a",
    "filter_items": "list_signal_b",
}

OPAQUE_INTERNAL_NAMES = {
    "get_weather": "svc_01",
    "get_air_quality": "svc_02",
    "get_temperature": "svc_03",
    "convert_unit": "svc_04",
    "convert_currency": "svc_05",
    "search_document": "svc_06",
    "search_policy": "svc_07",
    "search_faq": "svc_08",
    "calculate_route": "svc_09",
    "estimate_travel_time": "svc_10",
    "sort_items": "svc_11",
    "filter_items": "svc_12",
}

FAMILY_COLLISION_NAMES = {
    "get_weather": "envNodeA",
    "get_air_quality": "envNodeB",
    "get_temperature": "envNodeC",
    "convert_unit": "convertNodeA",
    "convert_currency": "convertNodeB",
    "search_document": "lookupNodeA",
    "search_policy": "lookupNodeB",
    "search_faq": "lookupNodeC",
    "calculate_route": "routeNodeA",
    "estimate_travel_time": "routeNodeB",
    "sort_items": "listNodeA",
    "filter_items": "listNodeB",
}

LIGHT_DESCRIPTION_REWRITE = {
    "get_weather": "Get the overall outdoor conditions for a requested place and time.",
    "get_air_quality": "Get the air-quality conditions for a requested place and time.",
    "get_temperature": "Get the numeric temperature reading for a requested place and time.",
    "convert_unit": "Convert a measured value from one unit to another.",
    "convert_currency": "Convert a money amount from one currency to another.",
    "search_document": "Search general documents within a chosen collection.",
    "search_policy": "Search policy-oriented materials within a chosen collection.",
    "search_faq": "Search help-answer materials within a chosen collection.",
    "calculate_route": "Plan how to travel between an origin and destination.",
    "estimate_travel_time": "Estimate how long travel should take between an origin and destination.",
    "sort_items": "Return the same items reordered under a supplied rule.",
    "filter_items": "Return only the items that satisfy a supplied rule.",
}

DESCRIPTION_ABSTRACTION_MEDIUM = {
    "get_weather": "Handle a place-specific outdoor conditions lookup for a requested time window.",
    "get_air_quality": "Handle a place-specific air conditions lookup for a requested time window.",
    "get_temperature": "Handle a place-specific numeric heat reading lookup for a requested time window.",
    "convert_unit": "Handle a quantity translation across measurement systems.",
    "convert_currency": "Handle a money-amount translation across currencies.",
    "search_document": "Find general reference material from a selected source using a user query.",
    "search_policy": "Find rule or governance material from a selected source using a user query.",
    "search_faq": "Find answer-style support material from a selected source using a user query.",
    "calculate_route": "Work out travel guidance between two endpoints for a chosen mode.",
    "estimate_travel_time": "Estimate travel duration between two endpoints for a chosen mode.",
    "sort_items": "Return a provided collection reordered under a supplied rule.",
    "filter_items": "Return only the entries from a provided collection that satisfy a supplied rule.",
}

DESCRIPTION_ABSTRACTION_HEAVY = {
    "get_weather": "Handle an environment-facing request for one place and time context.",
    "get_air_quality": "Handle an environment-facing request for one place and time context.",
    "get_temperature": "Handle an environment-facing request for one place and time context.",
    "convert_unit": "Handle a scoped conversion request for a provided value context.",
    "convert_currency": "Handle a scoped conversion request for a provided value context.",
    "search_document": "Handle a scoped retrieval request over one selected content channel.",
    "search_policy": "Handle a scoped retrieval request over one selected content channel.",
    "search_faq": "Handle a scoped retrieval request over one selected content channel.",
    "calculate_route": "Handle a scoped travel coordination request across provided endpoints.",
    "estimate_travel_time": "Handle a scoped travel coordination request across provided endpoints.",
    "sort_items": "Handle a scoped collection processing request over provided entries.",
    "filter_items": "Handle a scoped collection processing request over provided entries.",
}

SIBLING_SIMILARITY_HINT = {
    "weather_env": "This tool belongs to a small family of related environment lookups with closely aligned wording.",
    "conversion": "This tool belongs to a small family of related conversion actions with closely aligned wording.",
    "search_retrieval": "This tool belongs to a small family of retrieval actions with closely aligned wording.",
    "route_travel": "This tool belongs to a small family of travel actions with closely aligned wording.",
    "list_ops": "This tool belongs to a small family of collection actions with closely aligned wording.",
}

ALLOWED_INTERFACE_EXPOSURE_STAGE_TARGETS = {StageTarget.S1.value, StageTarget.S2.value}
INTERFACE_EXPOSURE_FORBIDDEN_FAILURES = {FailureTag.MISSING_PARAMETER.value, FailureTag.WRONG_PARAMETER.value}
LIGHT_NONZERO_THRESHOLD = 12.0


@dataclass(frozen=True, slots=True)
class VariantSpec:
    family: str
    subtype: str
    severity: str
    operators: list[OperatorSpec]
    stage_target: str
    registry_key: str
    template_name: str = ""
    primary_operator: str = ""
    amplifier_operator: str = ""
    exposure_axis: str = ""
    static_strength_score: float = 0.0


@dataclass(frozen=True, slots=True)
class CompositeVariantSpec:
    composition_id: str
    component_specs: tuple[VariantSpec, ...]
    severity: str
    registry_key: str
    families_applied: tuple[str, ...]
    subtypes_applied: tuple[str, ...]
    stage_targets_applied: tuple[str, ...]
    operators: tuple[OperatorSpec, ...]
    template_name: str = ""


@dataclass(frozen=True, slots=True)
class InterfaceExposureTemplate:
    subtype: str
    severity: str
    template_name: str
    primary_operator: str
    amplifier_operator: str | None
    exposure_axis: str


@dataclass(frozen=True, slots=True)
class CalibratedRuntimeTemplate:
    family: str
    subtype: str
    severity: str
    template_name: str
    primary_operator: str
    amplifier_operator: str | None
    exposure_axis: str
    extra_operators: tuple[str, ...] = ()

    @property
    def operator_names(self) -> tuple[str, ...]:
        names = [self.primary_operator]
        if self.amplifier_operator:
            names.append(self.amplifier_operator)
        names.extend(self.extra_operators)
        return tuple(names)

    @property
    def amplifier_operator_label(self) -> str:
        return " + ".join(self.operator_names[1:])


@dataclass(frozen=True, slots=True)
class CalibratedPromptTemplate:
    family: str
    subtype: str
    severity: str
    template_name: str
    operator_names: tuple[str, ...]
    exposure_axis: str

    @property
    def primary_operator(self) -> str:
        return self.operator_names[0] if self.operator_names else ""

    @property
    def amplifier_operator(self) -> str:
        return " + ".join(self.operator_names[1:])


INTERFACE_EXPOSURE_TEMPLATE_PLAN = {
    "A1 Name Surface Shift": {
        SeverityLevel.LIGHT.value: InterfaceExposureTemplate("A1 Name Surface Shift", SeverityLevel.LIGHT.value, "a1_light_transparent_alias", "transparent_alias", None, "name"),
        SeverityLevel.MEDIUM.value: InterfaceExposureTemplate("A1 Name Surface Shift", SeverityLevel.MEDIUM.value, "a1_medium_name_surface_shift", "semi_opaque_name", None, "name"),
        SeverityLevel.HEAVY.value: InterfaceExposureTemplate("A1 Name Surface Shift", SeverityLevel.HEAVY.value, "a1_heavy_semi_opaque_opaque_internal_name", "semi_opaque_name", "opaque_internal_name", "name"),
    },
    "A2 Description Abstraction": {
        SeverityLevel.LIGHT.value: InterfaceExposureTemplate("A2 Description Abstraction", SeverityLevel.LIGHT.value, "a2_light_description_rewrite", "light_description_rewrite", None, "description"),
        SeverityLevel.MEDIUM.value: InterfaceExposureTemplate("A2 Description Abstraction", SeverityLevel.MEDIUM.value, "a2_medium_description_abstraction", "description_abstraction", None, "description"),
        SeverityLevel.HEAVY.value: InterfaceExposureTemplate("A2 Description Abstraction", SeverityLevel.HEAVY.value, "a2_heavy_description_abstraction_family_collision", "description_abstraction", "family_collision_name", "description"),
    },
    "A3 Family Similarity Pressure": {
        SeverityLevel.LIGHT.value: InterfaceExposureTemplate("A3 Family Similarity Pressure", SeverityLevel.LIGHT.value, "a3_light_sibling_similarity_hint", "sibling_similarity_hint", None, "family_similarity"),
        SeverityLevel.MEDIUM.value: InterfaceExposureTemplate("A3 Family Similarity Pressure", SeverityLevel.MEDIUM.value, "a3_medium_family_collision_name", "family_collision_name", None, "family_similarity"),
        SeverityLevel.HEAVY.value: InterfaceExposureTemplate("A3 Family Similarity Pressure", SeverityLevel.HEAVY.value, "a3_heavy_family_collision_cross_blend", "family_collision_name", "cross_tool_description_blend", "family_similarity"),
    },
}

USER_INTENT_TEMPLATE_PLAN = {
    "B1 Indirectness": {
        SeverityLevel.LIGHT.value: CalibratedPromptTemplate(BenchmarkFamily.USER_INTENT.value, "B1 Indirectness", SeverityLevel.LIGHT.value, "b1_light_constraint_reposition", ("constraint_reposition",), "indirectness"),
        SeverityLevel.MEDIUM.value: CalibratedPromptTemplate(BenchmarkFamily.USER_INTENT.value, "B1 Indirectness", SeverityLevel.MEDIUM.value, "b1_medium_style_transfer_reposition", ("style_transfer", "constraint_reposition"), "indirectness"),
        SeverityLevel.HEAVY.value: CalibratedPromptTemplate(BenchmarkFamily.USER_INTENT.value, "B1 Indirectness", SeverityLevel.HEAVY.value, "b1_heavy_style_scatter_implication", ("style_transfer", "cross_sentence_scatter", "implication_rewrite"), "indirectness"),
    },
    "B2 Underspecification": {
        SeverityLevel.LIGHT.value: CalibratedPromptTemplate(BenchmarkFamily.USER_INTENT.value, "B2 Underspecification", SeverityLevel.LIGHT.value, "b2_light_vague_quantity", ("vague_quantity",), "underspecification"),
        SeverityLevel.MEDIUM.value: CalibratedPromptTemplate(BenchmarkFamily.USER_INTENT.value, "B2 Underspecification", SeverityLevel.MEDIUM.value, "b2_medium_slot_implicitization", ("slot_implicitization",), "underspecification"),
        SeverityLevel.HEAVY.value: CalibratedPromptTemplate(BenchmarkFamily.USER_INTENT.value, "B2 Underspecification", SeverityLevel.HEAVY.value, "b2_heavy_slot_plus_constraint_drop", ("slot_implicitization", "partial_constraint_drop"), "underspecification"),
    },
    "B3 Distraction": {
        SeverityLevel.LIGHT.value: CalibratedPromptTemplate(BenchmarkFamily.USER_INTENT.value, "B3 Distraction", SeverityLevel.LIGHT.value, "b3_light_redundant_context", ("redundant_context",), "distraction"),
        SeverityLevel.MEDIUM.value: CalibratedPromptTemplate(BenchmarkFamily.USER_INTENT.value, "B3 Distraction", SeverityLevel.MEDIUM.value, "b3_medium_semantic_distractor_plus_side_goal", ("near_semantic_distractor", "side_goal_insertion"), "distraction"),
        SeverityLevel.HEAVY.value: CalibratedPromptTemplate(BenchmarkFamily.USER_INTENT.value, "B3 Distraction", SeverityLevel.HEAVY.value, "b3_heavy_semantic_side_goal_topic_drift", ("near_semantic_distractor", "side_goal_insertion", "soft_topic_drift"), "distraction"),
    },
    "B4 Conflict": {
        SeverityLevel.LIGHT.value: CalibratedPromptTemplate(BenchmarkFamily.USER_INTENT.value, "B4 Conflict", SeverityLevel.LIGHT.value, "b4_light_ambiguity_probe", ("ambiguity_probe",), "conflict"),
        SeverityLevel.MEDIUM.value: CalibratedPromptTemplate(BenchmarkFamily.USER_INTENT.value, "B4 Conflict", SeverityLevel.MEDIUM.value, "b4_medium_contradictory_constraint", ("contradictory_constraint",), "conflict"),
        SeverityLevel.HEAVY.value: CalibratedPromptTemplate(BenchmarkFamily.USER_INTENT.value, "B4 Conflict", SeverityLevel.HEAVY.value, "b4_heavy_contradiction_revision_intra_conflict", ("contradictory_constraint", "late_revision", "intra_prompt_conflict"), "conflict"),
    },
}

RUNTIME_TEMPLATE_PLAN = {
    BenchmarkFamily.OBSERVATION.value: {
        "C1 Return Structure Noise": {
            SeverityLevel.LIGHT.value: CalibratedRuntimeTemplate(BenchmarkFamily.OBSERVATION.value, "C1 Return Structure Noise", SeverityLevel.LIGHT.value, "c1_light_key_rename_soft_plus_reorder", "key_rename_soft", "field_reorder", "structure"),
            SeverityLevel.MEDIUM.value: CalibratedRuntimeTemplate(BenchmarkFamily.OBSERVATION.value, "C1 Return Structure Noise", SeverityLevel.MEDIUM.value, "c1_medium_nesting_plus_key_rename_soft", "nesting_shift", "key_rename_soft", "structure"),
            SeverityLevel.HEAVY.value: CalibratedRuntimeTemplate(BenchmarkFamily.OBSERVATION.value, "C1 Return Structure Noise", SeverityLevel.HEAVY.value, "c1_heavy_wrapper_plus_key_rename", "wrapper_insertion", "key_rename", "structure"),
        },
        "C2 Return Evidence Loss": {
            SeverityLevel.LIGHT.value: CalibratedRuntimeTemplate(BenchmarkFamily.OBSERVATION.value, "C2 Return Evidence Loss", SeverityLevel.LIGHT.value, "c2_light_dropped_metadata", "dropped_metadata", None, "evidence_loss"),
            SeverityLevel.MEDIUM.value: CalibratedRuntimeTemplate(BenchmarkFamily.OBSERVATION.value, "C2 Return Evidence Loss", SeverityLevel.MEDIUM.value, "c2_medium_sparse_result", "sparse_result", None, "evidence_loss"),
            SeverityLevel.HEAVY.value: CalibratedRuntimeTemplate(BenchmarkFamily.OBSERVATION.value, "C2 Return Evidence Loss", SeverityLevel.HEAVY.value, "c2_heavy_missing_field_plus_partial", "missing_field", "partial_truncation", "evidence_loss"),
        },
        "C3 Return Evidence Corruption": {
            SeverityLevel.LIGHT.value: CalibratedRuntimeTemplate(BenchmarkFamily.OBSERVATION.value, "C3 Return Evidence Corruption", SeverityLevel.LIGHT.value, "c3_light_stale_value", "stale_value", None, "evidence_corruption"),
            SeverityLevel.MEDIUM.value: CalibratedRuntimeTemplate(BenchmarkFamily.OBSERVATION.value, "C3 Return Evidence Corruption", SeverityLevel.MEDIUM.value, "c3_medium_stale_value_supporting_conflict", "stale_value", "supporting_conflict", "evidence_corruption"),
            SeverityLevel.HEAVY.value: CalibratedRuntimeTemplate(BenchmarkFamily.OBSERVATION.value, "C3 Return Evidence Corruption", SeverityLevel.HEAVY.value, "c3_heavy_stale_supporting_unit_mismatch", "stale_value", "supporting_conflict", "evidence_corruption", ("unit_mismatch",)),
        },
        "C4 Return Evidence Competition": {
            SeverityLevel.LIGHT.value: CalibratedRuntimeTemplate(BenchmarkFamily.OBSERVATION.value, "C4 Return Evidence Competition", SeverityLevel.LIGHT.value, "c4_light_low_utility_redundancy_soft", "low_utility_redundancy_soft", None, "evidence_competition"),
            SeverityLevel.MEDIUM.value: CalibratedRuntimeTemplate(BenchmarkFamily.OBSERVATION.value, "C4 Return Evidence Competition", SeverityLevel.MEDIUM.value, "c4_medium_distractor_candidate_soft", "distractor_candidate_soft", None, "evidence_competition"),
            SeverityLevel.HEAVY.value: CalibratedRuntimeTemplate(BenchmarkFamily.OBSERVATION.value, "C4 Return Evidence Competition", SeverityLevel.HEAVY.value, "c4_heavy_distractor_candidate", "distractor_candidate", None, "evidence_competition"),
        },
    },
    BenchmarkFamily.EXECUTION_ENVIRONMENT.value: {
        "D1 Availability Failure": {
            SeverityLevel.LIGHT.value: CalibratedRuntimeTemplate(BenchmarkFamily.EXECUTION_ENVIRONMENT.value, "D1 Availability Failure", SeverityLevel.LIGHT.value, "d1_light_cached_availability_fallback", "cached_availability_fallback", None, "availability"),
            SeverityLevel.MEDIUM.value: CalibratedRuntimeTemplate(BenchmarkFamily.EXECUTION_ENVIRONMENT.value, "D1 Availability Failure", SeverityLevel.MEDIUM.value, "d1_medium_service_unavailable_with_stale_cache", "service_unavailable_with_stale_cache", None, "availability"),
            SeverityLevel.HEAVY.value: CalibratedRuntimeTemplate(BenchmarkFamily.EXECUTION_ENVIRONMENT.value, "D1 Availability Failure", SeverityLevel.HEAVY.value, "d1_heavy_service_unavailable_plus_timeout_partial", "service_unavailable_with_stale_cache", "timeout_after_partial_stream", "availability"),
        },
        "D2 Runtime Variability": {
            SeverityLevel.LIGHT.value: CalibratedRuntimeTemplate(BenchmarkFamily.EXECUTION_ENVIRONMENT.value, "D2 Runtime Variability", SeverityLevel.LIGHT.value, "d2_light_jittered_metadata", "jittered_metadata", None, "runtime_variability"),
            SeverityLevel.MEDIUM.value: CalibratedRuntimeTemplate(BenchmarkFamily.EXECUTION_ENVIRONMENT.value, "D2 Runtime Variability", SeverityLevel.MEDIUM.value, "d2_medium_conflicting_attempts_plus_flaky_shape", "delayed_fresh_partial_with_stale_full", "flaky_shape_conflicting_attempts", "runtime_variability"),
            SeverityLevel.HEAVY.value: CalibratedRuntimeTemplate(BenchmarkFamily.EXECUTION_ENVIRONMENT.value, "D2 Runtime Variability", SeverityLevel.HEAVY.value, "d2_heavy_contaminated_rotating_attempts", "delayed_fresh_partial_with_stale_full", "contaminated_flaky_attempts", "runtime_variability"),
        },
        "D3 Failure Feedback Quality": {
            SeverityLevel.LIGHT.value: CalibratedRuntimeTemplate(BenchmarkFamily.EXECUTION_ENVIRONMENT.value, "D3 Failure Feedback Quality", SeverityLevel.LIGHT.value, "d3_light_vague_retry_hint", "vague_retry_hint", None, "failure_feedback"),
            SeverityLevel.MEDIUM.value: CalibratedRuntimeTemplate(BenchmarkFamily.EXECUTION_ENVIRONMENT.value, "D3 Failure Feedback Quality", SeverityLevel.MEDIUM.value, "d3_medium_hidden_partial_wrong_label", "hidden_partial_wrong_label", None, "failure_feedback"),
            SeverityLevel.HEAVY.value: CalibratedRuntimeTemplate(BenchmarkFamily.EXECUTION_ENVIRONMENT.value, "D3 Failure Feedback Quality", SeverityLevel.HEAVY.value, "d3_heavy_hidden_partial_with_conflicting_diagnostics", "hidden_partial_wrong_label", "conflicting_diagnostic_decoy", "failure_feedback"),
        },
    },
}


class PerturbationPipeline:
    def __init__(self, config: BenchmarkV3Config) -> None:
        self.config = config
        self._operator_index = operator_index()
        self._validate_interface_exposure_catalog()
        self._preflight_gate_rows = self._build_preflight_gate_rows()
        self._assert_preflight_gate()

    @property
    def operator_catalog_version(self) -> str:
        return OPERATOR_CATALOG_VERSION

    def enabled_families(self) -> list[str]:
        return [family for family in catalog_family_order() if family in self.config.noise.enabled_families]

    def enabled_subtypes(self) -> list[str]:
        return catalog_subtype_order(self.enabled_families(), self.config.noise.enabled_subtypes)

    def family_order(self) -> list[str]:
        return self.enabled_families()

    def subtype_order(self) -> list[str]:
        return self.enabled_subtypes()

    def stage_order(self) -> list[str]:
        return [StageTarget.S1.value, StageTarget.S2.value, StageTarget.S3.value, StageTarget.S4.value, StageTarget.S5.value]

    def registry_key_for_sample(self, sample: TaskSample) -> str:
        return sample.registry_variant_id

    def preflight_gate_rows(self) -> list[dict[str, object]]:
        return [dict(row) for row in self._preflight_gate_rows]

    def preflight_gate_report_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(self.preflight_gate_rows())
        if frame.empty:
            return pd.DataFrame(columns=["family", "subtype", "severity", "template_name", "primary_operator", "amplifier_operator", "exposure_axis", "static_strength_score", "monotonic_pass", "subtype_purity_pass", "light_nonzero_pass", "note"])
        frame["severity"] = pd.Categorical(frame["severity"], categories=[level.value for level in SeverityLevel], ordered=True)
        return frame.sort_values(["family", "subtype", "severity"]).reset_index(drop=True)

    def variant_specs(self) -> list[VariantSpec]:
        specs: list[VariantSpec] = []
        preflight_lookup = {(row["family"], row["subtype"], row["severity"]): row for row in self._preflight_gate_rows}
        for family in self.enabled_families():
            for subtype, operators in FAMILY_CATALOG[family].items():
                if subtype not in self.enabled_subtypes():
                    continue
                active_ops = [spec for spec in operators if spec.operator_name not in set(self.config.noise.disabled_operators)]
                if not active_ops:
                    continue
                for severity in self.config.noise.severities:
                    if severity == SeverityLevel.CLEAN:
                        continue
                    selected = self._select_operators(family, subtype, severity, active_ops)
                    if not selected:
                        continue
                    preflight_row = preflight_lookup.get((family, subtype, severity.value), {})
                    specs.append(
                        VariantSpec(
                            family=family,
                            subtype=subtype,
                            severity=severity.value,
                            operators=selected,
                            stage_target=SUBTYPE_STAGE_TARGET[subtype],
                            registry_key=f"{family}:{self._slug(subtype)}:{severity.value}",
                            template_name=str(preflight_row.get("template_name", "")),
                            primary_operator=str(preflight_row.get("primary_operator", "")),
                            amplifier_operator=str(preflight_row.get("amplifier_operator", "")),
                            exposure_axis=str(preflight_row.get("exposure_axis", "")),
                            static_strength_score=float(preflight_row.get("static_strength_score", 0.0)),
                        )
                    )
        return specs

    def composite_variant_specs(self) -> list[CompositeVariantSpec]:
        if not self.config.cross_family.enabled:
            return []

        single_specs = self.variant_specs()
        enabled_compositions = self.config.cross_family.enabled_compositions
        enabled_composition_set = None if enabled_compositions == "all" else set(enabled_compositions)
        cross_severities = {severity.value for severity in self.config.cross_family.severities}
        specs_by_family_severity: dict[tuple[str, str], list[VariantSpec]] = {}
        for spec in single_specs:
            if spec.severity in cross_severities:
                specs_by_family_severity.setdefault((spec.family, spec.severity), []).append(spec)

        composite_specs: list[CompositeVariantSpec] = []
        for family_set in self._cross_family_sets():
            for severity in self.config.cross_family.severities:
                severity_value = severity.value
                component_options = [specs_by_family_severity.get((family, severity_value), []) for family in family_set]
                if any(not options for options in component_options):
                    continue
                for component_specs in product(*component_options):
                    composite_spec = self._build_composite_spec(tuple(component_specs), severity_value)
                    if enabled_composition_set is not None and composite_spec.composition_id not in enabled_composition_set:
                        continue
                    composite_specs.append(composite_spec)
        return composite_specs

    def _cross_family_sets(self) -> list[tuple[str, str]]:
        enabled = self.enabled_families()
        configured = self.config.cross_family.family_sets
        raw_sets = configured or [list(pair) for pair in combinations(enabled, 2)]
        family_rank = {family: index for index, family in enumerate(catalog_family_order())}
        ordered_sets: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for raw_set in raw_sets:
            if len(raw_set) != 2 or any(family not in enabled for family in raw_set):
                continue
            ordered = tuple(sorted(raw_set, key=lambda family: family_rank.get(family, 999)))
            if ordered == (BenchmarkFamily.OBSERVATION.value, BenchmarkFamily.EXECUTION_ENVIRONMENT.value):
                continue
            if ordered not in seen:
                seen.add(ordered)
                ordered_sets.append(ordered)
        return ordered_sets

    def _build_composite_spec(self, component_specs: tuple[VariantSpec, ...], severity: str) -> CompositeVariantSpec:
        families = tuple(spec.family for spec in component_specs)
        subtypes = tuple(spec.subtype for spec in component_specs)
        stages = tuple(spec.stage_target for spec in component_specs)
        operators = tuple(operator for spec in component_specs for operator in spec.operators)
        family_slug = "+".join(self._family_short_name(family) for family in families)
        subtype_slug = "__".join(self._slug(subtype) for subtype in subtypes)
        composition_id = f"{family_slug}:{subtype_slug}"
        return CompositeVariantSpec(
            composition_id=composition_id,
            component_specs=component_specs,
            severity=severity,
            registry_key=f"cross_family:{family_slug}:{subtype_slug}:{severity}",
            families_applied=families,
            subtypes_applied=subtypes,
            stage_targets_applied=stages,
            operators=operators,
            template_name=" + ".join(spec.template_name for spec in component_specs if spec.template_name),
        )

    def registry_index(self) -> dict[str, ToolRegistry]:
        clean_tools = RegistryBuilder().build_clean_registry(self.expanded_registry_tool_ids(), include_similar_tool_groups=False).list_tools()
        registry_map: dict[str, list[ToolSpec]] = {"clean": clean_tools}
        for spec in self.variant_specs():
            registry_map[spec.registry_key] = self._apply_registry_variant(clean_tools, spec)
        for spec in self.composite_variant_specs():
            registry_map[spec.registry_key] = self._apply_composite_registry_variant(clean_tools, spec)
        return {key: RegistryBuilder.build_from_specs(validate_tool_registry(specs)) for key, specs in registry_map.items()}

    def expanded_registry_tool_ids(self) -> list[str]:
        return expand_enabled_tool_ids(None, include_similar_tool_groups=False)

    def stage_mapping(self) -> dict[str, str]:
        return dict(SUBTYPE_STAGE_TARGET)

    def composite_stage_mapping(self) -> dict[str, str]:
        return {spec.composition_id: StageTarget.S5.value for spec in self.composite_variant_specs()}

    def subtype_family_mapping(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for family in self.enabled_families():
            for subtype in FAMILY_CATALOG[family]:
                if subtype in self.enabled_subtypes():
                    mapping[subtype] = family
        return mapping

    def composition_order(self) -> list[str]:
        return list(dict.fromkeys(spec.composition_id for spec in self.composite_variant_specs()))

    def operator_failure_risk_mapping(self) -> dict[str, list[str]]:
        return {name: spec.primary_failure_risks for name, spec in self._operator_index.items()}

    def operator_metadata_mapping(self) -> dict[str, dict[str, object]]:
        return {
            name: {
                "family": spec.family,
                "subtype": spec.subtype,
                "intrinsic_strength_score": spec.intrinsic_strength_score,
                "subtype_affinity": list(spec.subtype_affinity),
                "severity_fit": list(spec.severity_fit),
                "exposure_axis": spec.exposure_axis,
                "stack_role": spec.stack_role,
                "can_stack_with": list(spec.can_stack_with),
                "should_not_stack_with": list(spec.should_not_stack_with),
                "minimum_effect_expectation": spec.minimum_effect_expectation,
                "maximum_effect_expectation": spec.maximum_effect_expectation,
            }
            for name, spec in self._operator_index.items()
        }

    def _select_operators(self, family: str, subtype: str, severity: SeverityLevel, available_ops: list[OperatorSpec]) -> list[OperatorSpec]:
        if family == BenchmarkFamily.INTERFACE_EXPOSURE.value:
            return self._select_interface_exposure_operators(subtype, severity)
        if family == BenchmarkFamily.USER_INTENT.value:
            return self._select_user_intent_operators(subtype, severity)
        if family in {BenchmarkFamily.OBSERVATION.value, BenchmarkFamily.EXECUTION_ENVIRONMENT.value}:
            return self._select_runtime_operators(family, subtype, severity)

        mild_ops = [spec for spec in available_ops if spec.default_strength == OperatorStrength.MILD]
        strong_ops = [spec for spec in available_ops if spec.default_strength == OperatorStrength.STRONG]
        max_ops = max(1, self.config.noise.max_operators_per_sample)
        if severity == SeverityLevel.LIGHT:
            return mild_ops[:1]
        if severity == SeverityLevel.MEDIUM:
            if strong_ops:
                return strong_ops[:1]
            return mild_ops[: min(2, max_ops)]
        if severity == SeverityLevel.HEAVY:
            selected: list[OperatorSpec] = strong_ops[:1]
            while len(selected) < max_ops and len(selected) < len(strong_ops):
                selected.append(strong_ops[len(selected)])
            return selected[:max_ops]
        return []

    def _select_user_intent_operators(self, subtype: str, severity: SeverityLevel) -> list[OperatorSpec]:
        template = USER_INTENT_TEMPLATE_PLAN[subtype][severity.value]
        disabled = set(self.config.noise.disabled_operators)
        if any(name in disabled for name in template.operator_names):
            return []
        selected = [self._operator_index[name].model_copy(deep=True) for name in template.operator_names]
        if selected:
            selected[0].stack_role = "primary"
        for operator in selected[1:]:
            operator.stack_role = "amplifier"
        self._validate_calibrated_selection(subtype, severity, template.template_name, selected)
        return selected

    def _select_runtime_operators(self, family: str, subtype: str, severity: SeverityLevel) -> list[OperatorSpec]:
        template = RUNTIME_TEMPLATE_PLAN[family][subtype][severity.value]
        operator_names = template.operator_names
        disabled = set(self.config.noise.disabled_operators)
        if any(name in disabled for name in operator_names):
            return []
        selected = [self._operator_index[name].model_copy(deep=True) for name in operator_names]
        if len(selected) >= 1:
            selected[0].stack_role = "primary"
        for operator in selected[1:]:
            operator.stack_role = "amplifier"
        self._validate_calibrated_selection(subtype, severity, template.template_name, selected)
        return selected

    def _validate_calibrated_selection(self, subtype: str, severity: SeverityLevel, template_name: str, selected: list[OperatorSpec]) -> None:
        if not selected:
            raise ValueError(f"Calibrated template '{template_name}' has no operators")
        max_ops = max(1, self.config.noise.max_operators_per_sample)
        if len(selected) > max_ops:
            raise ValueError(f"Calibrated template '{template_name}' exceeds max_operators_per_sample={max_ops}")
        for operator in selected:
            if operator.severity_fit and severity.value not in operator.severity_fit:
                raise ValueError(f"Operator '{operator.operator_name}' is not calibrated for severity '{severity.value}'")
            if operator.subtype_affinity and subtype not in operator.subtype_affinity:
                raise ValueError(f"Operator '{operator.operator_name}' is not calibrated for subtype '{subtype}'")
        for index, operator in enumerate(selected):
            expected_role = "primary" if index == 0 else "amplifier"
            if operator.stack_role != expected_role:
                raise ValueError(f"Template '{template_name}' must mark '{operator.operator_name}' as {expected_role}")
        if len(selected) >= 2:
            for left_index, left in enumerate(selected):
                for right in selected[left_index + 1 :]:
                    if left.can_stack_with and right.operator_name not in left.can_stack_with:
                        raise ValueError(f"Operator '{left.operator_name}' cannot stack with '{right.operator_name}'")
                    if right.can_stack_with and left.operator_name not in right.can_stack_with:
                        raise ValueError(f"Operator '{right.operator_name}' cannot stack with '{left.operator_name}'")
                    if right.operator_name in left.should_not_stack_with or left.operator_name in right.should_not_stack_with:
                        raise ValueError(f"Template '{template_name}' contains forbidden stacking")

    def _select_interface_exposure_operators(self, subtype: str, severity: SeverityLevel) -> list[OperatorSpec]:
        template = INTERFACE_EXPOSURE_TEMPLATE_PLAN[subtype][severity.value]
        operator_names = [template.primary_operator] + ([template.amplifier_operator] if template.amplifier_operator else [])
        disabled = set(self.config.noise.disabled_operators)
        if any(name in disabled for name in operator_names):
            return []
        selected = [self._operator_index[name].model_copy(deep=True) for name in operator_names]
        if len(selected) >= 1:
            selected[0].stack_role = "primary"
        if len(selected) == 2:
            selected[1].stack_role = "amplifier"
        self._validate_interface_exposure_selection(subtype, severity, template, selected)
        return selected

    def _validate_interface_exposure_selection(self, subtype: str, severity: SeverityLevel, template: InterfaceExposureTemplate, selected: list[OperatorSpec]) -> None:
        if not selected:
            raise ValueError(f"Interface exposure template '{template.template_name}' has no operators")
        if len(selected) > 2:
            raise ValueError(f"Interface exposure template '{template.template_name}' must use at most two operators")
        if selected[0].stack_role != "primary":
            raise ValueError(f"Template '{template.template_name}' must start with a primary operator")
        if len(selected) == 2 and selected[1].stack_role != "amplifier":
            raise ValueError(f"Template '{template.template_name}' must use an amplifier as the second operator")
        for operator in selected:
            if operator.severity_fit and severity.value not in operator.severity_fit:
                raise ValueError(f"Operator '{operator.operator_name}' is not calibrated for severity '{severity.value}'")
            if operator.subtype_affinity and subtype not in operator.subtype_affinity:
                raise ValueError(f"Operator '{operator.operator_name}' is not calibrated for subtype '{subtype}'")
        if len(selected) == 2:
            primary, amplifier = selected
            if amplifier.operator_name not in primary.can_stack_with:
                raise ValueError(f"Primary operator '{primary.operator_name}' cannot stack with '{amplifier.operator_name}'")
            if primary.operator_name not in amplifier.can_stack_with:
                raise ValueError(f"Amplifier operator '{amplifier.operator_name}' cannot stack with '{primary.operator_name}'")
            if amplifier.operator_name in primary.should_not_stack_with or primary.operator_name in amplifier.should_not_stack_with:
                raise ValueError(f"Template '{template.template_name}' contains forbidden stacking")

    def _apply_registry_variant(self, clean_tools: list[ToolSpec], spec: VariantSpec) -> list[ToolSpec]:
        if spec.family != BenchmarkFamily.INTERFACE_EXPOSURE.value:
            return [self._annotate_tool(tool, spec) for tool in clean_tools]

        updated: list[ToolSpec] = []
        for tool in clean_tools:
            original = tool.model_copy(deep=True)
            clone = tool.model_copy(deep=True)
            for operator in spec.operators:
                self._apply_interface_exposure_operator(clone, operator.operator_name, clean_tools)
            self._validate_interface_exposure_variant(original, clone, spec)
            updated.append(self._annotate_tool(clone, spec))
        return updated

    def _apply_composite_registry_variant(self, clean_tools: list[ToolSpec], spec: CompositeVariantSpec) -> list[ToolSpec]:
        interface_specs = [
            component
            for component in spec.component_specs
            if component.family == BenchmarkFamily.INTERFACE_EXPOSURE.value
        ]
        if not interface_specs:
            return [self._annotate_composite_tool(tool, spec) for tool in clean_tools]

        updated: list[ToolSpec] = []
        for tool in clean_tools:
            original = tool.model_copy(deep=True)
            clone = tool.model_copy(deep=True)
            for component in interface_specs:
                for operator in component.operators:
                    self._apply_interface_exposure_operator(clone, operator.operator_name, clean_tools)
                self._validate_interface_exposure_variant(original, clone, component)
            updated.append(self._annotate_composite_tool(clone, spec))
        return updated

    def _apply_interface_exposure_operator(self, tool: ToolSpec, operator_name: str, clean_tools: list[ToolSpec]) -> None:
        if operator_name == "transparent_alias":
            tool.exposed_name = TRANSPARENT_ALIAS.get(tool.canonical_tool_id, f"{tool.canonical_tool_id}_query")
            return
        if operator_name == "semi_opaque_name":
            tool.exposed_name = SEMI_OPAQUE_NAMES.get(tool.canonical_tool_id, self._generic_opaque_name(tool.canonical_tool_id))
            return
        if operator_name == "alias_pruning":
            tool.aliases = []
            return
        if operator_name == "opaque_internal_name":
            tool.exposed_name = self._opaque_internal_name(tool.canonical_tool_id)
            return
        if operator_name == "light_description_rewrite":
            tool.description = LIGHT_DESCRIPTION_REWRITE.get(tool.canonical_tool_id, self._generic_light_description(tool.canonical_tool_id))
            return
        if operator_name == "description_abstraction":
            tool.description = DESCRIPTION_ABSTRACTION_MEDIUM.get(tool.canonical_tool_id, self._generic_medium_description(tool.canonical_tool_id))
            return
        if operator_name == "sibling_similarity_hint":
            hint = SIBLING_SIMILARITY_HINT.get(
                self._tool_group_for(tool.canonical_tool_id),
                "It belongs to a nearby operational family and can be confused with sibling tools.",
            )
            tool.description = f"{tool.description} {hint}".strip()
            return
        if operator_name == "family_collision_name":
            tool.exposed_name = FAMILY_COLLISION_NAMES.get(tool.canonical_tool_id, self._generic_collision_name(tool.canonical_tool_id))
            return
        if operator_name == "cross_tool_description_blend":
            tool.description = self._cross_tool_description(tool.canonical_tool_id, clean_tools, weight=0.5)
            return
        raise ValueError(f"Unsupported interface exposure operator: {operator_name}")

    def _validate_interface_exposure_catalog(self) -> None:
        family = BenchmarkFamily.INTERFACE_EXPOSURE.value
        for subtype, operators in FAMILY_CATALOG[family].items():
            stage_target = SUBTYPE_STAGE_TARGET[subtype]
            if stage_target not in ALLOWED_INTERFACE_EXPOSURE_STAGE_TARGETS:
                raise ValueError(f"Interface exposure subtype '{subtype}' must target S1/S2, got '{stage_target}'")
            for operator in operators:
                if operator.default_stage_target not in ALLOWED_INTERFACE_EXPOSURE_STAGE_TARGETS:
                    raise ValueError(f"Interface exposure operator '{operator.operator_name}' must not target S3/S4")
                forbidden = INTERFACE_EXPOSURE_FORBIDDEN_FAILURES.intersection(operator.primary_failure_risks)
                if forbidden:
                    joined = ", ".join(sorted(forbidden))
                    raise ValueError(f"Interface exposure operator '{operator.operator_name}' declares forbidden failure risks: {joined}")
                if operator.exposure_axis and operator.exposure_axis not in {"name", "description", "family_similarity"}:
                    raise ValueError(f"Interface exposure operator '{operator.operator_name}' has invalid exposure axis '{operator.exposure_axis}'")

    def _validate_interface_exposure_variant(self, original: ToolSpec, perturbed: ToolSpec, spec: VariantSpec) -> None:
        if spec.stage_target not in ALLOWED_INTERFACE_EXPOSURE_STAGE_TARGETS:
            raise ValueError(f"Interface exposure variant '{spec.registry_key}' drifted to forbidden stage '{spec.stage_target}'")
        if original.parameters.model_dump() != perturbed.parameters.model_dump():
            raise ValueError(f"Interface exposure variant '{spec.registry_key}' illegally modified parameter schema for '{original.canonical_tool_id}'")
        if original.parameter_name_map != perturbed.parameter_name_map:
            raise ValueError(f"Interface exposure variant '{spec.registry_key}' illegally modified parameter name mapping for '{original.canonical_tool_id}'")
        if list(original.parameters.properties.keys()) != list(perturbed.parameters.properties.keys()):
            raise ValueError(f"Interface exposure variant '{spec.registry_key}' illegally changed parameter order for '{original.canonical_tool_id}'")

    def _annotate_tool(self, tool: ToolSpec, spec: VariantSpec) -> ToolSpec:
        clone = tool.model_copy(deep=True)
        clone.metadata["v3_noise_family"] = spec.family
        clone.metadata["v3_noise_subtype"] = spec.subtype
        clone.metadata["v3_noise_operator_names"] = [operator.operator_name for operator in spec.operators]
        clone.metadata["v3_noise_stage_target"] = spec.stage_target
        clone.metadata["v3_template_name"] = spec.template_name
        clone.metadata["v3_primary_operator"] = spec.primary_operator
        clone.metadata["v3_amplifier_operator"] = spec.amplifier_operator
        clone.metadata["v3_exposure_axis"] = spec.exposure_axis
        clone.metadata["v3_static_strength_score"] = spec.static_strength_score
        return clone

    def _annotate_composite_tool(self, tool: ToolSpec, spec: CompositeVariantSpec) -> ToolSpec:
        clone = tool.model_copy(deep=True)
        clone.metadata["v3_noise_family"] = BenchmarkFamily.CROSS_FAMILY.value
        clone.metadata["v3_noise_subtype"] = spec.composition_id
        clone.metadata["v3_noise_operator_names"] = [operator.operator_name for operator in spec.operators]
        clone.metadata["v3_noise_stage_target"] = StageTarget.S5.value
        clone.metadata["v3_template_name"] = spec.template_name
        clone.metadata["v3_composition_id"] = spec.composition_id
        clone.metadata["v3_families_applied"] = list(spec.families_applied)
        clone.metadata["v3_subtypes_applied"] = list(spec.subtypes_applied)
        clone.metadata["v3_stage_targets_applied"] = list(spec.stage_targets_applied)
        return clone

    def _build_preflight_gate_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        rows.extend(self._build_interface_exposure_preflight_rows())
        rows.extend(self._build_user_intent_preflight_rows())
        rows.extend(self._build_runtime_preflight_rows())

        for subtype in {row["subtype"] for row in rows}:
            subtype_rows = [row for row in rows if row["subtype"] == subtype]
            if len(subtype_rows) != 3:
                continue
            score_map = {row["severity"]: float(row["static_strength_score"]) for row in subtype_rows}
            monotonic = score_map[SeverityLevel.LIGHT.value] < score_map[SeverityLevel.MEDIUM.value] < score_map[SeverityLevel.HEAVY.value]
            note = self._monotonic_note(score_map)
            for row in subtype_rows:
                row["monotonic_pass"] = monotonic
                row["note"] = self._compose_gate_note(row, note)
        return rows

    def _build_user_intent_preflight_rows(self) -> list[dict[str, object]]:
        if BenchmarkFamily.USER_INTENT.value not in self.enabled_families():
            return []
        disabled = set(self.config.noise.disabled_operators)
        rows: list[dict[str, object]] = []
        for subtype in self.enabled_subtypes():
            if subtype not in USER_INTENT_TEMPLATE_PLAN:
                continue
            for severity in [SeverityLevel.LIGHT.value, SeverityLevel.MEDIUM.value, SeverityLevel.HEAVY.value]:
                template = USER_INTENT_TEMPLATE_PLAN[subtype][severity]
                if any(name in disabled for name in template.operator_names):
                    rows.append({"family": BenchmarkFamily.USER_INTENT.value, "subtype": subtype, "severity": severity, "template_name": template.template_name, "primary_operator": template.primary_operator, "amplifier_operator": template.amplifier_operator, "exposure_axis": template.exposure_axis, "static_strength_score": 0.0, "monotonic_pass": False, "subtype_purity_pass": False, "light_nonzero_pass": False, "note": "template operator disabled"})
                    continue
                operators = [self._operator_index[name] for name in template.operator_names]
                strength_score = min(100.0, 100.0 * sum(operator.intrinsic_strength_score for operator in operators))
                subtype_purity_pass = bool(operators) and all(
                    subtype in operator.subtype_affinity
                    for operator in operators
                    if operator.subtype_affinity
                )
                light_nonzero_pass = severity != SeverityLevel.LIGHT.value or strength_score > LIGHT_NONZERO_THRESHOLD
                rows.append({"family": BenchmarkFamily.USER_INTENT.value, "subtype": subtype, "severity": severity, "template_name": template.template_name, "primary_operator": template.primary_operator, "amplifier_operator": template.amplifier_operator, "exposure_axis": template.exposure_axis, "static_strength_score": round(strength_score, 4), "monotonic_pass": True, "subtype_purity_pass": subtype_purity_pass, "light_nonzero_pass": light_nonzero_pass, "note": "prompt calibration ready"})
        return rows

    def _build_interface_exposure_preflight_rows(self) -> list[dict[str, object]]:
        if BenchmarkFamily.INTERFACE_EXPOSURE.value not in self.enabled_families():
            return []
        clean_tools = RegistryBuilder().build_clean_registry(self.expanded_registry_tool_ids(), include_similar_tool_groups=False).list_tools()
        disabled = set(self.config.noise.disabled_operators)
        rows: list[dict[str, object]] = []
        for subtype in self.enabled_subtypes():
            if subtype not in INTERFACE_EXPOSURE_TEMPLATE_PLAN:
                continue
            for severity in [SeverityLevel.LIGHT.value, SeverityLevel.MEDIUM.value, SeverityLevel.HEAVY.value]:
                template = INTERFACE_EXPOSURE_TEMPLATE_PLAN[subtype][severity]
                operator_names = [template.primary_operator] + ([template.amplifier_operator] if template.amplifier_operator else [])
                if any(name in disabled for name in operator_names):
                    rows.append({"family": BenchmarkFamily.INTERFACE_EXPOSURE.value, "subtype": subtype, "severity": severity, "template_name": template.template_name, "primary_operator": template.primary_operator, "amplifier_operator": template.amplifier_operator or "", "exposure_axis": template.exposure_axis, "static_strength_score": 0.0, "monotonic_pass": False, "subtype_purity_pass": False, "light_nonzero_pass": False, "note": "template operator disabled"})
                    continue
                variant = VariantSpec(family=BenchmarkFamily.INTERFACE_EXPOSURE.value, subtype=subtype, severity=severity, operators=[self._operator_index[name] for name in operator_names], stage_target=SUBTYPE_STAGE_TARGET[subtype], registry_key=f"preflight:{self._slug(subtype)}:{severity}", template_name=template.template_name, primary_operator=template.primary_operator, amplifier_operator=template.amplifier_operator or "", exposure_axis=template.exposure_axis, static_strength_score=0.0)
                perturbed_tools = self._apply_registry_variant(clean_tools, variant)
                structural_score = self._template_strength_score(subtype, clean_tools, perturbed_tools)
                operator_score = sum(operator.intrinsic_strength_score for operator in variant.operators)
                strength_score = min(100.0, structural_score * 0.2 + 80.0 * operator_score)
                subtype_purity_pass = self._template_subtype_purity_pass(variant)
                light_nonzero_pass = severity != SeverityLevel.LIGHT.value or strength_score > LIGHT_NONZERO_THRESHOLD
                rows.append({"family": BenchmarkFamily.INTERFACE_EXPOSURE.value, "subtype": subtype, "severity": severity, "template_name": template.template_name, "primary_operator": template.primary_operator, "amplifier_operator": template.amplifier_operator or "", "exposure_axis": template.exposure_axis, "static_strength_score": round(strength_score, 4), "monotonic_pass": True, "subtype_purity_pass": subtype_purity_pass, "light_nonzero_pass": light_nonzero_pass, "note": "static gate ready"})
        return rows

    def _build_runtime_preflight_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        disabled = set(self.config.noise.disabled_operators)
        for family in (BenchmarkFamily.OBSERVATION.value, BenchmarkFamily.EXECUTION_ENVIRONMENT.value):
            if family not in self.enabled_families():
                continue
            for subtype in self.enabled_subtypes():
                if subtype not in RUNTIME_TEMPLATE_PLAN.get(family, {}):
                    continue
                for severity in [SeverityLevel.LIGHT.value, SeverityLevel.MEDIUM.value, SeverityLevel.HEAVY.value]:
                    template = RUNTIME_TEMPLATE_PLAN[family][subtype][severity]
                    operator_names = template.operator_names
                    if any(name in disabled for name in operator_names):
                        rows.append({"family": family, "subtype": subtype, "severity": severity, "template_name": template.template_name, "primary_operator": template.primary_operator, "amplifier_operator": template.amplifier_operator_label, "exposure_axis": template.exposure_axis, "static_strength_score": 0.0, "monotonic_pass": False, "subtype_purity_pass": False, "light_nonzero_pass": False, "note": "template operator disabled"})
                        continue
                    operators = [self._operator_index[name] for name in operator_names]
                    strength_score = min(100.0, 100.0 * sum(operator.intrinsic_strength_score for operator in operators))
                    subtype_purity_pass = bool(operators) and all(
                        subtype in operator.subtype_affinity
                        for operator in operators
                        if operator.subtype_affinity
                    )
                    light_nonzero_pass = severity != SeverityLevel.LIGHT.value or strength_score > LIGHT_NONZERO_THRESHOLD
                    rows.append({"family": family, "subtype": subtype, "severity": severity, "template_name": template.template_name, "primary_operator": template.primary_operator, "amplifier_operator": template.amplifier_operator_label, "exposure_axis": template.exposure_axis, "static_strength_score": round(strength_score, 4), "monotonic_pass": True, "subtype_purity_pass": subtype_purity_pass, "light_nonzero_pass": light_nonzero_pass, "note": "runtime calibration ready"})
        return rows

    def _assert_preflight_gate(self) -> None:
        failures = [row for row in self._preflight_gate_rows if not row["monotonic_pass"] or not row["subtype_purity_pass"] or not row["light_nonzero_pass"]]
        if failures:
            joined = "; ".join(f"{row['subtype']} {row['severity']}: {row['note']}" for row in failures)
            raise ValueError(f"Preflight gate failed: {joined}")

    def _template_strength_score(self, subtype: str, clean_tools: list[ToolSpec], perturbed_tools: list[ToolSpec]) -> float:
        if subtype == "A1 Name Surface Shift":
            return self._name_exposure_strength(clean_tools, perturbed_tools)
        if subtype == "A2 Description Abstraction":
            return self._description_exposure_strength(clean_tools, perturbed_tools)
        return self._family_similarity_strength(clean_tools, perturbed_tools)

    def _template_subtype_purity_pass(self, spec: VariantSpec) -> bool:
        return bool(spec.operators) and all(
            spec.subtype in operator.subtype_affinity
            for operator in spec.operators
            if operator.subtype_affinity
        )

    def _name_exposure_strength(self, clean_tools: list[ToolSpec], perturbed_tools: list[ToolSpec]) -> float:
        clean_map = {tool.canonical_tool_id: tool for tool in clean_tools}
        perturbed_map = {tool.canonical_tool_id: tool for tool in perturbed_tools}
        scores: list[float] = []
        for tool_id, clean_tool in clean_map.items():
            perturbed_tool = perturbed_map[tool_id]
            name_similarity = self._surface_similarity(clean_tool.clean_tool_name, perturbed_tool.exposed_name)
            function_retention = self._retention_ratio(self._surface_tokens(clean_tool.clean_tool_name), self._surface_tokens(perturbed_tool.exposed_name))
            alias_penalty = 0.0 if perturbed_tool.aliases else 1.0
            sibling_collision = self._max_sibling_name_similarity(tool_id, perturbed_map)
            scores.append(100.0 * (0.45 * (1.0 - name_similarity) + 0.20 * (1.0 - function_retention) + 0.15 * alias_penalty + 0.20 * sibling_collision))
        return sum(scores) / len(scores)

    def _description_exposure_strength(self, clean_tools: list[ToolSpec], perturbed_tools: list[ToolSpec]) -> float:
        clean_map = {tool.canonical_tool_id: tool for tool in clean_tools}
        perturbed_map = {tool.canonical_tool_id: tool for tool in perturbed_tools}
        scores: list[float] = []
        for tool_id, clean_tool in clean_map.items():
            perturbed_tool = perturbed_map[tool_id]
            unique_tokens = self._unique_description_tokens(tool_id, clean_map)
            retained_unique = self._retention_ratio(unique_tokens, self._surface_tokens(perturbed_tool.description))
            function_tokens = self._surface_tokens(clean_tool.clean_tool_name) | {alias.lower() for alias in clean_tool.aliases}
            function_retention = self._retention_ratio(function_tokens, self._surface_tokens(perturbed_tool.description))
            sibling_similarity_gain = max(0.0, self._average_sibling_description_similarity(tool_id, perturbed_map) - self._average_sibling_description_similarity(tool_id, clean_map))
            compression = max(0.0, 1.0 - (len(self._surface_tokens(perturbed_tool.description)) / max(1, len(self._surface_tokens(clean_tool.description)))))
            scores.append(100.0 * (0.40 * (1.0 - retained_unique) + 0.25 * min(1.0, sibling_similarity_gain * 2.5) + 0.25 * (1.0 - function_retention) + 0.10 * compression))
        return sum(scores) / len(scores)

    def _family_similarity_strength(self, clean_tools: list[ToolSpec], perturbed_tools: list[ToolSpec]) -> float:
        clean_map = {tool.canonical_tool_id: tool for tool in clean_tools}
        perturbed_map = {tool.canonical_tool_id: tool for tool in perturbed_tools}
        scores: list[float] = []
        for tool_id in clean_map:
            name_gain = max(0.0, self._average_sibling_name_similarity(tool_id, perturbed_map) - self._average_sibling_name_similarity(tool_id, clean_map))
            description_gain = max(0.0, self._average_sibling_description_similarity(tool_id, perturbed_map) - self._average_sibling_description_similarity(tool_id, clean_map))
            distinctiveness_remaining = self._distinctiveness_remaining_ratio(tool_id, perturbed_map)
            scores.append(100.0 * (0.40 * min(1.0, name_gain * 2.0) + 0.35 * min(1.0, description_gain * 2.0) + 0.25 * (1.0 - distinctiveness_remaining)))
        return sum(scores) / len(scores)

    @staticmethod
    def _monotonic_note(score_map: dict[str, float]) -> str:
        light = score_map[SeverityLevel.LIGHT.value]
        medium = score_map[SeverityLevel.MEDIUM.value]
        heavy = score_map[SeverityLevel.HEAVY.value]
        if not light < medium:
            return "light strength is not lower than medium"
        if not medium < heavy:
            return "medium strength is not lower than heavy"
        return "static strengths are strictly increasing"

    @staticmethod
    def _compose_gate_note(row: dict[str, object], monotonic_note: str) -> str:
        notes: list[str] = [monotonic_note]
        if not row["subtype_purity_pass"]:
            notes.append("subtype purity failed")
        if not row["light_nonzero_pass"]:
            notes.append("light template is too close to clean")
        return "; ".join(notes)

    @staticmethod
    def _surface_tokens(text: str) -> set[str]:
        spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", text.replace("_", " "))
        return {token.lower() for token in re.findall(r"[A-Za-z0-9]+", spaced)}

    def _surface_similarity(self, left: str, right: str) -> float:
        left_tokens = self._surface_tokens(left)
        right_tokens = self._surface_tokens(right)
        token_score = self._jaccard(left_tokens, right_tokens)
        char_score = SequenceMatcher(None, left.lower(), right.lower()).ratio()
        return (token_score + char_score) / 2.0

    @staticmethod
    def _jaccard(left: set[str], right: set[str]) -> float:
        if not left and not right:
            return 1.0
        return len(left & right) / max(1, len(left | right))

    @staticmethod
    def _retention_ratio(reference_tokens: set[str], current_tokens: set[str]) -> float:
        if not reference_tokens:
            return 1.0
        return len(reference_tokens & current_tokens) / len(reference_tokens)

    def _unique_description_tokens(self, tool_id: str, tool_map: dict[str, ToolSpec]) -> set[str]:
        own = self._surface_tokens(tool_map[tool_id].description)
        sibling_union: set[str] = set()
        for sibling_id in self._tool_family_neighbors(tool_id):
            sibling_union |= self._surface_tokens(tool_map[sibling_id].description)
        return own - sibling_union

    def _max_sibling_name_similarity(self, tool_id: str, tool_map: dict[str, ToolSpec]) -> float:
        scores = [self._surface_similarity(tool_map[tool_id].exposed_name, tool_map[sibling_id].exposed_name) for sibling_id in self._tool_family_neighbors(tool_id)]
        return max(scores) if scores else 0.0

    def _average_sibling_name_similarity(self, tool_id: str, tool_map: dict[str, ToolSpec]) -> float:
        scores = [self._surface_similarity(tool_map[tool_id].exposed_name, tool_map[sibling_id].exposed_name) for sibling_id in self._tool_family_neighbors(tool_id)]
        return sum(scores) / len(scores) if scores else 0.0

    def _average_sibling_description_similarity(self, tool_id: str, tool_map: dict[str, ToolSpec]) -> float:
        scores = [self._jaccard(self._surface_tokens(tool_map[tool_id].description), self._surface_tokens(tool_map[sibling_id].description)) for sibling_id in self._tool_family_neighbors(tool_id)]
        return sum(scores) / len(scores) if scores else 0.0

    def _distinctiveness_remaining_ratio(self, tool_id: str, tool_map: dict[str, ToolSpec]) -> float:
        own = self._surface_tokens(tool_map[tool_id].exposed_name) | self._surface_tokens(tool_map[tool_id].description)
        sibling_union: set[str] = set()
        for sibling_id in self._tool_family_neighbors(tool_id):
            sibling_union |= self._surface_tokens(tool_map[sibling_id].exposed_name)
            sibling_union |= self._surface_tokens(tool_map[sibling_id].description)
        unique_tokens = own - sibling_union
        reference_unique = self._surface_tokens(tool_map[tool_id].description)
        if not reference_unique:
            return 1.0
        return len(unique_tokens) / len(reference_unique | unique_tokens)

    @staticmethod
    def _tool_family_neighbors(tool_id: str) -> list[str]:
        for group in TOOL_GROUPS.values():
            if tool_id in group:
                return [candidate for candidate in group if candidate != tool_id]
        return []

    @staticmethod
    def _tool_group_for(tool_id: str) -> str:
        for group_name, group_tools in TOOL_GROUPS.items():
            if tool_id in group_tools:
                return group_name
        return "misc"

    def _reduce_distinctiveness(self, description: str, tool_id: str, level: str) -> str:
        generic_map = {
            "get_weather": "Return the requested environment-facing result.",
            "get_air_quality": "Return the requested environment-facing result.",
            "get_temperature": "Return the requested environment-facing result.",
            "convert_unit": "Return the requested conversion result.",
            "convert_currency": "Return the requested conversion result.",
            "search_document": "Return the requested retrieval result.",
            "search_policy": "Return the requested retrieval result.",
            "search_faq": "Return the requested retrieval result.",
            "calculate_route": "Return the requested travel result.",
            "estimate_travel_time": "Return the requested travel result.",
            "sort_items": "Return the requested collection result.",
            "filter_items": "Return the requested collection result.",
        }
        if level == "light":
            return description.replace(" not ", " ").replace(" only ", " ").strip()
        if level == "medium":
            return generic_map.get(tool_id, self._generic_medium_description(tool_id))
        raise ValueError(f"Unsupported reduction level: {level}")

    @classmethod
    def _generic_opaque_name(cls, tool_id: str) -> str:
        group_name, offset = cls._generic_group_position(tool_id)
        return f"{group_name}_signal_{offset:02d}"

    @staticmethod
    def _opaque_internal_name(tool_id: str) -> str:
        index = 1
        for group_tools in TOOL_GROUPS.values():
            for candidate in group_tools:
                if candidate == tool_id:
                    return f"svc_{index:02d}"
                index += 1
        return "svc_99"

    @classmethod
    def _generic_collision_name(cls, tool_id: str) -> str:
        group_name, offset = cls._generic_group_position(tool_id)
        return f"{group_name}_node_{offset:02d}"

    @classmethod
    def _generic_light_description(cls, tool_id: str) -> str:
        group_name = cls._tool_group_for(tool_id).replace("_", " ")
        return f"Handle the requested {group_name} operation for the given entity, scope, and attribute."

    @classmethod
    def _generic_medium_description(cls, tool_id: str) -> str:
        group_name = cls._tool_group_for(tool_id).replace("_", " ")
        return f"Return the requested {group_name} result."

    @staticmethod
    def _generic_group_position(tool_id: str) -> tuple[str, int]:
        for group_name, group_tools in TOOL_GROUPS.items():
            if tool_id in group_tools:
                return group_name, group_tools.index(tool_id) + 1
        return "misc", 1

    def _cross_tool_description(self, tool_id: str, clean_tools: list[ToolSpec], weight: float) -> str:
        current = next(tool.description for tool in clean_tools if tool.canonical_tool_id == tool_id)
        neighbors = self._tool_family_neighbors(tool_id)
        if not neighbors:
            return f"{current.rstrip('.')} Neighboring tool cues are intentionally blended into this interface wording."
        sibling_id = neighbors[0]
        sibling_description = next(tool.description for tool in clean_tools if tool.canonical_tool_id == sibling_id)
        return (
            f"{current.rstrip('.')} "
            f"Sibling-facing cues are blended with neighboring interface wording: {sibling_description.rstrip('.')}. "
            "The family-level wording is intentionally overlapping, so the caller must separate this tool from nearby alternatives."
        )

    @staticmethod
    def _slug(value: str) -> str:
        return value.lower().replace(" ", "_")

    @staticmethod
    def _family_short_name(value: str) -> str:
        return {
            BenchmarkFamily.INTERFACE_EXPOSURE.value: "a",
            BenchmarkFamily.USER_INTENT.value: "b",
            BenchmarkFamily.OBSERVATION.value: "c",
            BenchmarkFamily.EXECUTION_ENVIRONMENT.value: "d",
        }.get(value, value)
