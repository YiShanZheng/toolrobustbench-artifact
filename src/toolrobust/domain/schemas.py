"""Core schemas for ToolRobust v3."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from toolrobust.domain.enums import BenchmarkFamily, DatasetSplit, FailureTag, OperatorStrength, OutputProfile, SeverityLevel, StageTarget


class ToolParameterSchema(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    type: str = "object"
    properties: dict[str, dict[str, Any]]
    required: list[str] = Field(default_factory=list)
    additionalProperties: bool = False


class ToolSpec(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    canonical_tool_id: str
    exposed_name: str
    clean_tool_name: str
    description: str
    parameters: ToolParameterSchema
    aliases: list[str] = Field(default_factory=list)
    parameter_name_map: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskSample(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    sample_id: str
    split: DatasetSplit
    family: str
    subtype: str
    operator: str
    severity: SeverityLevel
    stage_target: str
    registry_variant_id: str
    canonical_tool_id: str
    exposed_tool_name: str
    clean_tool_name: str
    task_text: str
    user_instruction: str
    expected_arguments: dict[str, Any]
    expected_result: Any | None = None
    operators_applied: list[str] = Field(default_factory=list)
    composition_id: str | None = None
    families_applied: list[str] = Field(default_factory=list)
    subtypes_applied: list[str] = Field(default_factory=list)
    stage_targets_applied: list[str] = Field(default_factory=list)
    primary_family: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def active_families(self) -> list[str]:
        if self.families_applied:
            return list(self.families_applied)
        if self.family == BenchmarkFamily.CLEAN.value:
            return []
        return [self.family]

    def has_family(self, family: str) -> bool:
        return family in self.active_families()

    def component_subtype(self, family: str) -> str | None:
        prefix_by_family = {
            BenchmarkFamily.INTERFACE_EXPOSURE.value: "A",
            BenchmarkFamily.USER_INTENT.value: "B",
            BenchmarkFamily.OBSERVATION.value: "C",
            BenchmarkFamily.EXECUTION_ENVIRONMENT.value: "D",
        }
        prefix = prefix_by_family.get(family)
        if prefix:
            for subtype in self.subtypes_applied:
                if subtype.startswith(prefix):
                    return subtype
        if self.family == family:
            return self.subtype
        return None


class ParsedInvocation(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    predicted_tool_raw: str | None = None
    predicted_args_raw: dict[str, Any] = Field(default_factory=dict)
    call_present: bool = False
    format_valid: bool = False
    parser_message: str = ""


class BackendResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    backend_name: str
    model_name: str
    raw_response: dict[str, Any]
    parsed: ParsedInvocation
    latency_ms: float | None = None
    cost_usd: float | None = None
    auxiliary: dict[str, Any] = Field(default_factory=dict)


class EvaluationRecord(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    sample_id: str
    model_name: str
    backend_name: str
    split: DatasetSplit
    family: str
    subtype: str
    operator: str
    severity: SeverityLevel
    stage_target: str
    source_sample_id: str
    registry_variant_id: str
    composition_id: str | None = None
    families_applied: list[str] = Field(default_factory=list)
    subtypes_applied: list[str] = Field(default_factory=list)
    stage_targets_applied: list[str] = Field(default_factory=list)
    primary_family: str | None = None
    gold_canonical_tool_id: str
    gold_exposed_tool_name: str
    operator_names: list[str] = Field(default_factory=list)
    expected_arguments: dict[str, Any] = Field(default_factory=dict)
    predicted_arguments: dict[str, Any] = Field(default_factory=dict)
    expected_result: Any | None = None
    predicted_tool_raw: str | None
    predicted_tool_resolved: str | None
    call_present: int
    interface_name_match: int
    canonical_tool_match: int
    format_valid: int
    missing_parameter_flag: int
    wrong_parameter_flag: int
    execution_success: int
    overall_task_success: int
    decision_status: str | None = None
    decision_retryable: bool | None = None
    decision_failure_type: str | None = None
    decision_final_result_summary: str | None = None
    expected_statuses: str | None = None
    expected_retryable: bool | None = None
    expected_failure_types: str | None = None
    expected_final_result_summary: str | None = None
    stage2_status_match: bool | None = None
    stage2_retryable_match: bool | None = None
    stage2_failure_type_match: bool | None = None
    stage2_result_match: bool | None = None
    status: str
    observed_failure_tag: FailureTag | None = None
    primary_failure_tag: FailureTag | None = None
    first_failure_stage: str | None = None
    is_cascade_failure: bool = False
    cascade_from_stage: str | None = None
    expected_primary_match: bool = False
    allowed_spillover_match: bool = False
    boundary_violation_flag: bool = False
    boundary_violation_reason: str | None = None
    secondary_reason: str | None = None
    notes: str = ""
    latency_ms: float | None = None
    cost_usd: float | None = None


class BackendConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    temperature: float = 0.0
    max_tokens: int = 512
    api_key_env: str | None = None
    base_url_env: str | None = None
    base_url: str | None = None
    request_timeout_sec: int | None = None
    request_max_retries: int | None = None
    structured_stage2_response: bool | None = None
    stage2_max_tokens: int | None = None
    tool_choice: str | dict[str, Any] | None = "required"
    extra_body: dict[str, Any] = Field(default_factory=dict)


class ExperimentSection(BaseModel):
    name: str = ""
    description: str = ""


class ModelsSection(BaseModel):
    include: list[str]


class DatasetSection(BaseModel):
    source: str
    task_scope: str
    max_clean_per_tool: int = 3
    sample_tool_count: int | None = None
    seed: int = 42

    @field_validator("sample_tool_count")
    @classmethod
    def _validate_sample_tool_count(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("dataset.sample_tool_count must be positive when set")
        return value


class NoiseSection(BaseModel):
    enabled_families: list[str]
    enabled_subtypes: list[str] | str = "all"
    disabled_operators: list[str] = Field(default_factory=list)
    severities: list[SeverityLevel]
    severity_policy: str = "standard"
    allow_cross_family_heavy: bool = False
    max_operators_per_sample: int = 2
    samples_per_bucket: int = 3

    @field_validator("enabled_subtypes", mode="before")
    @classmethod
    def _coerce_subtypes(cls, value: Any) -> list[str] | str:
        if value == "all":
            return "all"
        if isinstance(value, str):
            return [value]
        return list(value)


class CrossFamilySection(BaseModel):
    enabled: bool = False
    strategy: str = "pairwise"
    family_sets: list[list[str]] = Field(default_factory=list)
    enabled_compositions: list[str] | str = "all"
    severities: list[SeverityLevel] = Field(
        default_factory=lambda: [SeverityLevel.MEDIUM, SeverityLevel.HEAVY]
    )
    samples_per_combo_bucket: int = 1
    max_families_per_sample: int = 2
    include_single_family_baseline: bool = True

    @field_validator("family_sets", mode="before")
    @classmethod
    def _coerce_family_sets(cls, value: Any) -> list[list[str]]:
        if value in (None, "all"):
            return []
        return [list(item) for item in value]

    @field_validator("enabled_compositions", mode="before")
    @classmethod
    def _coerce_enabled_compositions(cls, value: Any) -> list[str] | str:
        if value in (None, "all"):
            return "all"
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return [str(item) for item in value]

    @model_validator(mode="after")
    def _validate(self) -> "CrossFamilySection":
        if self.strategy != "pairwise":
            raise ValueError("cross_family.strategy currently supports only 'pairwise'")
        if self.samples_per_combo_bucket <= 0:
            raise ValueError("cross_family.samples_per_combo_bucket must be positive")
        if self.max_families_per_sample != 2:
            raise ValueError("cross_family.max_families_per_sample must be 2 for pairwise mode")
        for family_set in self.family_sets:
            if len(family_set) != 2:
                raise ValueError("cross_family.family_sets must contain two-family sets in pairwise mode")
        return self


class ExecutionSection(BaseModel):
    timeout_sec: int = 60
    retry_times: int = 0


class OutputSection(BaseModel):
    profile: OutputProfile = OutputProfile.LITE
    clean_output: bool = True
    generate_plots: bool = True
    keep_missing_buckets: bool = True
    severity_order: list[str] = Field(default_factory=lambda: [level.value for level in SeverityLevel])


class BenchmarkV3Config(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    benchmark_version: str = "v3"
    experiment: ExperimentSection
    models: ModelsSection
    dataset: DatasetSection
    noise: NoiseSection
    cross_family: CrossFamilySection = Field(default_factory=CrossFamilySection)
    execution: ExecutionSection = Field(default_factory=ExecutionSection)
    output: OutputSection = Field(default_factory=OutputSection)
    output_root: Path = Path("data/results")
    generated_dir: Path = Path("data/generated")

    @field_validator("output_root", "generated_dir", mode="before")
    @classmethod
    def _coerce_path(cls, value: str | Path) -> Path:
        return Path(value)

    @model_validator(mode="after")
    def _validate(self) -> "BenchmarkV3Config":
        if self.benchmark_version != "v3":
            raise ValueError("benchmark_version must be v3")
        if not self.models.include:
            raise ValueError("models.include must not be empty")
        return self

    def resolved_experiment_name(self) -> str:
        name = self.experiment.name.strip()
        if name:
            return name
        return self._auto_experiment_name()

    def _auto_experiment_name(self) -> str:
        primary_model = self.models.include[0] if self.models.include else "unknown-model"
        family_token = self._family_abbreviation_token()
        total_samples = self._estimated_total_sample_count()
        return f"benchmark_v3_{primary_model}_{self.dataset.task_scope}_{family_token}_n{total_samples}"

    def _family_abbreviation_token(self) -> str:
        family_map = {
            "interface_exposure": "a",
            "user_intent": "b",
            "observation": "c",
            "execution_environment": "d",
        }
        ordered = ["interface_exposure", "user_intent", "observation", "execution_environment"]
        abbreviations = [family_map[family] for family in ordered if family in self.noise.enabled_families]
        if self.cross_family.enabled:
            abbreviations.append("x")
        if not abbreviations:
            return "fam-none"
        return f"fam-{''.join(abbreviations)}"

    def _estimated_total_sample_count(self) -> int:
        from toolrobust.datasets.builders import DatasetBuilder
        from toolrobust.perturbation.pipeline import PerturbationPipeline

        builder = DatasetBuilder(self)
        scenario_bank = builder._scenario_bank()
        selected_tool_ids = builder.selected_tool_ids()
        clean_samples = sum(min(len(scenario_bank[tool_id]), self.dataset.max_clean_per_tool) for tool_id in selected_tool_ids)
        pipeline = PerturbationPipeline(self)
        perturbed_samples = 0
        if not self.cross_family.enabled or self.cross_family.include_single_family_baseline:
            perturbed_samples += clean_samples * len(pipeline.variant_specs()) * self.noise.samples_per_bucket
        perturbed_samples += clean_samples * len(pipeline.composite_variant_specs()) * self.cross_family.samples_per_combo_bucket
        return clean_samples + perturbed_samples

    @property
    def clean_dataset_path(self) -> Path:
        return self.generated_dir / "clean_tasks_v3.jsonl"

    @property
    def perturbed_dataset_path(self) -> Path:
        return self.generated_dir / "perturbed_tasks_v3.jsonl"

    @property
    def export_path(self) -> Path:
        return self.generated_dir / "llamafactory_sharegpt_v3.json"


class OperatorSpec(BaseModel):
    family: str
    subtype: str
    operator_name: str
    default_strength: OperatorStrength
    default_stage_target: str
    primary_failure_risks: list[str]
    composable_within_family: bool = True
    intrinsic_strength_score: float = 0.0
    subtype_affinity: list[str] = Field(default_factory=list)
    severity_fit: list[str] = Field(default_factory=list)
    exposure_axis: str = ""
    stack_role: str = ""
    can_stack_with: list[str] = Field(default_factory=list)
    should_not_stack_with: list[str] = Field(default_factory=list)
    minimum_effect_expectation: float = 0.0
    maximum_effect_expectation: float = 1.0
    notes: str = ""


class RunManifest(BaseModel):
    benchmark_version: str
    operator_catalog_version: str
    experiment_name: str
    experiment_description: str
    models: list[str]
    dataset: dict[str, Any]
    enabled_families: list[str]
    enabled_subtypes: list[str] | str
    disabled_operators: list[str] = Field(default_factory=list)
    severities: list[str]
    severity_policy: str
    allow_cross_family_heavy: bool
    cross_family: dict[str, Any] = Field(default_factory=dict)
    max_operators_per_sample: int
    samples_per_bucket: int
    seed: int
    output_profile: str
    backend_map: dict[str, str]
    severity_order: list[str]
    family_order: list[str]
    subtype_order: list[str]
    subtype_family_mapping: dict[str, str] = Field(default_factory=dict)
    stage_order: list[str]
    stage_mapping: dict[str, str]
    family_failure_profiles: dict[str, dict[str, object]] = Field(default_factory=dict)
    stage_failure_profiles: dict[str, dict[str, object]] = Field(default_factory=dict)
    operator_primary_failure_risks: dict[str, list[str]]
    operator_metadata: dict[str, dict[str, object]] = Field(default_factory=dict)
    preflight_gate_rows: list[dict[str, object]] = Field(default_factory=list)


def default_clean_axes() -> dict[str, str]:
    return {
        "family": BenchmarkFamily.CLEAN.value,
        "subtype": "clean",
        "operator": "clean",
        "stage_target": StageTarget.CLEAN.value,
    }
