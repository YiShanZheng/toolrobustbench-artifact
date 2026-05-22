"""Shared family/stage failure profiles for ToolRobust v3."""

from __future__ import annotations

from dataclasses import dataclass

from toolrobust.domain.enums import BenchmarkFamily, FailureTag, StageTarget
from toolrobust.domain.schemas import TaskSample


@dataclass(frozen=True, slots=True)
class FailureProfile:
    expected_primary_failures: tuple[str, ...]
    allowed_spillover_failures: tuple[str, ...]
    disallowed_direct_perturbations: tuple[str, ...]
    explanation: str

    def to_manifest_dict(self) -> dict[str, object]:
        return {
            "expected_primary_failures": list(self.expected_primary_failures),
            "allowed_spillover_failures": list(self.allowed_spillover_failures),
            "disallowed_direct_perturbations": list(self.disallowed_direct_perturbations),
            "explanation": self.explanation,
        }


FAMILY_FAILURE_PROFILES: dict[str, FailureProfile] = {
    BenchmarkFamily.INTERFACE_EXPOSURE.value: FailureProfile(
        expected_primary_failures=(FailureTag.WRONG_TOOL.value,),
        allowed_spillover_failures=(FailureTag.MISSING_PARAMETER.value, FailureTag.WRONG_PARAMETER.value),
        disallowed_direct_perturbations=(
            "parameter schema semantic change",
            "parameter key rename",
            "required/default/order modification",
            "parameter-level grounding drift",
        ),
        explanation="Interface exposure perturbs names, aliases, descriptions, and separability. Parameter errors may appear downstream as spillover, but direct schema-level changes are out of bounds.",
    ),
    BenchmarkFamily.USER_INTENT.value: FailureProfile(
        expected_primary_failures=(FailureTag.NO_CALL.value, FailureTag.WRONG_TOOL.value),
        allowed_spillover_failures=(FailureTag.MISSING_PARAMETER.value, FailureTag.WRONG_PARAMETER.value),
        disallowed_direct_perturbations=(
            "tool registry / schema level mutation",
            "execution environment state mutation",
        ),
        explanation="User-intent perturbations directly distort request interpretation. Parameter mistakes are allowed as downstream spillover, but registry or environment mutations are disallowed direct effects.",
    ),
    BenchmarkFamily.OBSERVATION.value: FailureProfile(
        expected_primary_failures=(
            FailureTag.INVALID_FORMAT.value,
            FailureTag.EXECUTION_FAIL.value,
            FailureTag.WRONG_PARAMETER.value,
        ),
        allowed_spillover_failures=(FailureTag.NO_CALL.value,),
        disallowed_direct_perturbations=(
            "user intent rewrite",
            "tool interface exposure mutation",
        ),
        explanation="Observation perturbations directly affect returned data and execution handling. A downstream no-call can be cascade, but request rewriting or interface-surface mutation is out of bounds.",
    ),
    BenchmarkFamily.EXECUTION_ENVIRONMENT.value: FailureProfile(
        expected_primary_failures=(
            FailureTag.EXECUTION_FAIL.value,
            FailureTag.INVALID_FORMAT.value,
            FailureTag.NO_CALL.value,
        ),
        allowed_spillover_failures=(FailureTag.WRONG_PARAMETER.value,),
        disallowed_direct_perturbations=(
            "user request semantics rewrite",
            "interface exposure mutation",
            "parameter schema semantic change",
        ),
        explanation="Execution-environment perturbations directly affect runtime behavior and error surfaces. Wrong-parameter may appear as spillover, but request or interface mutations are disallowed direct effects.",
    ),
}


STAGE_FAILURE_PROFILES: dict[str, FailureProfile] = {
    StageTarget.S1.value: FailureProfile(
        expected_primary_failures=(FailureTag.WRONG_TOOL.value, FailureTag.NO_CALL.value),
        allowed_spillover_failures=(FailureTag.MISSING_PARAMETER.value, FailureTag.WRONG_PARAMETER.value),
        disallowed_direct_perturbations=(),
        explanation="S1 captures the earliest selection failure. Parameter mistakes can still show up later as spillover after a confused selection path.",
    ),
    StageTarget.S2.value: FailureProfile(
        expected_primary_failures=(
            FailureTag.WRONG_TOOL.value,
            FailureTag.MISSING_PARAMETER.value,
            FailureTag.WRONG_PARAMETER.value,
        ),
        allowed_spillover_failures=(FailureTag.NO_CALL.value,),
        disallowed_direct_perturbations=(),
        explanation="S2 covers interface grounding and description understanding. Wrong-tool and parameter-binding mistakes can both surface here, while no-call is treated as spillover from unresolved grounding.",
    ),
    StageTarget.S3.value: FailureProfile(
        expected_primary_failures=(FailureTag.MISSING_PARAMETER.value, FailureTag.WRONG_PARAMETER.value),
        allowed_spillover_failures=(FailureTag.INVALID_FORMAT.value,),
        disallowed_direct_perturbations=(),
        explanation="S3 is the binding stage. Invalid-format may appear as spillover when parameter binding destabilizes the emitted call.",
    ),
    StageTarget.S4.value: FailureProfile(
        expected_primary_failures=(
            FailureTag.INVALID_FORMAT.value,
            FailureTag.EXECUTION_FAIL.value,
            FailureTag.WRONG_PARAMETER.value,
        ),
        allowed_spillover_failures=(FailureTag.NO_CALL.value,),
        disallowed_direct_perturbations=(),
        explanation="S4 captures observation/execution handling. A no-call can still be a cascade when the model gives up after unstable execution feedback.",
    ),
}


FAMILY_ALLOWED_DIRECT_STAGES: dict[str, frozenset[str]] = {
    BenchmarkFamily.INTERFACE_EXPOSURE.value: frozenset({StageTarget.S1.value, StageTarget.S2.value}),
    BenchmarkFamily.USER_INTENT.value: frozenset({StageTarget.S1.value, StageTarget.S2.value, StageTarget.S3.value}),
    BenchmarkFamily.OBSERVATION.value: frozenset({StageTarget.S4.value}),
    BenchmarkFamily.EXECUTION_ENVIRONMENT.value: frozenset({StageTarget.S4.value}),
}


DISALLOWED_OPERATOR_NAMES: dict[str, frozenset[str]] = {
    BenchmarkFamily.INTERFACE_EXPOSURE.value: frozenset(
        {
            "parameter_rename",
            "parameter_reorder",
            "required_hint_shift",
            "default_hint_shift",
            "description_compression",
            "cross_tool_description_swap",
        }
    )
}


def family_failure_profiles_payload() -> dict[str, dict[str, object]]:
    return {name: profile.to_manifest_dict() for name, profile in FAMILY_FAILURE_PROFILES.items()}


def stage_failure_profiles_payload() -> dict[str, dict[str, object]]:
    return {name: profile.to_manifest_dict() for name, profile in STAGE_FAILURE_PROFILES.items()}


def detect_direct_perturbation_violation(task: TaskSample) -> tuple[bool, str | None]:
    if task.family == BenchmarkFamily.CLEAN.value:
        return False, None
    if task.family == BenchmarkFamily.CROSS_FAMILY.value:
        return False, None
    allowed_stages = FAMILY_ALLOWED_DIRECT_STAGES.get(task.family)
    if allowed_stages is not None and task.stage_target not in allowed_stages:
        return True, f"family `{task.family}` produced stage target `{task.stage_target}` outside allowed direct stages"
    disallowed_ops = DISALLOWED_OPERATOR_NAMES.get(task.family, frozenset())
    violating_ops = sorted(set(task.operators_applied).intersection(disallowed_ops))
    if violating_ops:
        return True, f"family `{task.family}` used disallowed direct operator(s): {', '.join(violating_ops)}"
    return False, None
