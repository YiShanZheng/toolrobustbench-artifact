"""Observed-failure and cascade-aware failure analysis for ToolRobust v3."""

from __future__ import annotations

from dataclasses import dataclass

from toolrobust.domain.enums import BenchmarkFamily, FailureTag, StageTarget
from toolrobust.domain.schemas import TaskSample
from toolrobust.evaluation.failure_profiles import (
    FAMILY_FAILURE_PROFILES,
    STAGE_FAILURE_PROFILES,
    detect_direct_perturbation_violation,
)


@dataclass(slots=True)
class FailureDecision:
    status: str
    observed_failure_tag: FailureTag | None
    secondary_reason: str | None


@dataclass(slots=True)
class FailureAnalysis:
    status: str
    observed_failure_tag: FailureTag | None
    primary_failure_tag: FailureTag | None
    first_failure_stage: str | None
    is_cascade_failure: bool
    cascade_from_stage: str | None
    expected_primary_match: bool
    allowed_spillover_match: bool
    boundary_violation_flag: bool
    boundary_violation_reason: str | None
    secondary_reason: str | None


def decide_observed_failure(
    *,
    call_present: int,
    format_valid: int,
    predicted_tool_raw: str | None,
    predicted_tool_resolved: str | None,
    interface_name_match: int,
    canonical_tool_match: int,
    missing_parameter_flag: int,
    wrong_parameter_flag: int,
    execution_success: int,
    notes: str,
) -> FailureDecision:
    if overall_success(canonical_tool_match, format_valid, missing_parameter_flag, wrong_parameter_flag, execution_success, call_present):
        return FailureDecision(status="success", observed_failure_tag=None, secondary_reason=None)
    if call_present == 0:
        return FailureDecision(status="failure", observed_failure_tag=FailureTag.NO_CALL, secondary_reason="no_tool_call")
    if format_valid == 0:
        return FailureDecision(
            status="failure",
            observed_failure_tag=FailureTag.INVALID_FORMAT,
            secondary_reason="invalid_json_arguments" if "json" in notes.lower() else "unparsable_tool_call",
        )
    if predicted_tool_resolved is None:
        return FailureDecision(
            status="failure",
            observed_failure_tag=FailureTag.WRONG_TOOL,
            secondary_reason="alias_not_resolved" if predicted_tool_raw else "unknown_tool_name",
        )
    if canonical_tool_match == 0:
        return FailureDecision(status="failure", observed_failure_tag=FailureTag.WRONG_TOOL, secondary_reason="wrong_canonical_tool")
    if interface_name_match == 0:
        return FailureDecision(status="failure", observed_failure_tag=FailureTag.WRONG_TOOL, secondary_reason="wrong_exposed_tool_name")
    if missing_parameter_flag == 1:
        return FailureDecision(
            status="failure",
            observed_failure_tag=FailureTag.MISSING_PARAMETER,
            secondary_reason="missing_required_parameter",
        )
    if wrong_parameter_flag == 1:
        return FailureDecision(
            status="failure",
            observed_failure_tag=FailureTag.WRONG_PARAMETER,
            secondary_reason="parameter_value_mismatch",
        )
    return FailureDecision(status="failure", observed_failure_tag=FailureTag.EXECUTION_FAIL, secondary_reason="runtime_execution_error")


def analyze_failure(
    *,
    task: TaskSample,
    decision: FailureDecision,
    call_present: int,
    canonical_tool_match: int,
    format_valid: int,
    interface_name_match: int,
    missing_parameter_flag: int,
    wrong_parameter_flag: int,
    execution_success: int,
    notes: str,
) -> FailureAnalysis:
    if decision.status == "success":
        return FailureAnalysis(
            status="success",
            observed_failure_tag=None,
            primary_failure_tag=None,
            first_failure_stage=None,
            is_cascade_failure=False,
            cascade_from_stage=None,
            expected_primary_match=False,
            allowed_spillover_match=False,
            boundary_violation_flag=False,
            boundary_violation_reason=None,
            secondary_reason=None,
        )

    observed_tag = decision.observed_failure_tag
    assert observed_tag is not None
    direct_violation, direct_violation_reason = detect_direct_perturbation_violation(task)
    observed_stage = _infer_observed_stage(
        task=task,
        observed_failure_tag=observed_tag,
        call_present=call_present,
    )

    family_profile = _family_profile_for_task(task)
    if family_profile is None:
        primary_tag = observed_tag
        first_stage = observed_stage
        is_cascade_failure = False
    elif observed_tag.value in family_profile.allowed_spillover_failures:
        primary_tag, first_stage = _infer_upstream_primary_failure(
            task=task,
            observed_failure_tag=observed_tag,
            call_present=call_present,
            canonical_tool_match=canonical_tool_match,
            format_valid=format_valid,
            interface_name_match=interface_name_match,
            missing_parameter_flag=missing_parameter_flag,
            wrong_parameter_flag=wrong_parameter_flag,
            execution_success=execution_success,
            notes=notes,
        )
        is_cascade_failure = True
    else:
        primary_tag = observed_tag
        first_stage = observed_stage
        is_cascade_failure = False

    first_stage_profile = STAGE_FAILURE_PROFILES.get(first_stage or "")
    observed_stage_profile = STAGE_FAILURE_PROFILES.get(observed_stage)

    expected_primary_match = bool(
        family_profile
        and primary_tag.value in family_profile.expected_primary_failures
        and first_stage_profile
        and primary_tag.value in first_stage_profile.expected_primary_failures
    )
    allowed_spillover_match = bool(
        family_profile
        and is_cascade_failure
        and observed_tag.value in family_profile.allowed_spillover_failures
        and observed_stage_profile
        and observed_tag.value in observed_stage_profile.expected_primary_failures
    )

    boundary_violation_flag = False
    boundary_violation_reason = None
    if direct_violation:
        boundary_violation_flag = True
        boundary_violation_reason = direct_violation_reason
    elif expected_primary_match:
        boundary_violation_flag = False
    elif allowed_spillover_match:
        boundary_violation_flag = False
    else:
        boundary_violation_flag = True
        if family_profile is None:
            boundary_violation_reason = "clean or unknown family produced a failure outside the perturbation boundary model"
        else:
            boundary_violation_reason = (
                f"primary failure `{primary_tag.value}` does not match expected primary failures for family `{task.family}`"
            )

    return FailureAnalysis(
        status="failure",
        observed_failure_tag=observed_tag,
        primary_failure_tag=primary_tag,
        first_failure_stage=first_stage,
        is_cascade_failure=is_cascade_failure,
        cascade_from_stage=first_stage if is_cascade_failure else None,
        expected_primary_match=expected_primary_match,
        allowed_spillover_match=allowed_spillover_match,
        boundary_violation_flag=boundary_violation_flag,
        boundary_violation_reason=boundary_violation_reason,
        secondary_reason=decision.secondary_reason,
    )


def overall_success(
    canonical_tool_match: int,
    format_valid: int,
    missing_parameter_flag: int,
    wrong_parameter_flag: int,
    execution_success: int,
    call_present: int,
) -> bool:
    return all(
        [
            call_present == 1,
            canonical_tool_match == 1,
            format_valid == 1,
            missing_parameter_flag == 0,
            wrong_parameter_flag == 0,
            execution_success == 1,
        ]
    )


def _infer_observed_stage(*, task: TaskSample, observed_failure_tag: FailureTag, call_present: int) -> str:
    if observed_failure_tag in {FailureTag.INVALID_FORMAT, FailureTag.EXECUTION_FAIL}:
        return StageTarget.S4.value
    if observed_failure_tag in {FailureTag.MISSING_PARAMETER, FailureTag.WRONG_PARAMETER}:
        return StageTarget.S3.value
    if observed_failure_tag == FailureTag.WRONG_TOOL:
        return StageTarget.S1.value if task.stage_target == StageTarget.S1.value else StageTarget.S2.value
    if observed_failure_tag == FailureTag.NO_CALL and task.family in {
        BenchmarkFamily.OBSERVATION.value,
        BenchmarkFamily.EXECUTION_ENVIRONMENT.value,
    }:
        return StageTarget.S4.value
    if observed_failure_tag == FailureTag.NO_CALL and task.family == BenchmarkFamily.CROSS_FAMILY.value and (
        task.has_family(BenchmarkFamily.OBSERVATION.value) or task.has_family(BenchmarkFamily.EXECUTION_ENVIRONMENT.value)
    ):
        return StageTarget.S4.value
    if observed_failure_tag == FailureTag.NO_CALL and task.stage_target == StageTarget.S1.value:
        return StageTarget.S1.value
    if observed_failure_tag == FailureTag.NO_CALL and call_present == 0:
        return StageTarget.S2.value if task.stage_target == StageTarget.S2.value else task.stage_target
    return task.stage_target


def _infer_upstream_primary_failure(
    *,
    task: TaskSample,
    observed_failure_tag: FailureTag,
    call_present: int,
    canonical_tool_match: int,
    format_valid: int,
    interface_name_match: int,
    missing_parameter_flag: int,
    wrong_parameter_flag: int,
    execution_success: int,
    notes: str,
) -> tuple[FailureTag, str]:
    if task.family == BenchmarkFamily.INTERFACE_EXPOSURE.value:
        stage = StageTarget.S1.value if task.stage_target == StageTarget.S1.value else StageTarget.S2.value
        return FailureTag.WRONG_TOOL, stage
    if task.family == BenchmarkFamily.USER_INTENT.value:
        if call_present == 0:
            return FailureTag.NO_CALL, StageTarget.S1.value if task.stage_target == StageTarget.S1.value else StageTarget.S2.value
        if canonical_tool_match == 0 or interface_name_match == 0:
            return FailureTag.WRONG_TOOL, StageTarget.S1.value if task.stage_target == StageTarget.S1.value else StageTarget.S2.value
        return FailureTag.WRONG_TOOL, StageTarget.S2.value
    if task.family == BenchmarkFamily.OBSERVATION.value:
        if format_valid == 0:
            return FailureTag.INVALID_FORMAT, StageTarget.S4.value
        return FailureTag.EXECUTION_FAIL, StageTarget.S4.value
    if task.family == BenchmarkFamily.EXECUTION_ENVIRONMENT.value:
        if execution_success == 0 or "runtime" in notes.lower():
            return FailureTag.EXECUTION_FAIL, StageTarget.S4.value
        if format_valid == 0:
            return FailureTag.INVALID_FORMAT, StageTarget.S4.value
        return FailureTag.NO_CALL, StageTarget.S4.value
    if task.family == BenchmarkFamily.CROSS_FAMILY.value:
        families = set(task.active_families())
        if BenchmarkFamily.INTERFACE_EXPOSURE.value in families and observed_failure_tag in {FailureTag.MISSING_PARAMETER, FailureTag.WRONG_PARAMETER}:
            stage = StageTarget.S1.value if StageTarget.S1.value in task.stage_targets_applied else StageTarget.S2.value
            return FailureTag.WRONG_TOOL, stage
        if BenchmarkFamily.USER_INTENT.value in families:
            if call_present == 0:
                return FailureTag.NO_CALL, StageTarget.S1.value if StageTarget.S1.value in task.stage_targets_applied else StageTarget.S2.value
            if canonical_tool_match == 0 or interface_name_match == 0:
                return FailureTag.WRONG_TOOL, StageTarget.S1.value if StageTarget.S1.value in task.stage_targets_applied else StageTarget.S2.value
        if BenchmarkFamily.EXECUTION_ENVIRONMENT.value in families:
            if execution_success == 0 or "runtime" in notes.lower():
                return FailureTag.EXECUTION_FAIL, StageTarget.S4.value
            if format_valid == 0:
                return FailureTag.INVALID_FORMAT, StageTarget.S4.value
        if BenchmarkFamily.OBSERVATION.value in families:
            if format_valid == 0:
                return FailureTag.INVALID_FORMAT, StageTarget.S4.value
            return FailureTag.EXECUTION_FAIL, StageTarget.S4.value
    if observed_failure_tag == FailureTag.MISSING_PARAMETER:
        return FailureTag.MISSING_PARAMETER, StageTarget.S3.value
    if observed_failure_tag == FailureTag.WRONG_PARAMETER:
        return FailureTag.WRONG_PARAMETER, StageTarget.S3.value
    return observed_failure_tag, task.stage_target


def _family_profile_for_task(task: TaskSample):
    if task.family != BenchmarkFamily.CROSS_FAMILY.value:
        return FAMILY_FAILURE_PROFILES.get(task.family)
    profiles = [FAMILY_FAILURE_PROFILES[family] for family in task.active_families() if family in FAMILY_FAILURE_PROFILES]
    if not profiles:
        return None
    expected = tuple(dict.fromkeys(item for profile in profiles for item in profile.expected_primary_failures))
    spillover = tuple(dict.fromkeys(item for profile in profiles for item in profile.allowed_spillover_failures))
    disallowed = tuple(dict.fromkeys(item for profile in profiles for item in profile.disallowed_direct_perturbations))
    return type(profiles[0])(
        expected_primary_failures=expected,
        allowed_spillover_failures=spillover,
        disallowed_direct_perturbations=disallowed,
        explanation="Cross-family composition uses the union of component family envelopes.",
    )
