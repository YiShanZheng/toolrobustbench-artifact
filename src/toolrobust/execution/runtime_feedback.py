"""Runtime feedback injection for S4 observation/execution families."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from toolrobust.domain.enums import BenchmarkFamily
from toolrobust.domain.schemas import TaskSample
from toolrobust.tools.base import ToolExecutionError
from toolrobust.tools.executor import ToolExecutor

_ARG_PRIORITY = {
    "get_weather": ("place", "slot"),
    "get_air_quality": ("place", "slot"),
    "get_temperature": ("place", "slot", "scale"),
    "convert_unit": ("value", "source", "target"),
    "convert_currency": ("value", "source", "target"),
    "search_document": ("query", "scope"),
    "search_policy": ("query", "scope"),
    "search_faq": ("query", "scope"),
    "calculate_route": ("origin", "target", "profile"),
    "estimate_travel_time": ("origin", "target", "profile"),
    "sort_items": ("items", "rule"),
    "filter_items": ("items", "rule"),
}

_TOGGLE_VALUES = {
    "today": "tomorrow",
    "tomorrow": "today",
    "c": "f",
    "f": "c",
    "usd": "eur",
    "eur": "usd",
    "cny": "usd",
    "gbp": "usd",
    "km": "m",
    "m": "km",
    "asc": "desc",
    "desc": "asc",
    "metro": "walk",
    "walk": "drive",
    "drive": "metro",
    "finance": "hr",
    "hr": "it",
    "it": "finance",
    "engineering": "travel",
    "travel": "finance",
    "cloud_app": "router_hub",
    "router_hub": "payroll_portal",
    "payroll_portal": "cloud_app",
    "hotel": "station",
    "station": "home",
    "home": "hotel",
    "expo": "museum",
    "museum": "office",
    "office": "expo",
}

_RENAMED_KEY_ALIASES = {
    "condition": "weather_reading",
    "temperature_c": "temperature_reading",
    "aqi": "air_reading",
    "category": "air_band",
    "temperature": "temperature_value",
    "scale": "temperature_scale",
    "converted_value": "result_value",
    "target": "target_label",
    "source": "source_label",
    "value": "input_value",
    "matches": "result_items",
    "query": "lookup_query",
    "scope": "lookup_scope",
    "distance_km": "route_distance",
    "duration_min": "eta_minutes",
    "profile": "travel_mode",
    "sorted_items": "result_items",
    "filtered_items": "result_items",
    "rule": "applied_rule",
    "place": "location_name",
    "slot": "time_slot",
}


@dataclass(slots=True)
class RuntimeFeedback:
    execution_success: int
    status: str
    content: Any
    note: str
    clean_result: Any | None = None

    def as_internal_dict(self) -> dict[str, Any]:
        return {
            "execution_success": self.execution_success,
            "status": self.status,
            "content": self.content,
            "note": self.note,
            "clean_result": self.clean_result,
        }

    def as_public_dict(self) -> dict[str, Any]:
        payload = {
            "execution_success": self.execution_success,
            "status": self.status,
            "content": _deep_copy(self.content),
            "note": self.note,
        }
        _assert_no_forbidden_runtime_keys(payload)
        return payload

    def as_dict(self) -> dict[str, Any]:
        return self.as_internal_dict()


def run_perturbed_execution(
    task: TaskSample,
    predicted_tool_resolved: str | None,
    normalized_args: dict[str, Any],
    *,
    executor: ToolExecutor | None = None,
) -> RuntimeFeedback:
    runner = executor or ToolExecutor()
    if not predicted_tool_resolved:
        return RuntimeFeedback(
            execution_success=0,
            status="error",
            content={"error_type": "unresolved_tool", "message": "tool name could not be resolved"},
            note="runtime skipped because tool name was unresolved",
        )

    try:
        clean_result = runner.execute(predicted_tool_resolved, normalized_args)
        clean_error = None
    except (ToolExecutionError, KeyError, ValueError, TypeError) as exc:
        clean_result = None
        clean_error = str(exc)

    if task.has_family(BenchmarkFamily.EXECUTION_ENVIRONMENT.value):
        return _execution_feedback(task, predicted_tool_resolved, normalized_args, clean_result, clean_error)
    if task.has_family(BenchmarkFamily.OBSERVATION.value):
        return _observation_feedback(task, predicted_tool_resolved, normalized_args, clean_result, clean_error)
    if clean_error is not None:
        return RuntimeFeedback(
            execution_success=0,
            status="error",
            content={"error_type": "tool_execution_error", "message": clean_error},
            note=f"clean runtime error: {clean_error}",
        )
    return RuntimeFeedback(
        execution_success=1,
        status="success",
        content=clean_result,
        note="clean runtime result",
        clean_result=clean_result,
    )


def build_observation_target(tool_id: str | None, clean_result: Any | None) -> Any:
    if tool_id is None or clean_result is None or not isinstance(clean_result, dict):
        return _deep_copy(clean_result)
    # Observation recovery should depend on the salient evidence-bearing view of
    # the runtime payload, not just the narrow anchor subset. Otherwise evidence
    # loss / competition operators can become score-neutral because the dropped
    # fields never participate in the target.
    target_keys = _salient_output_keys(tool_id)
    if not target_keys:
        return _deep_copy(clean_result)
    target = {key: _deep_copy(clean_result[key]) for key in target_keys if key in clean_result}
    return target or _deep_copy(clean_result)


def _observation_feedback(
    task: TaskSample,
    tool_id: str,
    normalized_args: dict[str, Any],
    clean_result: Any | None,
    clean_error: str | None,
) -> RuntimeFeedback:
    if clean_error is not None:
        return RuntimeFeedback(
            execution_success=0,
            status="error",
            content={"error_type": "tool_execution_error", "message": clean_error},
            note=f"observation stage received a runtime error before perturbation: {clean_error}",
        )

    assert clean_result is not None
    content = clean_result
    notes: list[str] = []
    for operator_name in task.operators_applied:
        content, note = _apply_observation_operator(tool_id, normalized_args, content, operator_name)
        notes.append(note)
    return RuntimeFeedback(
        execution_success=1,
        status="success",
        content=content,
        note="; ".join(notes) if notes else "clean observation payload",
        clean_result=clean_result,
    )


def _execution_feedback(
    task: TaskSample,
    tool_id: str,
    normalized_args: dict[str, Any],
    clean_result: Any | None,
    clean_error: str | None,
) -> RuntimeFeedback:
    if clean_error is not None:
        return RuntimeFeedback(
            execution_success=0,
            status="error",
            content={"error_type": "tool_execution_error", "message": clean_error},
            note=f"execution stage received a clean runtime error before perturbation: {clean_error}",
            clean_result=clean_result,
        )

    assert clean_result is not None
    subtype = task.component_subtype(BenchmarkFamily.EXECUTION_ENVIRONMENT.value) or task.subtype
    if subtype == "D1 Availability Failure":
        return _availability_feedback(task, tool_id, clean_result)
    if subtype == "D2 Runtime Variability":
        return _runtime_variability_feedback(task, tool_id, normalized_args, clean_result)
    if subtype == "D3 Failure Feedback Quality":
        return _failure_feedback_quality(task, tool_id, normalized_args, clean_result)
    return RuntimeFeedback(
        execution_success=1,
        status="success",
        content=_deep_copy(clean_result),
        note="clean execution feedback",
        clean_result=clean_result,
    )


def _apply_observation_operator(tool_id: str, args: dict[str, Any], content: Any, operator_name: str) -> tuple[Any, str]:
    payload = _deep_copy(content)
    if operator_name == "key_rename_soft":
        return _rename_salient_keys(tool_id, payload, aggressive=False), "added an alternate alias for a single salient observation key"
    if operator_name == "key_rename":
        return _rename_salient_keys(tool_id, payload, aggressive=True), "renamed the primary observation keys"
    if operator_name == "camel_snake_shift":
        return _camelize_keys(payload), "shifted observation keys to camelCase"
    if operator_name == "field_reorder":
        return _reorder_fields(payload), "reordered observation fields"
    if operator_name == "nesting_shift":
        return _nest_observation_payload(tool_id, payload), "moved the observation payload into nested frames"
    if operator_name == "wrapper_insertion":
        return _wrap_observation_payload(tool_id, payload), "wrapped the observation inside multi-block runtime packets"
    if operator_name == "missing_field":
        return _drop_salient_field(tool_id, payload), "dropped a salient observation field"
    if operator_name == "partial_truncation":
        return _truncate_payload(tool_id, payload), "truncated part of the observation evidence"
    if operator_name == "dropped_metadata":
        return _drop_metadata(payload), "removed contextual metadata from the observation"
    if operator_name == "sparse_result":
        return _sparsify_payload(tool_id, payload), "reduced the observation to a sparse but still recoverable subset"
    if operator_name == "stale_value":
        return _corrupt_supporting_evidence(tool_id, args, payload, all_supporting=False), "added one stale-looking competing variant for a supporting observation cue"
    if operator_name == "supporting_conflict":
        return _corrupt_supporting_evidence(tool_id, args, payload, all_supporting=True), "introduced conflicting stale-looking variants for supporting observation cues"
    if operator_name == "wrong_fact_field":
        return _corrupt_primary_evidence(tool_id, args, payload), "corrupted the main fact-bearing observation field"
    if operator_name == "unit_mismatch":
        return _inject_unit_mismatch(tool_id, args, payload), "introduced a unit or label mismatch"
    if operator_name == "inconsistent_pair":
        return _inject_inconsistent_pair(tool_id, args, payload), "introduced a conflicting evidence pair"
    if operator_name == "misleading_note":
        return _attach_misleading_note(tool_id, args, payload), "attached a misleading note favoring the wrong candidate"
    if operator_name == "distractor_candidate_soft":
        return _inject_candidate_bundle(tool_id, args, payload, prefer_wrong_candidate=False), "turned the observation into a softly competing candidate bundle"
    if operator_name == "distractor_candidate":
        return _inject_candidate_bundle(tool_id, args, payload, prefer_wrong_candidate=True), "turned the observation into a competing-candidate bundle"
    if operator_name == "low_utility_redundancy_soft":
        return _add_low_utility_redundancy(payload, biased=False), "added mild redundant bookkeeping around the observation"
    if operator_name == "low_utility_redundancy":
        return _add_low_utility_redundancy(payload, biased=True), "added redundant low-utility evidence around the observation"
    if operator_name == "spurious_auxiliary_field":
        wrapped = _ensure_object(payload)
        wrapped["auxiliary_candidate"] = _alternate_candidate(tool_id, args)
        return wrapped, "added a spurious auxiliary candidate field"
    return payload, f"left observation unchanged for unknown operator `{operator_name}`"


def _availability_feedback(task: TaskSample, tool_id: str, clean_result: Any) -> RuntimeFeedback:
    operators = set(task.operators_applied)
    if "cached_availability_fallback" in operators:
        return RuntimeFeedback(
            execution_success=1,
            status="success",
            content={
                "result": _deep_copy(clean_result),
                "delivery_meta": {
                    "delivery_mode": "complete",
                    "fallback_path": "transparent_retry",
                    "payload_state": "complete",
                    "result_integrity": "complete_payload",
                },
            },
            note="returned a complete payload through a transparent availability fallback",
            clean_result=clean_result,
        )
    if "service_unavailable_with_stale_cache" in operators and "timeout_after_partial_stream" in operators:
        recoverable = _medium_partial_payload(tool_id, _deep_copy(clean_result))
        partial_hint = _lossy_partial_hint(tool_id, _deep_copy(clean_result), strength="heavy")
        stale_replay = _corrupt_primary_evidence(tool_id, {}, _deep_copy(clean_result))
        return RuntimeFeedback(
            execution_success=0,
            status="error",
            content={
                "error_type": "timeout_after_service_unavailable",
                "status_code": 503,
                "message": "primary upstream unavailable and stream stalled after partial frames",
                "retry_after_ms": 2200,
                "cache_snapshot": _deep_copy(clean_result),
                "cache_state": "stale_fallback",
                "recoverable_fragment": recoverable,
                "raw": {
                    "fragment_hint": partial_hint,
                    "completion_state": "stream_interrupted",
                },
                "stream_frames": [
                    {"frame_id": "frame_1", "state": "partial", "payload": recoverable},
                    {"frame_id": "frame_2", "state": "stale_replay", "payload": stale_replay},
                    {"frame_id": "frame_3", "state": "missing"},
                ],
            },
            note="reported service unavailability with stale cache plus a timed-out partial stream",
            clean_result=clean_result,
        )
    if "service_unavailable_with_stale_cache" in operators:
        partial = _medium_partial_payload(tool_id, _deep_copy(clean_result))
        return RuntimeFeedback(
            execution_success=0,
            status="error",
            content={
                "error_type": "service_unavailable",
                "status_code": 503,
                "message": "primary upstream unavailable",
                "retry_after_ms": 2200,
                "fragment": partial,
                "cache_snapshot": _deep_copy(clean_result),
                "cache_state": "stale_fallback",
            },
            note="reported service unavailability with a stale cache and a usable partial fragment",
            clean_result=clean_result,
        )
    if "timeout_after_partial_stream" in operators:
        recoverable = _medium_partial_payload(tool_id, _deep_copy(clean_result))
        partial_hint = _lossy_partial_hint(tool_id, _deep_copy(clean_result), strength="heavy")
        stale_replay = _corrupt_primary_evidence(tool_id, {}, _deep_copy(clean_result))
        return RuntimeFeedback(
            execution_success=0,
            status="error",
            content={
                "error_type": "timeout",
                "elapsed_ms": 60000,
                "message": "stream stalled after partial frames",
                "recoverable_fragment": recoverable,
                "raw": {
                    "fragment_hint": partial_hint,
                    "completion_state": "stream_interrupted",
                },
                "stream_frames": [
                    {"frame_id": "frame_1", "state": "partial", "payload": recoverable},
                    {"frame_id": "frame_2", "state": "stale_replay", "payload": stale_replay},
                    {"frame_id": "frame_3", "state": "missing"},
                ],
            },
            note="simulated a timeout after partial frames with noisy but recoverable evidence",
            clean_result=clean_result,
        )
    if "timeout" in operators:
        return RuntimeFeedback(
            execution_success=0,
            status="error",
            content={
                "elapsed_ms": 60000,
                "message": "connection closed before the final frame arrived",
            },
            note="simulated a timeout after the request had already started streaming",
            clean_result=clean_result,
        )
    if "service_unavailable" in operators:
        return RuntimeFeedback(
            execution_success=0,
            status="error",
            content={
                "status_code": 529,
                "message": "upstream overloaded",
                "retry_after_ms": 1800,
            },
            note="simulated an upstream availability failure",
            clean_result=clean_result,
        )
    if "empty_response" in operators:
        return RuntimeFeedback(
            execution_success=0,
            status="error",
            content={
                "response_meta": {"body_bytes": 0, "headers_only": True},
                "stream_state": "closed",
            },
            note="returned a near-empty runtime payload",
            clean_result=clean_result,
        )
    if "auth_denied" in operators:
        return RuntimeFeedback(
            execution_success=0,
            status="error",
            content={
                "status_code": 401,
                "message": "session challenge required",
                "challenge": "reauth",
            },
            note="simulated an authorization failure",
            clean_result=clean_result,
        )
    return RuntimeFeedback(
        execution_success=1,
        status="success",
        content=_deep_copy(clean_result),
        note="clean execution feedback",
        clean_result=clean_result,
    )


def _runtime_variability_feedback(task: TaskSample, tool_id: str, normalized_args: dict[str, Any], clean_result: Any) -> RuntimeFeedback:
    operators = set(task.operators_applied)
    notes: list[str] = []

    if "jittered_metadata" in operators:
        wrapped = {
            "result": _deep_copy(clean_result),
            "delivery_meta": {
                "delivery_mode": "complete",
                "trace_state": "complete_payload_with_header_jitter",
                "jitter_scope": "delivery_headers_only",
                "payload_state": "complete",
                "result_integrity": "complete_payload",
            },
        }
        notes.append("wrapped a complete payload with harmless delivery header jitter")
        return RuntimeFeedback(
            execution_success=1,
            status="success",
            content=wrapped,
            note="; ".join(notes),
            clean_result=clean_result,
        )

    if "delayed_fresh_partial_with_stale_full" in operators and "contaminated_flaky_attempts" in operators:
        partial = _heavy_partial_payload(tool_id, _deep_copy(clean_result))
        stale_full = _corrupt_primary_evidence(tool_id, normalized_args, _deep_copy(clean_result))
        decoy_partial = _corrupt_primary_evidence(tool_id, normalized_args, _deep_copy(partial))
        notes.append("returned contaminated rotating attempts with a buried fresh fragment")
        return RuntimeFeedback(
            execution_success=0,
            status="error",
            content={
                "error_type": "attempt_resolution_conflict",
                "attempts": [
                    {
                        "attempt_id": "t1",
                        "envelope": "canonical",
                        "capture_order": 1,
                        "result": stale_full,
                        "state": "complete_replay",
                    },
                    {
                        "attempt_id": "t2",
                        "envelope": "preview",
                        "capture_order": 2,
                        "fragment": decoy_partial,
                        "state": "candidate_preview",
                    },
                    {
                        "attempt_id": "t3",
                        "envelope": "compat",
                        "capture_order": 3,
                        "compat_envelope": {
                            "data": partial,
                            "completion_state": "tail_missing",
                        },
                        "state": "unverified_tail",
                    },
                ],
                "raw": {
                    "envelope": {
                        "data": partial,
                        "completion_state": "tail_missing",
                    },
                    "resume_token": "rotating-attempt",
                    "transport_retryable": True,
                },
                "attempt_consistency": "conflicting_with_contaminated_replay",
                "schema_mode": "rotating_attempts",
                "selection_bias": {
                    "surface_preference": "complete_replay",
                    "candidate_order": ["t1", "t2", "t3"],
                },
            },
            note="; ".join(notes),
            clean_result=clean_result,
        )

    if "delayed_fresh_partial_with_stale_full" in operators and "flaky_shape_conflicting_attempts" in operators:
        partial = _heavy_partial_payload(tool_id, _deep_copy(clean_result))
        stale_full = _deep_copy(clean_result)
        decoy_partial = _corrupt_primary_evidence(tool_id, normalized_args, _deep_copy(partial))
        notes.append("returned conflicting runtime attempts inside changing envelopes")
        return RuntimeFeedback(
            execution_success=0,
            status="error",
            content={
                "error_type": "runtime_attempt_conflict",
                "attempts": [
                    {
                        "attempt_id": "t1",
                        "envelope": "canonical",
                        "capture_order": 1,
                        "result": stale_full,
                        "state": "complete_replay",
                    },
                    {
                        "attempt_id": "t2",
                        "envelope": "compat",
                        "capture_order": 2,
                        "compat_envelope": {"data": partial},
                        "state": "tail_missing",
                    },
                    {
                        "attempt_id": "t3",
                        "envelope": "preview",
                        "capture_order": 3,
                        "fragment": decoy_partial,
                        "confidence": "low",
                    },
                ],
                "raw": {
                    "envelope": {
                        "data": partial,
                        "completion_state": "tail_missing",
                    },
                    "resume_token": "rotating-attempt",
                    "transport_retryable": True,
                },
                "attempt_consistency": "conflicting",
                "schema_mode": "rotating_attempts",
            },
            note="; ".join(notes),
            clean_result=clean_result,
        )

    if "delayed_fresh_partial_with_stale_full" in operators:
        partial = _medium_partial_payload(tool_id, _deep_copy(clean_result))
        stale_full = _deep_copy(clean_result)
        notes.append("returned a stale complete candidate beside a fresher partial fragment")
        return RuntimeFeedback(
            execution_success=0,
            status="error",
            content={
                "error_type": "partial_response",
                "attempts": [
                    {"attempt_id": "cached", "freshness": "stale", "result": stale_full},
                    {"attempt_id": "live", "freshness": "fresh", "fragment": partial, "completion_state": "truncated"},
                ],
                "fragment": partial,
                "completion_state": "truncated",
                "resume_token": "tail-missing",
            },
            note="; ".join(notes),
            clean_result=clean_result,
        )

    if "unstable_optional_field" in operators:
        wrapped = _ensure_object(_deep_copy(clean_result))
        wrapped["_runtime_meta"] = {
            "delivery_mode": "complete",
            "sequence_id": 2,
            "trace_state": "volatile",
        }
        notes.append("added non-essential runtime metadata around an otherwise complete payload")
        return RuntimeFeedback(
            execution_success=1,
            status="success",
            content=wrapped,
            note="; ".join(notes),
            clean_result=clean_result,
        )

    if "nondeterministic_order" in operators:
        notes.append("shuffled runtime field order without removing fields")
        return RuntimeFeedback(
            execution_success=1,
            status="success",
            content=_reorder_fields(_deep_copy(clean_result)),
            note="; ".join(notes),
            clean_result=clean_result,
        )

    medium_fragment = _medium_partial_payload(tool_id, _deep_copy(clean_result))
    if "delayed_partial_response" in operators and "flaky_response_shape" in operators:
        heavy_fragment = _heavy_partial_payload(tool_id, _deep_copy(clean_result))
        notes.append("returned a truncated fragment inside a compatibility envelope")
        return RuntimeFeedback(
            execution_success=0,
            status="error",
            content={
                "compat_envelope": {
                    "data": heavy_fragment,
                    "completion_state": "truncated",
                    "resume_token": "tail-missing",
                },
                "schema_mode": "compat",
                "shape_variant": "unstable",
            },
            note="; ".join(notes),
            clean_result=clean_result,
        )

    if "delayed_partial_response" in operators:
        notes.append("returned an incomplete but directly usable fragment")
        return RuntimeFeedback(
            execution_success=0,
            status="error",
            content={
                "fragment": medium_fragment,
                "completion_state": "truncated",
                "resume_token": "tail-missing",
            },
            note="; ".join(notes),
            clean_result=clean_result,
        )

    if "flaky_response_shape" in operators:
        notes.append("returned a compatibility wrapper around the clean payload")
        return RuntimeFeedback(
            execution_success=1,
            status="success",
            content={"compat_envelope": {"data": _deep_copy(clean_result)}, "schema_mode": "compat"},
            note="; ".join(notes),
            clean_result=clean_result,
        )

    return RuntimeFeedback(
        execution_success=1,
        status="success",
        content=_deep_copy(clean_result),
        note="clean execution feedback",
        clean_result=clean_result,
    )


def _failure_feedback_quality(
    task: TaskSample,
    tool_id: str,
    normalized_args: dict[str, Any],
    clean_result: Any,
) -> RuntimeFeedback:
    operators = set(task.operators_applied)
    notes: list[str] = []

    if "hidden_partial_wrong_label" in operators and "conflicting_diagnostic_decoy" in operators:
        hidden_hint = _lossy_partial_hint(tool_id, _deep_copy(clean_result), strength="heavy")
        decoy_hint = _lossy_partial_hint(
            tool_id,
            _corrupt_primary_evidence(tool_id, normalized_args, _heavy_partial_payload(tool_id, _deep_copy(clean_result))),
            strength="heavy",
        )
        stale_hint = _lossy_partial_hint(
            tool_id,
            _corrupt_supporting_evidence(tool_id, normalized_args, _deep_copy(clean_result), all_supporting=True),
            strength="heavy",
        )
        notes.append("mixed lossy partial-looking hints with competing diagnostics and no recoverable fragment")
        return RuntimeFeedback(
            execution_success=0,
            status="error",
            content={
                "status_code": 502,
                "message": "response envelope alternated between partial-looking previews and conflicting diagnostics",
                "reported_issue": "schema_parse_failure",
                "raw": {
                    "frames": [
                        {"frame_id": "f1", "reported_issue": "missing_input", "diagnostic": {"missing_keys": ["payload"]}},
                        {"frame_id": "f2", "reported_issue": "schema_parse_failure", "fragment_hint": decoy_hint, "candidate_role": "surface_preview"},
                        {"frame_id": "f3", "reported_issue": "transport_tail_missing", "envelope": {"fragment_hint": hidden_hint, "tail_state": "missing"}},
                        {"frame_id": "f4", "reported_issue": "stale_candidate_replay", "fragment_hint": stale_hint, "candidate_role": "stale_preview"},
                    ],
                    "envelope": {
                        "fragment_hint": hidden_hint,
                        "completion_state": "tail_missing",
                        "payload_integrity": "conflicting_preview_only",
                    },
                    "audit_tail": {
                        "preview_bundle": [hidden_hint, decoy_hint, stale_hint],
                        "selection_hint": "newest_preview_may_be_incomplete",
                        "missing_tail_bytes": True,
                    },
                    "transport_retryable": True,
                },
                "diagnostic_panel": {
                    "label_confidence": 0.19,
                    "observed_issue": "competing_diagnostics_after_partial_payload",
                    "preferred_surface_label": "schema_parse_failure",
                    "decoy_frame": "f2",
                    "surface_bias": "partial_preview_is_visually_salient_but_not_recoverable",
                },
            },
            note="; ".join(notes),
            clean_result=clean_result,
        )

    if "hidden_partial_wrong_label" in operators:
        hidden_partial = _medium_partial_payload(tool_id, _deep_copy(clean_result))
        decoy_partial = _corrupt_primary_evidence(tool_id, normalized_args, _deep_copy(hidden_partial))
        notes.append("hid a usable partial fragment under an actively wrong failure label")
        return RuntimeFeedback(
            execution_success=0,
            status="error",
            content={
                "status_code": 502,
                "message": "response envelope malformed after partial data",
                "reported_issue": "schema_parse_failure",
                "raw": {
                    "recoverable_fragment": hidden_partial,
                    "shadow_fragment": decoy_partial,
                    "recoverable_fragment_keys": sorted(hidden_partial) if isinstance(hidden_partial, dict) else [],
                    "completion_state": "truncated",
                    "transport_retryable": True,
                },
                "diagnostic_panel": {
                    "label_confidence": 0.34,
                    "observed_issue": "label_conflicts_with_payload_evidence",
                    "contradiction": "payload fragment was produced before the schema label was attached",
                    "usable_fragment_path": "raw.recoverable_fragment",
                },
            },
            note="; ".join(notes),
            clean_result=clean_result,
        )

    if "wrong_failure_label_with_evidence" in operators:
        notes.append("reported a validation-style label that conflicts with runtime evidence")
        return RuntimeFeedback(
            execution_success=0,
            status="error",
            content={
                "status_code": 400,
                "message": "runtime wrapper reported required field missing, but invocation audit received all required keys",
                "reported_issue": "missing_input",
                "echo_keys": sorted(normalized_args),
                "diagnostic_panel": {
                    "received_invocation": "complete",
                    "missing_keys": [],
                    "observed_issue": "label_conflicts_with_invocation_audit",
                    "label_confidence": 0.38,
                    "upstream_status_code": 502,
                    "transport_retryable": True,
                },
            },
            note="; ".join(notes),
            clean_result=clean_result,
        )

    if "vague_retry_hint" in operators:
        notes.append("returned a vague runtime error with a weak retryability hint")
        return RuntimeFeedback(
            execution_success=0,
            status="error",
            content={
                "message": "request could not be completed",
                "detail": "try the same call again later",
                "reported_issue": "temporary_dependency_unclear",
                "retryable_hint": True,
                "diagnostic_code": "temporary_dependency_unclear",
            },
            note="; ".join(notes),
            clean_result=clean_result,
        )

    if "partial_success_no_status" in operators and "wrong_failure_label" in operators:
        hidden_partial = _heavy_partial_payload(tool_id, _deep_copy(clean_result))
        notes.append("reported the wrong runtime category while hiding a usable partial fragment")
        return RuntimeFeedback(
            execution_success=0,
            status="error",
            content={
                "status_code": 502,
                "message": "response envelope malformed",
                "raw": {
                    "fragment": hidden_partial,
                    "completion_state": "truncated",
                },
            },
            note="; ".join(notes),
            clean_result=clean_result,
        )

    if "wrong_failure_label" in operators:
        notes.append("reported an incorrect runtime category without exposing a usable result")
        return RuntimeFeedback(
            execution_success=0,
            status="error",
            content={
                "status_code": 502,
                "message": "response envelope malformed",
                "reported_issue": "schema_parse_failure",
            },
            note="; ".join(notes),
            clean_result=clean_result,
        )

    if "vague_error" in operators:
        notes.append("returned a vague retryable runtime error")
        return RuntimeFeedback(
            execution_success=0,
            status="error",
            content={
                "message": "request could not be completed",
                "detail": "try the same call again later",
            },
            note="; ".join(notes),
            clean_result=clean_result,
        )

    if "misleading_error" in operators:
        notes.append("returned a misleading validation-style error despite receiving a full invocation")
        return RuntimeFeedback(
            execution_success=0,
            status="error",
            content={
                "status_code": 400,
                "message": "required field missing",
                "reported_issue": "missing_input",
                "echo_keys": sorted(normalized_args),
            },
            note="; ".join(notes),
            clean_result=clean_result,
        )

    if "partial_success_no_status" in operators:
        notes.append("returned a usable fragment without any explicit success state")
        return RuntimeFeedback(
            execution_success=0,
            status="error",
            content={
                "fragment": _medium_partial_payload(tool_id, _deep_copy(clean_result)),
                "completion_state": "truncated",
            },
            note="; ".join(notes),
            clean_result=clean_result,
        )

    return RuntimeFeedback(
        execution_success=1,
        status="success",
        content=_deep_copy(clean_result),
        note="clean execution feedback",
        clean_result=clean_result,
    )


def _rename_salient_keys(tool_id: str, payload: Any, *, aggressive: bool) -> Any:
    if not aggressive:
        renamed_payload, _ = _rename_single_salient_key(tool_id, payload)
        return renamed_payload
    if isinstance(payload, dict):
        renamed = OrderedDict()
        for key, value in payload.items():
            alias = _RENAMED_KEY_ALIASES.get(key)
            renamed[alias or f"observed_{key}"] = _rename_salient_keys(tool_id, value, aggressive=True)
            if alias is not None:
                renamed[f"{key}_shadow"] = _structure_shadow(value)
        return dict(renamed)
    if isinstance(payload, list):
        return [_rename_salient_keys(tool_id, item, aggressive=True) for item in payload]
    return payload


def _rename_single_salient_key(tool_id: str, payload: Any) -> tuple[Any, bool]:
    salient_keys = set(_salient_output_keys(tool_id))
    if isinstance(payload, dict):
        renamed = OrderedDict()
        renamed_any = False
        for key, value in payload.items():
            if not renamed_any and key in salient_keys:
                alias = _RENAMED_KEY_ALIASES.get(key)
                renamed[alias or f"observed_{key}"] = _deep_copy(value)
                renamed[key] = _deep_copy(value)
                renamed_any = True
                continue
            nested_value, nested_renamed = _rename_single_salient_key(tool_id, value)
            renamed[key] = nested_value
            renamed_any = renamed_any or nested_renamed
        return dict(renamed), renamed_any
    if isinstance(payload, list):
        renamed_items: list[Any] = []
        renamed_any = False
        for item in payload:
            nested_item, nested_renamed = _rename_single_salient_key(tool_id, item)
            renamed_items.append(nested_item)
            renamed_any = renamed_any or nested_renamed
        return renamed_items, renamed_any
    return payload, False


def _camelize_keys(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {_camelize(key): _camelize_keys(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_camelize_keys(item) for item in payload]
    return payload


def _camelize(value: str) -> str:
    pieces = value.split("_")
    if len(pieces) == 1:
        return value
    return pieces[0] + "".join(piece.capitalize() for piece in pieces[1:])


def _reorder_fields(payload: Any) -> Any:
    if isinstance(payload, dict):
        ordered = OrderedDict()
        for key in reversed(list(payload.keys())):
            ordered[key] = _reorder_fields(payload[key])
        return dict(ordered)
    if isinstance(payload, list):
        return list(reversed([_reorder_fields(item) for item in payload]))
    return payload


def _drop_salient_field(tool_id: str, payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    mutated = dict(payload)
    anchor_keys = [key for key in _evidence_anchor_keys(tool_id) if key in mutated]
    if anchor_keys:
        key = anchor_keys[-1]
        mutated.pop(key, None)
        mutated[_hint_field_name(key)] = _lossy_hint_value(payload[key], strength="medium")
        mutated["evidence_status"] = "anchor_partially_missing"
        return mutated
    context_keys = [key for key in _evidence_context_keys(tool_id) if key in mutated]
    if context_keys:
        key = context_keys[-1]
        mutated.pop(key, None)
        mutated[_hint_field_name(key)] = _lossy_hint_value(payload[key], strength="medium")
        mutated["evidence_status"] = "context_missing"
        return mutated
    if mutated:
        key = next(iter(mutated))
        mutated.pop(key, None)
        mutated[_hint_field_name(key)] = _lossy_hint_value(payload[key], strength="medium")
    return mutated


def _truncate_payload(tool_id: str, payload: Any) -> Any:
    if isinstance(payload, dict):
        mutated = dict(payload)
        anchor_keys = [key for key in _evidence_anchor_keys(tool_id) if key in mutated]
        context_keys = [key for key in _evidence_context_keys(tool_id) if key in mutated]
        hint_keys = [key for key in list(mutated.keys()) if key.endswith("_hint")]
        if hint_keys or anchor_keys or context_keys:
            for key in hint_keys:
                mutated[key] = _coarsen_lossy_hint(mutated[key])
            for key in anchor_keys + context_keys:
                mutated.pop(key, None)
                mutated[_hint_field_name(key)] = _lossy_hint_value(payload[key], strength="heavy")
            mutated["evidence_status"] = "anchor_hint_only"
            return mutated
        if len(mutated) > 1:
            mutated.pop(list(mutated.keys())[-1], None)
        return mutated
    if isinstance(payload, list):
        return list(payload)[:1]
    if isinstance(payload, str):
        return payload[:8]
    return payload


def _stable_partial_payload(tool_id: str, payload: Any) -> Any:
    if isinstance(payload, dict):
        partial: dict[str, Any] = {}
        for key in _evidence_anchor_keys(tool_id):
            if key in payload:
                partial[key] = _deep_copy(payload[key])
        if not partial:
            for key in _salient_output_keys(tool_id):
                if key in payload:
                    partial[key] = _deep_copy(payload[key])
                    break
        for key in _evidence_context_keys(tool_id):
            if key in payload and key not in partial:
                partial[key] = _deep_copy(payload[key])
                break
        if partial:
            return partial
        if payload:
            first_key = next(iter(payload))
            return {first_key: _deep_copy(payload[first_key])}
    if isinstance(payload, list):
        return list(payload)[:1]
    if isinstance(payload, str):
        return payload[:8]
    return payload


def _medium_partial_payload(tool_id: str, payload: Any) -> Any:
    return _stable_partial_payload(tool_id, payload)


def _heavy_partial_payload(tool_id: str, payload: Any) -> Any:
    if isinstance(payload, dict):
        partial: dict[str, Any] = {}
        for key in _evidence_anchor_keys(tool_id):
            if key in payload:
                partial[key] = _deep_copy(payload[key])
        if partial:
            return partial
    return _stable_partial_payload(tool_id, payload)


def _lossy_partial_hint(tool_id: str, payload: Any, *, strength: str) -> Any:
    if not isinstance(payload, dict):
        return _lossy_hint_value(payload, strength=strength)
    hint: dict[str, Any] = {}
    for key in _evidence_anchor_keys(tool_id):
        if key in payload:
            hint[_hint_field_name(key)] = _lossy_hint_value(payload[key], strength=strength)
    if hint:
        hint["evidence_status"] = "lossy_hint_only"
        return hint
    return _lossy_hint_value(_stable_partial_payload(tool_id, payload), strength=strength)


def _drop_metadata(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    mutated = dict(payload)
    for key in ("place", "slot", "scope", "query", "origin", "target", "profile", "source", "value"):
        if key in mutated:
            mutated.pop(key, None)
            break
    return mutated


def _sparsify_payload(tool_id: str, payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    sparse: dict[str, Any] = {}
    anchor_keys = [key for key in _evidence_anchor_keys(tool_id) if key in payload]
    context_keys = [key for key in _evidence_context_keys(tool_id) if key in payload]
    passthrough_keys = [key for key in payload if key not in set(anchor_keys) | set(context_keys)]

    if anchor_keys:
        first_anchor = anchor_keys[0]
        sparse[first_anchor] = _deep_copy(payload[first_anchor])
        for key in anchor_keys[1:]:
            sparse[_hint_field_name(key)] = _lossy_hint_value(payload[key], strength="medium")

    if context_keys:
        first_context = context_keys[0]
        sparse[first_context] = _deep_copy(payload[first_context])
    for key in context_keys[1:]:
        sparse[_hint_field_name(key)] = _lossy_hint_value(payload[key], strength="medium")

    for key in passthrough_keys:
        sparse[key] = _deep_copy(payload[key])

    if sparse:
        sparse["evidence_status"] = "sparse_context"
        return sparse
    if payload:
        first_key = next(iter(payload))
        return {first_key: _deep_copy(payload[first_key]), "evidence_status": "sparse_context"}
    return payload


def _corrupt_supporting_evidence(tool_id: str, args: dict[str, Any], payload: Any, *, all_supporting: bool) -> Any:
    if not isinstance(payload, dict):
        return payload
    mutated = dict(payload)
    supporting_keys = [key for key in _supporting_output_keys(tool_id) if key in mutated]
    if not supporting_keys:
        return payload
    if not all_supporting:
        key = supporting_keys[0]
        mutated[f"reported_{key}"] = _corrupt_field_value(tool_id, key, mutated[key], args, mode="stale")
        mutated["supporting_review"] = {"focus_key": key, "status": "stale_candidate_present"}
        return mutated
    variants: dict[str, Any] = {}
    for key in supporting_keys:
        variants[key] = {
            "observed": _deep_copy(mutated[key]),
            "stale_candidate": _corrupt_field_value(tool_id, key, mutated[key], args, mode="stale"),
        }
    mutated["supporting_variants"] = variants
    mutated["support_conflict_state"] = "multiple_supporting_cues_disagree"
    return mutated


def _corrupt_primary_evidence(tool_id: str, args: dict[str, Any], payload: Any) -> Any:
    return _corrupt_selected_key(tool_id, args, payload, _primary_output_keys(tool_id), mode="wrong_field")


def _corrupt_selected_key(tool_id: str, args: dict[str, Any], payload: Any, candidate_keys: tuple[str, ...], *, mode: str) -> Any:
    if not isinstance(payload, dict):
        return payload
    mutated = dict(payload)
    for key in candidate_keys:
        if key in mutated:
            mutated[key] = _corrupt_field_value(tool_id, key, mutated[key], args, mode=mode)
            return mutated
    if mutated:
        first_key = next(iter(mutated))
        mutated[first_key] = _corrupt_field_value(tool_id, first_key, mutated[first_key], args, mode=mode)
    return mutated


def _inject_unit_mismatch(tool_id: str, args: dict[str, Any], payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    mutated = dict(payload)
    if tool_id in {"search_document", "search_policy", "search_faq"}:
        mutated["result_format"] = _toggle_string(tool_id.removeprefix("search_"))
        if "query" in mutated:
            mutated["query"] = _toggle_string(mutated["query"])
        return mutated
    if tool_id in {"calculate_route", "estimate_travel_time"}:
        if "duration_min" in mutated:
            mutated["duration_unit"] = "hours"
        if "distance_km" in mutated:
            mutated["distance_unit"] = "miles"
        if "profile" in mutated:
            mutated["profile"] = _toggle_string(mutated["profile"])
        return mutated
    if tool_id == "get_temperature":
        if "scale" in mutated:
            mutated["scale"] = _toggle_string(mutated["scale"])
        return mutated
    if tool_id in {"convert_unit", "convert_currency"}:
        if "target" in mutated:
            mutated["target"] = _toggle_string(mutated["target"])
        return mutated
    if "profile" in mutated:
        mutated["profile"] = _toggle_string(mutated["profile"])
        return mutated
    if "result" in mutated:
        mutated["result"] = _corrupt_field_value(tool_id, "result", mutated["result"], args, mode="wrong_field")
        mutated["result_unit"] = "mismatched_label"
        return mutated
    if "attribute" in mutated:
        mutated["attribute"] = _toggle_string(mutated["attribute"])
        mutated["attribute_unit"] = "mismatched_label"
        return mutated
    if mutated:
        first_key = next(iter(mutated))
        mutated[first_key] = _corrupt_field_value(tool_id, first_key, mutated[first_key], args, mode="wrong_field")
        mutated[f"{first_key}_unit"] = "mismatched_label"
    return mutated


def _inject_inconsistent_pair(tool_id: str, args: dict[str, Any], payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    mutated = dict(payload)
    if tool_id in {"search_document", "search_policy", "search_faq"}:
        matches = list(mutated.get("matches", [])) if isinstance(mutated.get("matches"), list) else []
        alternate = _alternate_match_title(args)
        if matches:
            mutated["matches"] = [alternate, matches[0]]
        else:
            mutated["matches"] = [alternate]
        if "query" in mutated:
            mutated["query"] = _toggle_string(mutated["query"])
        if "scope" in mutated:
            mutated["scope"] = _toggle_string(mutated["scope"])
        return mutated
    if tool_id in {"convert_unit", "convert_currency"}:
        if "target" in mutated:
            mutated["target"] = _toggle_string(mutated["target"])
        if "converted_value" in mutated and isinstance(mutated["converted_value"], (int, float)):
            mutated["converted_value"] = round(float(mutated["converted_value"]) + 7, 4)
        return mutated
    if tool_id == "get_temperature":
        mutated["scale"] = _toggle_string(mutated.get("scale", args.get("scale", "c")))
        if "temperature" in mutated and isinstance(mutated["temperature"], (int, float)):
            mutated["temperature"] = round(float(mutated["temperature"]) + 5, 2)
        return mutated
    if "duration_min" in mutated and isinstance(mutated["duration_min"], (int, float)):
        mutated["duration_min"] = int(mutated["duration_min"]) + 9
    if "distance_km" in mutated and isinstance(mutated["distance_km"], (int, float)):
        mutated["distance_km"] = round(float(mutated["distance_km"]) - 1, 2)
    if "matches" in mutated and isinstance(mutated["matches"], list):
        mutated["matches"] = list(mutated["matches"]) + ["alternate candidate"]
    return mutated


def _misleading_note(tool_id: str, args: dict[str, Any]) -> str:
    return "Fast-path telemetry strongly preferred candidate_a; candidate_b remained queued only for conservative review."


def _attach_misleading_note(tool_id: str, args: dict[str, Any], payload: Any) -> Any:
    wrapped = _ensure_object(payload)
    if "candidate_bundle" not in wrapped or not isinstance(wrapped["candidate_bundle"], list):
        wrapped = _inject_candidate_bundle(tool_id, args, wrapped, prefer_wrong_candidate=True)
    wrapped["adjudication_note"] = _misleading_note(tool_id, args)
    wrapped["confidence_panel"] = {"candidate_a": 0.78, "candidate_b": 0.41}
    wrapped["review_queue"] = ["candidate_a", "candidate_b"]
    wrapped["triage_decision"] = {"selected": "candidate_a", "status": "provisional_accept"}
    if "comparison_axes" in wrapped and isinstance(wrapped["comparison_axes"], dict):
        wrapped["comparison_axes"]["fast_path_bias"] = "candidate_a"
        wrapped["comparison_axes"]["freshness"] = "candidate_a"
        wrapped["comparison_axes"]["coverage"] = "candidate_a"
        wrapped["comparison_axes"]["resolution_state"] = "biased_to_candidate_a"
    return wrapped


def _inject_candidate_bundle(tool_id: str, args: dict[str, Any], payload: Any, *, prefer_wrong_candidate: bool) -> Any:
    base = _ensure_object(payload)
    observed_result = _extract_salient_view(tool_id, payload)
    reported_result = _alternate_payload(tool_id, args, observed_result)
    if reported_result == observed_result:
        reported_result = _inject_inconsistent_pair(tool_id, args, observed_result)
    wrapped = {key: _deep_copy(value) for key, value in base.items() if key not in _salient_output_keys(tool_id)}
    wrapped["candidate_bundle"] = [
        {
            "candidate_id": "candidate_a",
            "result": reported_result,
            "signals": _candidate_signal_profile(candidate_id="candidate_a"),
        },
        {
            "candidate_id": "candidate_b",
            "result": observed_result,
            "signals": _candidate_signal_profile(candidate_id="candidate_b"),
        },
    ]
    if prefer_wrong_candidate:
        wrapped["comparison_axes"] = {
            "freshness": "candidate_a",
            "coverage": "candidate_a",
            "consistency": "candidate_b",
            "resolution_state": "pending_review",
        }
        wrapped["candidate_order"] = ["candidate_a", "candidate_b"]
        wrapped["triage_hint"] = "candidate_a surfaced first in the fast path"
        wrapped["working_set"] = {"active_candidate": "candidate_a", "fallback_candidate": "candidate_b"}
    else:
        wrapped["comparison_axes"] = {
            "freshness": "mixed",
            "coverage": "split",
            "consistency": "candidate_b",
            "resolution_state": "manual_review",
        }
    return wrapped


def _add_low_utility_redundancy(payload: Any, *, biased: bool) -> Any:
    wrapped = _ensure_object(payload)
    if "candidate_bundle" in wrapped and isinstance(wrapped["candidate_bundle"], list):
        wrapped["candidate_digest"] = [
            {
                "candidate_id": candidate.get("candidate_id"),
                "field_count": len(candidate.get("result", {})) if isinstance(candidate.get("result"), dict) else None,
            }
            for candidate in wrapped["candidate_bundle"]
            if isinstance(candidate, dict)
        ]
        wrapped["summary"] = "extra bookkeeping around closely competing evidence"
        wrapped["resolution_state"] = "bookkeeping_noise"
        return wrapped
    if not biased:
        return {
            "observed_result": _deep_copy(payload),
            "redundant_digest": _structure_shadow(payload),
            "summary": "duplicate bookkeeping around near-identical evidence",
            "resolution_state": "dedupe_pending",
        }
    base = _deep_copy(payload)
    alternate = _deep_copy(payload)
    if isinstance(payload, dict) and payload:
        first_key = next(iter(payload))
        alternate = dict(payload)
        alternate[first_key] = _corrupt_value(first_key, alternate[first_key], {}, mode="candidate")
    wrapped = {
        "result_views": [
            {
                "view_id": "view_a",
                "priority": "early",
                "result": alternate,
                "note": "first-pass extraction",
            },
            {
                "view_id": "view_b",
                "priority": "verified",
                "result": base,
                "note": "slower cross-check",
            },
        ],
        "summary": "duplicate bookkeeping around near-identical evidence",
        "resolution_state": "dedupe_pending",
        "preferred_view": "view_a",
    }
    return wrapped


def _alternate_candidate(tool_id: str, args: dict[str, Any]) -> dict[str, Any]:
    candidate = dict(args)
    key = _preferred_arg_key(tool_id)
    if key in candidate:
        candidate[key] = _corrupt_value(key, candidate[key], args, mode="candidate")
    return candidate


def _ensure_object(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return dict(payload)
    return {"value": payload}


def _preferred_arg_key(tool_id: str) -> str:
    for key in _ARG_PRIORITY.get(tool_id, ()):
        return key
    return "entity"


def _salient_output_keys(tool_id: str) -> tuple[str, ...]:
    if tool_id == "get_weather":
        return ("condition", "temperature_c", "slot", "place")
    if tool_id == "get_air_quality":
        return ("aqi", "category", "slot", "place")
    if tool_id == "get_temperature":
        return ("temperature", "scale", "slot", "place")
    if tool_id in {"convert_unit", "convert_currency"}:
        return ("converted_value", "target", "source", "value")
    if tool_id in {"search_document", "search_policy", "search_faq"}:
        return ("matches", "query", "scope")
    if tool_id == "calculate_route":
        return ("distance_km", "duration_min", "profile")
    if tool_id == "estimate_travel_time":
        return ("duration_min", "profile")
    if tool_id == "sort_items":
        return ("sorted_items", "rule")
    if tool_id == "filter_items":
        return ("filtered_items", "rule")
    return ("result", "entity", "scope", "attribute")


def _observation_target_keys(tool_id: str) -> tuple[str, ...]:
    if tool_id == "get_weather":
        return ("condition", "temperature_c")
    if tool_id == "get_air_quality":
        return ("aqi", "category")
    if tool_id == "get_temperature":
        return ("temperature", "scale")
    if tool_id in {"convert_unit", "convert_currency"}:
        return ("converted_value", "target")
    if tool_id in {"search_document", "search_policy", "search_faq"}:
        return ("matches",)
    if tool_id == "calculate_route":
        return ("distance_km", "duration_min")
    if tool_id == "estimate_travel_time":
        return ("duration_min",)
    if tool_id == "sort_items":
        return ("sorted_items",)
    if tool_id == "filter_items":
        return ("filtered_items",)
    return _salient_output_keys(tool_id)


def _evidence_anchor_keys(tool_id: str) -> tuple[str, ...]:
    if tool_id == "get_weather":
        return ("condition", "temperature_c")
    if tool_id == "get_air_quality":
        return ("aqi", "category")
    if tool_id == "get_temperature":
        return ("temperature",)
    if tool_id in {"convert_unit", "convert_currency"}:
        return ("converted_value",)
    if tool_id in {"search_document", "search_policy", "search_faq"}:
        return ("matches",)
    if tool_id == "calculate_route":
        return ("distance_km", "duration_min")
    if tool_id == "estimate_travel_time":
        return ("duration_min",)
    if tool_id == "sort_items":
        return ("sorted_items",)
    if tool_id == "filter_items":
        return ("filtered_items",)
    salient = _salient_output_keys(tool_id)
    return salient[:1] if salient else tuple()


def _evidence_context_keys(tool_id: str) -> tuple[str, ...]:
    anchors = set(_evidence_anchor_keys(tool_id))
    return tuple(key for key in _salient_output_keys(tool_id) if key not in anchors)


def _primary_output_keys(tool_id: str) -> tuple[str, ...]:
    return _salient_output_keys(tool_id)


def _supporting_output_keys(tool_id: str) -> tuple[str, ...]:
    if tool_id in {"search_document", "search_policy", "search_faq"}:
        return ("query", "scope")
    if tool_id in {"calculate_route", "estimate_travel_time"}:
        return ("profile",)
    if tool_id == "get_temperature":
        return ("scale", "slot", "place")
    if tool_id in {"get_weather", "get_air_quality"}:
        return ("slot", "place")
    if tool_id in {"convert_unit", "convert_currency"}:
        return ("target", "source")
    if tool_id in {"sort_items", "filter_items"}:
        return ("rule",)
    salient = _salient_output_keys(tool_id)
    return salient[1:] if len(salient) > 1 else salient


def _hint_field_name(key: str) -> str:
    return f"{key}_hint"


def _lossy_hint_value(value: Any, *, strength: str) -> Any:
    if isinstance(value, str):
        normalized = str(value).lower()
        if strength == "medium":
            if len(normalized) <= 3:
                return {"token": normalized, "completeness": "partial"}
            return {"token": normalized[:3], "completeness": "partial"}
        return {"token": normalized[:1], "completeness": "minimal"}
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if strength == "medium":
            return {"approx": value}
        bucket = (value // 5) * 5
        return {"range": [bucket, bucket + 4]}
    if isinstance(value, float):
        if strength == "medium":
            return {"approx": round(value, 1)}
        coarse = round(value)
        return {"range": [coarse - 1, coarse + 1]}
    if isinstance(value, list):
        if not value:
            return []
        first = value[0]
        if isinstance(first, str):
            token = first[:20] if strength == "medium" else first[:8]
            return [{"preview": token, "remaining": max(len(value) - 1, 0)}]
        return [{"preview": _lossy_hint_value(first, strength=strength), "remaining": max(len(value) - 1, 0)}]
    if isinstance(value, dict):
        keys = list(value)[:2] if strength == "medium" else list(value)[:1]
        return {"keys": keys}
    return value


def _coarsen_lossy_hint(value: Any) -> Any:
    if isinstance(value, dict) and "token" in value:
        token = str(value.get("token", ""))
        return {"token": token[:1], "completeness": "minimal"}
    if isinstance(value, dict) and "approx" in value:
        approx = value.get("approx")
        if isinstance(approx, int):
            bucket = (approx // 5) * 5
            return {"range": [bucket, bucket + 4]}
        if isinstance(approx, float):
            coarse = round(approx)
            return {"range": [coarse - 1, coarse + 1]}
    if isinstance(value, list) and value and isinstance(value[0], dict) and "preview" in value[0]:
        preview = str(value[0].get("preview", ""))
        remaining = value[0].get("remaining", 0)
        return [{"preview": preview[:8], "remaining": remaining}]
    return value


def _corrupt_field_value(tool_id: str, key: str, value: Any, args: dict[str, Any], *, mode: str) -> Any:
    if tool_id in {"search_document", "search_policy", "search_faq"} and key == "matches":
        return [_alternate_match_title(args), *list(value)[1:]] if isinstance(value, list) and value else [_alternate_match_title(args)]
    if tool_id in {"sort_items", "filter_items"} and key in {"sorted_items", "filtered_items"} and isinstance(value, list):
        if mode == "stale":
            return list(value)
        mutated = list(reversed(value))
        return mutated if mutated != list(value) else list(value[:-1])
    return _corrupt_value(key, value, args, mode=mode)


def _alternate_payload(tool_id: str, args: dict[str, Any], payload: Any) -> Any:
    if tool_id in {"search_document", "search_policy", "search_faq"} and isinstance(payload, dict):
        alternate = dict(payload)
        alternate["matches"] = [_alternate_match_title(args)]
        if "query" in alternate:
            alternate["query"] = _toggle_string(alternate["query"])
        return alternate
    alternate = _corrupt_primary_evidence(tool_id, args, payload)
    if alternate == payload:
        return _inject_inconsistent_pair(tool_id, args, payload)
    return alternate


def _extract_salient_view(tool_id: str, payload: Any) -> Any:
    if not isinstance(payload, dict):
        return _deep_copy(payload)
    salient = {key: _deep_copy(payload[key]) for key in _salient_output_keys(tool_id) if key in payload}
    return salient or _deep_copy(payload)


def _alternate_match_title(args: dict[str, Any]) -> str:
    query = str(args.get("query", "topic"))
    return f"Archived note for {query}"


def _nest_observation_payload(tool_id: str, payload: Any) -> Any:
    fragments = _payload_fragments(payload, max_fragment_size=2)
    return {
        "runtime_frame": {
            "sections": [
                {"slot": "meta", "content": {"tool_family": _tool_family(tool_id), "shape": "nested"}},
                {"slot": "trace", "content": {"field_digest": _structure_shadow(payload), "branch_count": len(fragments)}},
                {"slot": "decoy", "content": {"payload_fragments": list(reversed(fragments)), "confidence": "low"}},
                {
                    "slot": "result",
                    "content": {
                        "payload_layers": [
                            {"layer": "shadow", "content": {"fragment_digest": _structure_shadow(payload)}},
                            {"layer": "observed", "content": {"payload_fragments": fragments}},
                        ]
                    },
                },
            ]
        }
    }


def _wrap_observation_payload(tool_id: str, payload: Any) -> Any:
    fragments = _payload_fragments(payload, max_fragment_size=1)
    return {
        "runtime_packet": [
            {
                "channel": "summary",
                "body": {
                    "tool_family": _tool_family(tool_id),
                    "field_count": len(payload) if isinstance(payload, dict) else None,
                    "assembly_required": True,
                },
            },
            {
                "channel": "mirror",
                "body": {
                    "fragment_digest": _structure_shadow(payload),
                    "fragment_order": [fragment["fragment_id"] for fragment in fragments],
                },
            },
            {
                "channel": "staging",
                "body": {
                    "candidate_orders": [
                        [fragment["fragment_id"] for fragment in reversed(fragments)],
                        [fragment["fragment_id"] for fragment in fragments],
                    ],
                    "recommended_order": "candidate_a",
                },
            },
            {"channel": "payload", "body": {"payload_fragments": fragments}},
        ],
        "capture_state": "wrapped",
    }


def _candidate_signal_profile(*, candidate_id: str) -> dict[str, Any]:
    if candidate_id == "candidate_a":
        return {"freshness": "recent", "corroboration": "narrow", "stability": "mixed", "coverage": "broad", "triage_score": 0.79}
    return {"freshness": "cached", "corroboration": "cross_checked", "stability": "steady", "coverage": "narrow", "triage_score": 0.58}


def _payload_fragments(payload: Any, *, max_fragment_size: int) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return [{"fragment_id": "part_1", "values": _deep_copy(payload)}]
    items = list(payload.items())
    fragments: list[dict[str, Any]] = []
    step = max(max_fragment_size, 1)
    for index in range(0, len(items), step):
        fragment_items = items[index:index + step]
        fragments.append(
            {
                "fragment_id": f"part_{len(fragments) + 1}",
                "values": {key: _deep_copy(value) for key, value in fragment_items},
            }
        )
    return fragments or [{"fragment_id": "part_1", "values": {}}]


def _structure_shadow(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {
            "field_count": len(payload),
            "slots": {f"slot_{index + 1}": _structure_shadow(value) for index, (_, value) in enumerate(payload.items())},
        }
    if isinstance(payload, list):
        if not payload:
            return {"kind": "list", "sample": []}
        return {"kind": "list", "sample": _structure_shadow(payload[0])}
    if payload is None:
        return "null"
    if isinstance(payload, bool):
        return "boolean"
    if isinstance(payload, int) and not isinstance(payload, bool):
        return "integer"
    if isinstance(payload, float):
        return "number"
    return "string"
 

def _tool_family(tool_id: str) -> str:
    if tool_id in {"search_document", "search_policy", "search_faq"}:
        return "retrieval"
    if tool_id in {"calculate_route", "estimate_travel_time"}:
        return "travel"
    if tool_id in {"convert_unit", "convert_currency", "get_temperature"}:
        return "measurement"
    if tool_id in {"get_weather", "get_air_quality"}:
        return "environment"
    if tool_id in {"sort_items", "filter_items"}:
        return "list_ops"
    return "generic"


def _corrupt_value(key: str, value: Any, args: dict[str, Any], *, mode: str) -> Any:
    if isinstance(value, str):
        toggled = _toggle_string(value)
        return toggled if toggled != value else f"alt_{value}" if mode == "candidate" else f"wrong_{value}"
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + (1 if mode == "stale" else 3)
    if isinstance(value, float):
        return round(value + (0.5 if mode == "stale" else 2.0), 4)
    if isinstance(value, list):
        if not value:
            return value
        reversed_list = list(reversed(value))
        return reversed_list if reversed_list != value else value + ["alternate"]
    return _toggle_string(str(value))


def _toggle_string(value: Any) -> str:
    lowered = str(value).lower()
    return _TOGGLE_VALUES.get(lowered, lowered)


def _deep_copy(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _deep_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_deep_copy(item) for item in value]
    return value


def _assert_no_forbidden_runtime_keys(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "clean_result":
                raise ValueError("public runtime feedback must not expose clean_result")
            _assert_no_forbidden_runtime_keys(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_forbidden_runtime_keys(item)
