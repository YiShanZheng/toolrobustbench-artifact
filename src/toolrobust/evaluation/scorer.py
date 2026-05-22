"""Scoring logic for ToolRobust v3."""

from __future__ import annotations

from dataclasses import dataclass

from toolrobust.domain.schemas import BackendResponse, EvaluationRecord, TaskSample
from toolrobust.evaluation.failure_taxonomy import analyze_failure, decide_observed_failure, overall_success
from toolrobust.execution.runtime_feedback import build_observation_target
from toolrobust.tools.base import ToolExecutionError
from toolrobust.tools.executor import ToolExecutor
from toolrobust.tools.registry import ToolRegistry


@dataclass(frozen=True, slots=True)
class ExecutionExpectation:
    status: str | tuple[str, ...]
    retryable: bool
    final_result: object
    accepted_failure_types: tuple[str, ...] | None


class Scorer:
    def __init__(self, executor: ToolExecutor | None = None) -> None:
        self.executor = executor or ToolExecutor()

    def score(self, task: TaskSample, response: BackendResponse, registry: ToolRegistry) -> EvaluationRecord:
        predicted_tool_raw = response.parsed.predicted_tool_raw
        predicted_tool_resolved = registry.resolve_tool_name(predicted_tool_raw)
        normalized_args = registry.normalize_arguments(predicted_tool_resolved, response.parsed.predicted_args_raw) if predicted_tool_resolved else {}
        call_present = int(response.parsed.call_present)
        format_valid = int(response.parsed.format_valid)
        interface_name_match = int(predicted_tool_raw == task.exposed_tool_name)
        canonical_tool_match = int(predicted_tool_resolved == task.canonical_tool_id)
        missing_parameter_flag = int(self._missing_required_parameters(task, predicted_tool_resolved, normalized_args, registry)) if format_valid else 0
        wrong_parameter_flag = int(self._wrong_parameters(task, predicted_tool_resolved, normalized_args)) if format_valid and missing_parameter_flag == 0 else 0
        execution_success = 0
        notes = response.parsed.parser_message
        auxiliary = response.auxiliary or {}
        decision_status_audit: str | None = None
        decision_retryable_audit: bool | None = None
        decision_failure_type_audit: str | None = None
        decision_final_result_summary: str | None = None
        expected_statuses_audit: str | None = None
        expected_retryable_audit: bool | None = None
        expected_failure_types_audit: str | None = None
        expected_final_result_summary: str | None = None
        stage2_status_match: bool | None = None
        stage2_retryable_match: bool | None = None
        stage2_failure_type_match: bool | None = None
        stage2_result_match: bool | None = None

        if task.has_family("observation") and not task.has_family("execution_environment") and "runtime_execution_success" in auxiliary:
            stage1_tool_resolved = auxiliary.get("stage1_predicted_tool_resolved")
            stage1_normalized_args = auxiliary.get("stage1_normalized_arguments", {})
            stage1_missing = self._missing_required_parameters(task, stage1_tool_resolved, stage1_normalized_args, registry)
            stage1_wrong = self._wrong_parameters(task, stage1_tool_resolved, stage1_normalized_args)
            stage1_ready = bool(
                stage1_tool_resolved == task.canonical_tool_id
                and not stage1_missing
                and not stage1_wrong
            )
            observation_format_valid = int(auxiliary.get("observation_stage2_format_valid", 0))
            observation_result = auxiliary.get("observation_stage2_result")
            observation_target = auxiliary.get("observation_target", task.expected_result)
            observation_match = int(
                observation_format_valid == 1
                and self._observation_result_matches(observation_target, observation_result)
            )
            format_valid = int(response.parsed.format_valid)
            if stage1_ready:
                format_valid = observation_format_valid
            missing_parameter_flag = int(stage1_missing) if int(response.parsed.format_valid) == 1 else 0
            wrong_parameter_flag = int(stage1_wrong or (observation_format_valid == 1 and observation_match == 0))
            execution_success = int(stage1_ready and int(auxiliary.get("runtime_execution_success", 0)) == 1 and observation_format_valid == 1)
            observation_note = str(auxiliary.get("observation_stage2_parser_message") or "")
            if observation_format_valid == 1 and observation_match == 0:
                observation_note = "; ".join(
                    part for part in (observation_note, "observation target mismatch") if part
                )
            runtime_note = str(auxiliary.get("runtime_note") or "")
            runtime_status = str(auxiliary.get("runtime_status") or "")
            details = [part for part in (runtime_status, runtime_note, observation_note) if part]
            if details:
                notes = "; ".join(part for part in (notes, "runtime " + " / ".join(details)) if part)
        elif task.has_family("execution_environment") and "runtime_execution_success" in auxiliary:
            stage1_tool_resolved = auxiliary.get("stage1_predicted_tool_resolved")
            stage1_normalized_args = auxiliary.get("stage1_normalized_arguments", {})
            stage1_missing = self._missing_required_parameters(task, stage1_tool_resolved, stage1_normalized_args, registry)
            stage1_wrong = self._wrong_parameters(task, stage1_tool_resolved, stage1_normalized_args)
            stage1_ready = bool(
                stage1_tool_resolved == task.canonical_tool_id
                and not stage1_missing
                and not stage1_wrong
            )
            decision_format_valid = int(auxiliary.get("execution_stage2_format_valid", 0))
            decision = auxiliary.get("execution_stage2_decision") or {}
            expectation = self._expected_execution_expectation(task, auxiliary)
            expected_statuses = expectation.status if isinstance(expectation.status, tuple) else (expectation.status,)
            decision_status = str(decision.get("status") or "")
            status_match = decision_status in expected_statuses
            retryable_match = decision.get("retryable") == expectation.retryable
            failure_type_match = self._failure_type_matches(expectation, decision.get("failure_type"))
            result_status = decision_status if status_match else expected_statuses[0]
            result_match = self._execution_result_matches(
                result_status,
                expectation.final_result,
                decision.get("final_result"),
            )
            decision_status_audit = decision_status or None
            decision_retryable_audit = decision.get("retryable") if isinstance(decision.get("retryable"), bool) else None
            decision_failure_type_audit = str(decision.get("failure_type")) if decision.get("failure_type") is not None else None
            decision_final_result_summary = self._result_summary(decision.get("final_result"))
            expected_statuses_audit = "|".join(str(status) for status in expected_statuses)
            expected_retryable_audit = expectation.retryable
            expected_failure_types_audit = None if expectation.accepted_failure_types is None else "|".join(expectation.accepted_failure_types)
            expected_final_result_summary = self._result_summary(expectation.final_result)
            stage2_status_match = bool(status_match)
            stage2_retryable_match = bool(retryable_match)
            stage2_failure_type_match = bool(failure_type_match)
            stage2_result_match = bool(result_match)
            format_valid = int(response.parsed.format_valid)
            if stage1_ready:
                format_valid = decision_format_valid
            missing_parameter_flag = int(stage1_missing) if int(response.parsed.format_valid) == 1 else 0
            expects_result = decision_status in {"success", "partial"} if status_match else any(status in {"success", "partial"} for status in expected_statuses)
            wrong_parameter_flag = int(stage1_wrong or (decision_format_valid == 1 and expects_result and not result_match))
            execution_success = int(
                stage1_ready
                and decision_format_valid == 1
                and status_match
                and retryable_match
                and failure_type_match
                and result_match
            )
            decision_note = str(auxiliary.get("execution_stage2_parser_message") or "")
            if decision_format_valid == 1 and not status_match:
                decision_note = "; ".join(part for part in (decision_note, "execution status mismatch") if part)
            if decision_format_valid == 1 and not retryable_match:
                decision_note = "; ".join(part for part in (decision_note, "execution retryability mismatch") if part)
            if decision_format_valid == 1 and not failure_type_match:
                decision_note = "; ".join(part for part in (decision_note, "execution failure-type mismatch") if part)
            if decision_format_valid == 1 and expects_result and not result_match:
                decision_note = "; ".join(part for part in (decision_note, "execution result mismatch") if part)
            runtime_note = str(auxiliary.get("runtime_note") or "")
            runtime_status = str(auxiliary.get("runtime_status") or "")
            details = [part for part in (runtime_status, runtime_note, decision_note) if part]
            if details:
                notes = "; ".join(part for part in (notes, "runtime " + " / ".join(details)) if part)
        elif (task.has_family("observation") or task.has_family("execution_environment")) and "runtime_execution_success" in auxiliary:
            stage1_tool_resolved = auxiliary.get("stage1_predicted_tool_resolved")
            stage1_normalized_args = auxiliary.get("stage1_normalized_arguments", {})
            stage1_missing = self._missing_required_parameters(task, stage1_tool_resolved, stage1_normalized_args, registry)
            stage1_wrong = self._wrong_parameters(task, stage1_tool_resolved, stage1_normalized_args)
            stage1_ready = bool(
                stage1_tool_resolved == task.canonical_tool_id
                and not stage1_missing
                and not stage1_wrong
            )
            execution_success = int(stage1_ready and int(auxiliary.get("runtime_execution_success", 0)) == 1)
            runtime_note = str(auxiliary.get("runtime_note") or "")
            runtime_status = str(auxiliary.get("runtime_status") or "")
            if runtime_status or runtime_note:
                details = [part for part in (runtime_status, runtime_note) if part]
                notes = "; ".join(part for part in (notes, "runtime " + " / ".join(details)) if part)
        elif call_present and format_valid and predicted_tool_resolved and canonical_tool_match and not missing_parameter_flag and not wrong_parameter_flag:
            try:
                self.executor.execute(predicted_tool_resolved, normalized_args)
                execution_success = 1
            except (ToolExecutionError, KeyError, ValueError, TypeError) as exc:
                notes = f"{notes}; runtime execution error: {exc}"

        overall_task_success = int(
            overall_success(canonical_tool_match, format_valid, missing_parameter_flag, wrong_parameter_flag, execution_success, call_present)
        )
        decision = decide_observed_failure(
            call_present=call_present,
            format_valid=format_valid,
            predicted_tool_raw=predicted_tool_raw,
            predicted_tool_resolved=predicted_tool_resolved,
            interface_name_match=interface_name_match,
            canonical_tool_match=canonical_tool_match,
            missing_parameter_flag=missing_parameter_flag,
            wrong_parameter_flag=wrong_parameter_flag,
            execution_success=execution_success,
            notes=notes,
        )
        failure = analyze_failure(
            task=task,
            decision=decision,
            call_present=call_present,
            canonical_tool_match=canonical_tool_match,
            format_valid=format_valid,
            interface_name_match=interface_name_match,
            missing_parameter_flag=missing_parameter_flag,
            wrong_parameter_flag=wrong_parameter_flag,
            execution_success=execution_success,
            notes=notes,
        )

        return EvaluationRecord(
            sample_id=task.sample_id,
            model_name=response.model_name,
            backend_name=response.backend_name,
            split=task.split,
            family=task.family,
            subtype=task.subtype,
            operator=task.operator,
            severity=task.severity,
            stage_target=task.stage_target,
            source_sample_id=str(task.metadata.get("source_sample_id", task.sample_id)),
            registry_variant_id=task.registry_variant_id,
            composition_id=task.composition_id,
            families_applied=list(task.families_applied),
            subtypes_applied=list(task.subtypes_applied),
            stage_targets_applied=list(task.stage_targets_applied),
            primary_family=task.primary_family,
            gold_canonical_tool_id=task.canonical_tool_id,
            gold_exposed_tool_name=task.exposed_tool_name,
            operator_names=list(task.operators_applied),
            expected_arguments=task.expected_arguments,
            predicted_arguments=response.parsed.predicted_args_raw,
            expected_result=task.expected_result,
            predicted_tool_raw=predicted_tool_raw,
            predicted_tool_resolved=predicted_tool_resolved,
            call_present=call_present,
            interface_name_match=interface_name_match,
            canonical_tool_match=canonical_tool_match,
            format_valid=format_valid,
            missing_parameter_flag=missing_parameter_flag,
            wrong_parameter_flag=wrong_parameter_flag,
            execution_success=execution_success,
            overall_task_success=overall_task_success,
            decision_status=decision_status_audit,
            decision_retryable=decision_retryable_audit,
            decision_failure_type=decision_failure_type_audit,
            decision_final_result_summary=decision_final_result_summary,
            expected_statuses=expected_statuses_audit,
            expected_retryable=expected_retryable_audit,
            expected_failure_types=expected_failure_types_audit,
            expected_final_result_summary=expected_final_result_summary,
            stage2_status_match=stage2_status_match,
            stage2_retryable_match=stage2_retryable_match,
            stage2_failure_type_match=stage2_failure_type_match,
            stage2_result_match=stage2_result_match,
            status=failure.status,
            observed_failure_tag=failure.observed_failure_tag,
            primary_failure_tag=failure.primary_failure_tag,
            first_failure_stage=failure.first_failure_stage,
            is_cascade_failure=failure.is_cascade_failure,
            cascade_from_stage=failure.cascade_from_stage,
            expected_primary_match=failure.expected_primary_match,
            allowed_spillover_match=failure.allowed_spillover_match,
            boundary_violation_flag=failure.boundary_violation_flag,
            boundary_violation_reason=failure.boundary_violation_reason,
            secondary_reason=failure.secondary_reason,
            notes=notes,
            latency_ms=response.latency_ms,
            cost_usd=response.cost_usd,
        )

    @staticmethod
    def _missing_required_parameters(task: TaskSample, predicted_tool_resolved: str | None, normalized_args: dict[str, object], registry: ToolRegistry) -> bool:
        if predicted_tool_resolved != task.canonical_tool_id:
            return False
        required = registry.required_canonical_parameters(task.canonical_tool_id)
        return any(parameter not in normalized_args for parameter in required)

    @staticmethod
    def _wrong_parameters(task: TaskSample, predicted_tool_resolved: str | None, normalized_args: dict[str, object]) -> bool:
        if predicted_tool_resolved != task.canonical_tool_id:
            return False
        return any(normalized_args.get(key) != expected for key, expected in task.expected_arguments.items())

    @classmethod
    def _observation_result_matches(cls, expected: object, observed: object) -> bool:
        expected_normalized = cls._normalize_expected(expected)
        observed_normalized = cls._normalize_observed(observed, expected)
        return expected_normalized == observed_normalized

    @classmethod
    def _execution_result_matches(cls, expected_status: str, expected: object, observed: object) -> bool:
        if expected_status == "error":
            return observed is None
        return cls._observation_result_matches(expected, observed)

    @classmethod
    def _expected_execution_expectation(cls, task: TaskSample, auxiliary: dict[str, object]) -> ExecutionExpectation:
        runtime_success = int(auxiliary.get("runtime_execution_success", 0)) == 1
        runtime_content = auxiliary.get("runtime_feedback")
        clean_result = auxiliary.get("runtime_clean_result")
        if runtime_success:
            return ExecutionExpectation("success", False, build_observation_target(task.canonical_tool_id, clean_result), ("none",))
        partial_payload = cls._extract_partial_runtime_payload(runtime_content, clean_result)
        if partial_payload is not None:
            return ExecutionExpectation(
                "partial",
                cls._expected_retryable(runtime_content),
                partial_payload,
                cls._runtime_failure_type_candidates(runtime_content, include_partial=True),
            )
        return ExecutionExpectation(
            "error",
            cls._expected_retryable(runtime_content),
            None,
            cls._runtime_failure_type_candidates(runtime_content, include_partial=False),
        )

    @classmethod
    def _runtime_failure_type_candidates(cls, payload: object, *, include_partial: bool) -> tuple[str, ...]:
        labels: set[str] = set()
        expected = cls._expected_failure_type(payload)
        if expected:
            labels.add(cls._canonical_failure_label(expected))
        if include_partial:
            labels.add("partial_response")
        if isinstance(payload, dict):
            for key in ("error_type", "reported_issue", "diagnostic_code", "schema_mode", "attempt_consistency"):
                value = payload.get(key)
                if value is not None:
                    labels.add(cls._canonical_failure_label(value))
            if payload.get("retryable_hint") is True:
                labels.update({"runtime_error", "service_unavailable", "vague_error"})
            raw = payload.get("raw")
            if isinstance(raw, dict):
                for key in ("error_type", "reported_issue", "diagnostic_code", "schema_mode", "attempt_consistency", "completion_state"):
                    value = raw.get(key)
                    if value is not None:
                        labels.add(cls._canonical_failure_label(value))
        labels.discard("")
        return tuple(sorted(labels or {"vague_error"}))

    @classmethod
    def _failure_type_matches(cls, expectation: ExecutionExpectation, observed: object) -> bool:
        if expectation.accepted_failure_types is None:
            return True
        observed_label = cls._canonical_failure_label(observed)
        accepted = {cls._canonical_failure_label(value) for value in expectation.accepted_failure_types}
        if accepted == {"none"}:
            return observed_label in {"", "none", "null"}
        return observed_label in accepted

    @classmethod
    def _canonical_failure_label(cls, value: object) -> str:
        label = cls._normalize_failure_label(value)
        if label in {"", "none", "null"}:
            return label
        aliases = {
            "timeout_after_partial_stream": "timeout",
            "timeout_stream": "timeout",
            "stream_timeout": "timeout",
            "timeout_stream_interrupted": "timeout",
            "timeout_stream_stalled": "timeout",
            "stream_interrupted_timeout": "timeout",
            "stream_stalled_timeout": "timeout",
            "stream_interrupted": "timeout",
            "stream_stalled": "timeout",
            "connection_timeout": "timeout",
            "request_timeout": "timeout",
            "deadline_exceeded": "timeout",
            "missing_input": "wrong_failure_label",
            "missing_parameter": "wrong_failure_label",
            "validation_error": "wrong_failure_label",
            "bad_request": "wrong_failure_label",
            "schema_parse_failure": "wrong_failure_label",
            "malformed_response": "wrong_failure_label",
            "response_malformed": "wrong_failure_label",
            "response_envelope_malformed": "wrong_failure_label",
            "misleading_error": "wrong_failure_label",
            "contradictory_error": "wrong_failure_label",
            "conflicting_diagnostics": "wrong_failure_label",
            "competing_diagnostics": "wrong_failure_label",
            "conflicting_previews": "wrong_failure_label",
            "corrupted_response": "wrong_failure_label",
            "payload_integrity_error": "wrong_failure_label",
            "partial": "partial_response",
            "partial_result": "partial_response",
            "incomplete_response": "partial_response",
            "truncated_response": "partial_response",
            "degraded_transport": "runtime_variability",
            "runtime_attempt_conflict": "runtime_variability",
            "attempt_resolution_conflict": "runtime_variability",
            "attempt_conflict": "runtime_variability",
            "partial_response_truncated": "partial_response",
            "truncated_response": "partial_response",
            "truncated_partial": "partial_response",
            "service_unavailable_stale_fallback": "service_unavailable",
            "service_unavailable_stale_cache_fallback": "service_unavailable",
            "service_unavailable_with_stale_fallback": "service_unavailable",
            "service_unavailable_with_stale_fragment": "service_unavailable",
            "primary_upstream_unavailable_stale_fallback": "service_unavailable",
            "upstream_service_unavailable_stale_fallback": "service_unavailable",
            "stale_fallback": "service_unavailable",
            "stale_cache_fallback": "service_unavailable",
            "stale_result": "service_unavailable",
            "stale_data": "service_unavailable",
            "stale_cache_fallback_service_unavailable": "service_unavailable",
            "stale_fallback_from_service_unavailable": "service_unavailable",
            "stale_fallback_from_upstream_unavailability": "service_unavailable",
            "attempt_conflict_with_tail_missing": "runtime_variability",
            "attempt_conflict_tail_missing": "runtime_variability",
            "conflicting_attempts_tail_missing": "runtime_variability",
            "attempt_conflict_with_incomplete_delivery": "runtime_variability",
            "incomplete_delivery_with_conflicting_attempts": "runtime_variability",
            "attempt_resolution_conflict_with_contaminated_replay": "runtime_variability",
            "wrong_failure_label_conflicting_unverified_tail": "runtime_variability",
            "unstable_runtime": "runtime_variability",
            "transport_variability": "runtime_variability",
            "shape_changed": "runtime_variability",
            "shape_mismatch": "runtime_variability",
            "transient_error": "runtime_error",
            "temporary_error": "runtime_error",
            "temporary_dependency_unclear": "vague_error",
            "unknown": "vague_error",
        }
        if (
            "wrong_failure_label" in label
            or "schema_parse_failure" in label
            or "schema_parse_error" in label
            or "malformed_response" in label
            or "response_malformed" in label
            or "label_conflict" in label
            or "conflicting_diagnostic" in label
            or "competing_diagnostic" in label
            or "conflicting_preview" in label
            or "conflicting_lossy" in label
            or "conflicting_hint" in label
            or "conflicting_payload" in label
            or "conflicting_partial" in label
            or "conflicting partial" in label
            or "contradictory_payload" in label
            or "corrupted_response" in label
            or "payload_integrity_error" in label
            or "transport_tail_missing_with_conflicting" in label
            or "transport_payload_missing_with_conflicting" in label
            or "transport_and_parse_failure" in label
            or "transport_error_with_conflicting" in label
            or "transport_failure_with_conflicting" in label
            or "transport_incomplete_with_conflicting" in label
            or "transport_incomplete_conflicting" in label
            or "incomplete_transport_with_conflicting" in label
            or "incomplete_payload_with_conflicting" in label
            or "incomplete_response_with_conflicting" in label
            or "incomplete_and_conflicting" in label
            or "incomplete_and_contradictory" in label
            or "transport_incomplete_and_conflicting" in label
            or "transport_tail_missing_and_conflicting" in label
        ):
            return "wrong_failure_label"
        if "service_unavailable" in label or "upstream_unavailable" in label or "upstream_service_unavailable" in label or "dependency_unavailable" in label:
            return "service_unavailable"
        if "stale" in label and any(term in label for term in ("fallback", "cache", "result", "data")):
            return "service_unavailable"
        if (
            "attempt_conflict" in label
            or "conflicting_attempt" in label
            or "attempt_resolution_conflict" in label
            or "contaminated_replay" in label
        ):
            return "runtime_variability"
        if "partial" in label or "recoverable_fragment" in label or "recoverable" in label or "truncated" in label:
            return "partial_response"
        if label in {"request_failed", "tool_execution_error", "tool_error"}:
            return "runtime_error"
        if "timeout" in label and ("stream" in label or "stalled" in label or "interrupted" in label):
            return "timeout"
        return aliases.get(label, label)

    @classmethod
    def _is_partial_runtime_payload(cls, payload: object) -> bool:
        return cls._extract_partial_runtime_payload(payload, None) is not None

    @classmethod
    def _extract_partial_runtime_payload(cls, payload: object, clean_result: object | None) -> object:
        if not isinstance(payload, dict):
            return None
        if payload.get("status") == "partial" and "partial" in payload:
            return payload.get("partial")
        if "partial" in payload and cls._normalize_failure_label(payload.get("error_type")) != "shape_mismatch":
            return payload.get("partial")
        if payload.get("completion") in {"degraded", "partial"} and "result" in payload:
            return payload.get("result")
        if "fragment" in payload and cls._looks_like_partial_payload(payload.get("fragment"), clean_result):
            return payload.get("fragment")
        if "data" in payload and cls._looks_like_partial_payload(payload.get("data"), clean_result):
            return payload.get("data")
        if "candidate_fragment" in payload and cls._looks_like_partial_payload(payload.get("candidate_fragment"), clean_result):
            return payload.get("candidate_fragment")
        if "recoverable_fragment" in payload and cls._looks_like_partial_payload(payload.get("recoverable_fragment"), clean_result):
            return payload.get("recoverable_fragment")
        for nested_key in ("payload", "raw", "fragment", "data", "compat_envelope", "envelope", "recovery_trace", "recovery", "audit_tail"):
            nested = payload.get(nested_key)
            if isinstance(nested, dict):
                extracted = cls._extract_partial_runtime_payload(nested, clean_result)
                if extracted is not None:
                    return extracted
        raw = payload.get("raw")
        if isinstance(raw, dict):
            nested = cls._extract_partial_runtime_payload(raw, clean_result)
            if nested is not None:
                return nested
            if cls._looks_like_partial_payload(raw, clean_result):
                return raw
        if cls._looks_like_partial_payload(payload, clean_result):
            return payload
        return None

    @classmethod
    def _expected_failure_type(cls, payload: object) -> object:
        if payload == {}:
            return "empty_response"
        if isinstance(payload, dict):
            error_type = cls._normalize_failure_label(payload.get("error_type"))
            message = str(payload.get("message") or payload.get("detail") or payload.get("error") or "").strip().lower()
            status_code = payload.get("status_code")
            if error_type == "empty_response":
                return "empty_response"
            if payload.get("response_meta", {}).get("body_bytes") == 0:
                return "empty_response"
            diagnostic_panel = payload.get("diagnostic_panel")
            if isinstance(diagnostic_panel, dict) and diagnostic_panel.get("received_invocation") == "complete" and payload.get("reported_issue") == "missing_input":
                return "wrong_failure_label"
            if isinstance(payload.get("raw"), dict) and payload.get("reported_issue") in {"auth_denied", "schema_parse_failure"}:
                return "wrong_failure_label"
            if error_type == "timeout" or "timed out" in message or "execution window exceeded" in message or "timeout" in message or "timeout_ms" in payload or "elapsed_ms" in payload:
                return "timeout"
            if error_type == "service_unavailable" or status_code in {503, 529} or "unavailable" in message or "overloaded" in message:
                return "service_unavailable"
            if error_type == "auth_denied" or status_code in {401, 403} or "credential" in message or "authorization" in message or "reauth" in message or payload.get("challenge") == "reauth":
                return "auth_denied"
            if error_type == "missing_parameter" or payload.get("reported_issue") == "missing_input" or "missing" in payload or "required field missing" in message:
                return "misleading_error"
            if error_type == "schema_parse_failure" or payload.get("reported_issue") == "schema_parse_failure" or ("raw" in payload and ("malformed" in message or status_code in {500, 502})) or status_code == 502:
                return "wrong_failure_label"
            if error_type == "shape_mismatch" or payload.get("shape_variant") == "unstable" or payload.get("schema_mode") == "compat" or "shape changed unexpectedly" in message:
                return "shape_mismatch"
            if cls._extract_partial_runtime_payload(payload, None) is not None:
                return "partial_response"
            if "error" in payload or "message" in payload or "detail" in payload:
                return "vague_error"
        return None

    @classmethod
    def _expected_retryable(cls, payload: object) -> bool:
        if payload == {}:
            return True
        if isinstance(payload, dict):
            error_type = cls._normalize_failure_label(payload.get("error_type"))
            message = str(payload.get("message") or payload.get("detail") or payload.get("error") or "").strip().lower()
            status_code = payload.get("status_code")
            diagnostic_panel = payload.get("diagnostic_panel")
            raw = payload.get("raw")
            if isinstance(diagnostic_panel, dict) and diagnostic_panel.get("transport_retryable") is True:
                return True
            if isinstance(raw, dict) and raw.get("transport_retryable") is True:
                return True
            if payload.get("retryable_hint") is True or payload.get("transport_retryable") is True:
                return True
            if payload.get("completion_state") in {"truncated", "tail_missing", "stream_interrupted"}:
                return True
            if payload.get("response_meta", {}).get("body_bytes") == 0:
                return True
            if error_type in {"empty_response", "timeout", "service_unavailable", "shape_mismatch", "schema_parse_failure"}:
                return True
            if status_code in {500, 502, 503, 529} or "unavailable" in message or "overloaded" in message or "timed out" in message or "timeout" in message or payload.get("shape_variant") == "unstable" or payload.get("schema_mode") == "compat" or "elapsed_ms" in payload:
                return True
            if error_type == "auth_denied" or status_code in {401, 403} or payload.get("challenge") == "reauth":
                return False
            if error_type == "missing_parameter" or status_code == 400 or "required field missing" in message:
                return False
            if "error" in payload or "message" in payload:
                return True
        return False

    @staticmethod
    def _normalize_failure_label(value: object) -> str:
        return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")

    @classmethod
    def _result_summary(cls, value: object) -> str:
        if value is None:
            return "null"
        if isinstance(value, dict):
            keys = ",".join(str(key) for key in list(value)[:5])
            suffix = "" if len(value) <= 5 else ",..."
            return f"dict[{keys}{suffix}]"
        if isinstance(value, list):
            return f"list[len={len(value)}]"
        return type(value).__name__

    @classmethod
    def _looks_like_partial_payload(cls, candidate: object, clean_result: object | None) -> bool:
        if clean_result is None:
            return False
        if candidate == clean_result:
            return False
        return cls._is_partial_subset(candidate, clean_result)

    @classmethod
    def _is_partial_subset(cls, candidate: object, clean_result: object) -> bool:
        if isinstance(candidate, dict) and isinstance(clean_result, dict):
            if not candidate:
                return False
            if not set(candidate).issubset(set(clean_result)):
                return False
            reduced = len(candidate) < len(clean_result)
            for key, value in candidate.items():
                if key not in clean_result:
                    return False
                if not cls._is_partial_subset(value, clean_result[key]) and value != clean_result[key]:
                    return False
                reduced = reduced or value != clean_result[key]
            return reduced
        if isinstance(candidate, list) and isinstance(clean_result, list):
            if not candidate or len(candidate) > len(clean_result):
                return False
            reduced = len(candidate) < len(clean_result)
            for index, value in enumerate(candidate):
                if index >= len(clean_result):
                    return False
                if not cls._is_partial_subset(value, clean_result[index]) and value != clean_result[index]:
                    return False
                reduced = reduced or value != clean_result[index]
            return reduced
        if isinstance(candidate, str) and isinstance(clean_result, str):
            return bool(candidate) and candidate != clean_result and clean_result.startswith(candidate)
        return False

    @classmethod
    def _normalize_expected(cls, value: object) -> object:
        if isinstance(value, dict):
            return {key: cls._normalize_expected(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._normalize_expected(item) for item in value]
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, float):
            return round(value, 6)
        return value

    @classmethod
    def _normalize_observed(cls, observed: object, expected: object) -> object:
        if isinstance(expected, dict):
            if not isinstance(observed, dict):
                return observed
            unwrapped = cls._unwrap_result_envelope(observed)
            if not isinstance(unwrapped, dict):
                return unwrapped
            normalized: dict[str, object] = {}
            for expected_key, expected_value in expected.items():
                matched_key = cls._match_result_key(unwrapped, expected_key)
                if matched_key is None:
                    normalized[expected_key] = object()
                    continue
                normalized[expected_key] = cls._normalize_observed(unwrapped[matched_key], expected_value)
            return normalized
        if isinstance(expected, list):
            if not isinstance(observed, list):
                return observed
            if not expected:
                return []
            template = expected[0]
            return [cls._normalize_observed(item, template) for item in observed]
        if isinstance(expected, str):
            return str(observed).strip() if observed is not None else observed
        if isinstance(expected, float):
            try:
                return round(float(observed), 6)
            except (TypeError, ValueError):
                return observed
        if isinstance(expected, int) and not isinstance(expected, bool):
            try:
                return int(observed)
            except (TypeError, ValueError):
                return observed
        return observed

    @classmethod
    def _unwrap_result_envelope(cls, payload: dict[str, object]) -> object:
        current: object = payload
        metadata_keys = {"delivery_meta", "runtime_meta", "_runtime_meta", "metadata", "trace"}
        while isinstance(current, dict):
            if len(current) == 1:
                key = next(iter(current))
                if key not in {"result", "payload", "final_result"}:
                    break
                current = current[key]
                continue
            if "result" in current and set(current).issubset(metadata_keys | {"result"}):
                current = current["result"]
                continue
            break
        return current

    @staticmethod
    def _match_result_key(payload: dict[str, object], expected_key: str) -> str | None:
        if expected_key in payload:
            return expected_key
        snake = Scorer._to_snake_case(expected_key)
        camel = Scorer._to_camel_case(expected_key)
        for candidate in payload:
            if candidate == snake or candidate == camel or Scorer._to_snake_case(candidate) == snake:
                return candidate
        return None

    @staticmethod
    def _to_snake_case(value: str) -> str:
        characters: list[str] = []
        for index, char in enumerate(value):
            if char.isupper() and index > 0 and value[index - 1] != "_":
                characters.append("_")
            characters.append(char.lower())
        return "".join(characters)

    @staticmethod
    def _to_camel_case(value: str) -> str:
        pieces = value.split("_")
        if len(pieces) == 1:
            return value
        return pieces[0] + "".join(piece.capitalize() for piece in pieces[1:])
