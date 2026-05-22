"""Offline mock backend for ToolRobust v3."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from toolrobust.backends.base import BackendBase
from toolrobust.domain.enums import BenchmarkFamily
from toolrobust.domain.schemas import TaskSample, ToolSpec
from toolrobust.execution.runtime_feedback import build_observation_target, run_perturbed_execution
from toolrobust.tools.executor import ToolExecutor
from toolrobust.tools.local_tools import TOOL_GROUPS
from toolrobust.tools.registry import RegistryBuilder


class MockBackend(BackendBase):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.executor = ToolExecutor()

    @property
    def name(self) -> str:
        return "mock"

    def predict(self, task: TaskSample, tool_registry: list[ToolSpec]):
        if task.has_family(BenchmarkFamily.EXECUTION_ENVIRONMENT.value):
            return self._predict_execution_stage(task, tool_registry)
        if task.has_family(BenchmarkFamily.OBSERVATION.value):
            return self._predict_observation_stage(task, tool_registry)

        predicted_tool_raw, predicted_args_raw = self._mock_stage1_invocation(task, tool_registry)
        call_present = True
        format_valid = True
        parser_message = "mock backend deterministic response"

        return self.default_parsed(
            raw_response={"tool_name": predicted_tool_raw, "arguments": predicted_args_raw},
            predicted_tool_raw=predicted_tool_raw,
            predicted_args_raw=predicted_args_raw,
            call_present=call_present,
            format_valid=format_valid,
            parser_message=parser_message,
        )

    def _predict_observation_stage(self, task: TaskSample, tool_registry: list[ToolSpec]):
        registry = RegistryBuilder.build_from_specs(tool_registry)
        stage1_tool_raw, stage1_args_raw = self._mock_stage1_invocation(task, tool_registry)
        stage1_resolved = registry.resolve_tool_name(stage1_tool_raw)
        stage1_normalized_args = (
            registry.normalize_arguments(stage1_resolved, stage1_args_raw)
            if stage1_resolved
            else {}
        )
        runtime_feedback = run_perturbed_execution(
            task,
            stage1_resolved,
            stage1_normalized_args,
            executor=self.executor,
        )
        public_runtime_feedback = runtime_feedback.as_public_dict()
        observation_target = build_observation_target(stage1_resolved, runtime_feedback.clean_result)
        final_result, final_format_valid, parser_message = self._mock_observation_result(task, observation_target)
        return self.default_parsed(
            raw_response={
                "stage1": {"tool_name": stage1_tool_raw, "arguments": stage1_args_raw},
                "runtime_feedback": public_runtime_feedback,
                "stage2": {"final_result": final_result},
            },
            predicted_tool_raw=stage1_tool_raw,
            predicted_args_raw=stage1_args_raw,
            call_present=True,
            format_valid=True,
            parser_message="mock stage1 tool call",
            auxiliary={
                "stage1_predicted_tool_raw": stage1_tool_raw,
                "stage1_predicted_args_raw": stage1_args_raw,
                "stage1_predicted_tool_resolved": stage1_resolved,
                "stage1_normalized_arguments": stage1_normalized_args,
                "runtime_execution_success": runtime_feedback.execution_success,
                "runtime_status": runtime_feedback.status,
                "runtime_feedback": runtime_feedback.content,
                "runtime_note": runtime_feedback.note,
                "runtime_clean_result": runtime_feedback.clean_result,
                "observation_target": observation_target,
                "observation_stage2_format_valid": int(final_format_valid),
                "observation_stage2_result": final_result,
                "observation_stage2_parser_message": parser_message,
            },
        )

    def _predict_execution_stage(self, task: TaskSample, tool_registry: list[ToolSpec]):
        registry = RegistryBuilder.build_from_specs(tool_registry)
        stage1_tool_raw, stage1_args_raw = self._mock_stage1_invocation(task, tool_registry)
        stage1_resolved = registry.resolve_tool_name(stage1_tool_raw)
        stage1_normalized_args = (
            registry.normalize_arguments(stage1_resolved, stage1_args_raw)
            if stage1_resolved
            else {}
        )
        runtime_feedback = run_perturbed_execution(
            task,
            stage1_resolved,
            stage1_normalized_args,
            executor=self.executor,
        )
        public_runtime_feedback = runtime_feedback.as_public_dict()
        execution_decision, final_format_valid, parser_message = self._mock_execution_decision(
            task,
            runtime_feedback.content,
            runtime_feedback.clean_result,
        )
        return self.default_parsed(
            raw_response={
                "stage1": {"tool_name": stage1_tool_raw, "arguments": stage1_args_raw},
                "runtime_feedback": public_runtime_feedback,
                "stage2": execution_decision,
            },
            predicted_tool_raw=stage1_tool_raw,
            predicted_args_raw=stage1_args_raw,
            call_present=True,
            format_valid=True,
            parser_message="mock stage1 tool call",
            auxiliary={
                "stage1_predicted_tool_raw": stage1_tool_raw,
                "stage1_predicted_args_raw": stage1_args_raw,
                "stage1_predicted_tool_resolved": stage1_resolved,
                "stage1_normalized_arguments": stage1_normalized_args,
                "runtime_execution_success": runtime_feedback.execution_success,
                "runtime_status": runtime_feedback.status,
                "runtime_feedback": runtime_feedback.content,
                "runtime_note": runtime_feedback.note,
                "runtime_clean_result": runtime_feedback.clean_result,
                "execution_stage2_format_valid": int(final_format_valid),
                "execution_stage2_decision": execution_decision if final_format_valid else None,
                "execution_stage2_parser_message": parser_message,
            },
        )

    def _mock_stage1_invocation(self, task: TaskSample, tool_registry: list[ToolSpec]) -> tuple[str, dict[str, Any]]:
        predicted_tool_raw = task.exposed_tool_name
        predicted_args_raw = deepcopy(task.expected_arguments)

        if task.has_family(BenchmarkFamily.INTERFACE_EXPOSURE.value):
            if any(name in task.operators_applied for name in {"family_collision_name", "cross_tool_description_blend"}):
                predicted_tool_raw = self._sibling_tool_name(task, tool_registry) or task.clean_tool_name
            elif any(name in task.operators_applied for name in {"semi_opaque_name", "description_abstraction"}) and task.severity.value in {"medium", "heavy"}:
                predicted_tool_raw = task.clean_tool_name
            elif any(name in task.operators_applied for name in {"light_description_rewrite", "sibling_similarity_hint", "alias_pruning"}):
                predicted_tool_raw = self._sibling_tool_name(task, tool_registry) or predicted_tool_raw

        user_intent_subtype = task.component_subtype(BenchmarkFamily.USER_INTENT.value)
        if user_intent_subtype == "B2 Underspecification":
            first_key = sorted(task.expected_arguments)[0]
            predicted_args_raw.pop(first_key, None)
        elif user_intent_subtype in {"B3 Distraction", "B4 Conflict"}:
            first_key = sorted(task.expected_arguments)[0]
            predicted_args_raw[first_key] = self._wrong_value(predicted_args_raw[first_key])
        elif user_intent_subtype == "B1 Indirectness" and "cross_sentence_scatter" in task.operators_applied:
            predicted_tool_raw = self._sibling_tool_name(task, tool_registry) or task.clean_tool_name

        return predicted_tool_raw, predicted_args_raw

    def _mock_observation_result(self, task: TaskSample, observation_target: Any) -> tuple[Any, bool, str]:
        final_result = deepcopy(observation_target)
        format_valid = True
        parser_message = "mock observation final result"
        scenario_rank = (sum(ord(char) for char in task.canonical_tool_id) + int(task.metadata.get("bucket_index", 0))) % 5
        subtype = task.component_subtype(BenchmarkFamily.OBSERVATION.value) or task.subtype

        if subtype == "C1 Return Structure Noise":
            if "field_reorder" in task.operators_applied:
                final_result = self._reorder_result(final_result)
                parser_message = "mock reordered the final observation fields"
            if "camel_snake_shift" in task.operators_applied:
                final_result = self._camelize_result(final_result)
                parser_message = "mock emitted camelCase final observation fields"
            if "key_rename" in task.operators_applied:
                parser_message = "mock key rename kept the result recoverable but less direct"
            if "nesting_shift" in task.operators_applied:
                if scenario_rank % 2 == 1:
                    format_valid = False
                    parser_message = "mock structure noise broke the final observation json"
                else:
                    parser_message = "mock nested structure stayed recoverable on the easier scenario"
            if "wrapper_insertion" in task.operators_applied:
                if scenario_rank != 0:
                    format_valid = False
                    parser_message = "mock wrapper insertion broke the final observation json"
                else:
                    parser_message = "mock wrapper insertion remained barely recoverable on the easiest scenario"
        elif subtype == "C2 Return Evidence Loss":
            if "dropped_metadata" in task.operators_applied:
                parser_message = "mock metadata loss kept the result recoverable"
            if "sparse_result" in task.operators_applied:
                final_result = self._sparsify_result(final_result)
                parser_message = "mock sparse evidence preserved only part of the result"
            if "missing_field" in task.operators_applied:
                final_result = self._drop_first_result_field(final_result)
                parser_message = "mock evidence loss removed one salient result field"
            if "partial_truncation" in task.operators_applied:
                final_result = self._truncate_result(final_result)
                parser_message = "mock partial evidence produced a truncated result"
        elif subtype == "C3 Return Evidence Corruption":
            if "stale_value" in task.operators_applied:
                final_result = self._corrupt_secondary_result_value(final_result, stale=True)
                parser_message = "mock stale evidence shifted one result value"
            if "supporting_conflict" in task.operators_applied:
                final_result = self._corrupt_two_result_values(final_result)
                parser_message = "mock conflicting supporting evidence changed multiple result values"
            if "wrong_fact_field" in task.operators_applied:
                final_result = self._corrupt_first_result_value(final_result, stale=False)
                parser_message = "mock corrupted evidence changed one result value"
            if "unit_mismatch" in task.operators_applied:
                final_result = self._corrupt_unit_like_result_field(final_result)
                parser_message = "mock unit mismatch changed a unit-bearing result field"
            if "inconsistent_pair" in task.operators_applied:
                final_result = self._corrupt_two_result_values(final_result)
                parser_message = "mock conflicting evidence changed two result values"
        elif subtype == "C4 Return Evidence Competition":
            if "low_utility_redundancy_soft" in task.operators_applied:
                parser_message = "mock mild redundant evidence added bookkeeping but stayed recoverable"
            if "low_utility_redundancy" in task.operators_applied:
                parser_message = "mock redundant evidence left the recovered result intact"
            if "distractor_candidate_soft" in task.operators_applied:
                if scenario_rank == 2:
                    final_result = self._corrupt_secondary_result_value(final_result, stale=True)
                    parser_message = "mock softly competing evidence occasionally nudged a secondary result field off target"
                else:
                    parser_message = "mock softly competing evidence stayed recoverable on easier scenarios"
            if "distractor_candidate" in task.operators_applied:
                if scenario_rank % 2 == 1:
                    final_result = self._corrupt_secondary_result_value(final_result, stale=True)
                    parser_message = "mock competing evidence nudged a secondary result field off target"
                else:
                    parser_message = "mock competing evidence stayed recoverable on the easier scenario"
            if "misleading_note" in task.operators_applied:
                if scenario_rank != 0:
                    final_result = self._corrupt_two_result_values(final_result)
                    parser_message = "mock misleading note reinforced the competing result path"
                else:
                    parser_message = "mock misleading note was not strong enough to flip the easiest scenario"
            if "spurious_auxiliary_field" in task.operators_applied:
                final_result = self._corrupt_two_result_values(final_result)
                parser_message = "mock auxiliary evidence pushed the result off target"

        return final_result, format_valid, parser_message

    def _mock_execution_decision(self, task: TaskSample, runtime_content: Any, clean_result: Any) -> tuple[dict[str, Any], bool, str]:
        decision = {
            "status": "success",
            "retryable": False,
            "failure_type": None,
            "final_result": deepcopy(clean_result),
        }
        parser_message = "mock execution decision"
        format_valid = True
        subtype = task.component_subtype(BenchmarkFamily.EXECUTION_ENVIRONMENT.value) or task.subtype

        if subtype == "D1 Availability Failure":
            if "cached_availability_fallback" in task.operators_applied:
                decision.update({"status": "success", "retryable": False, "failure_type": None, "final_result": deepcopy(clean_result)})
                parser_message = "mock stripped degraded availability metadata"
            if "service_unavailable_with_stale_cache" in task.operators_applied:
                partial = deepcopy(runtime_content.get("fragment")) if isinstance(runtime_content, dict) else None
                decision.update({"status": "partial", "retryable": True, "failure_type": "service_unavailable", "final_result": partial})
                parser_message = "mock recovered a partial result from service unavailability"
            if "timeout_after_partial_stream" in task.operators_applied:
                partial = None
                if isinstance(runtime_content, dict) and isinstance(runtime_content.get("raw"), dict):
                    partial = deepcopy(runtime_content["raw"].get("fragment"))
                decision.update({"status": "partial", "retryable": True, "failure_type": "timeout", "final_result": partial})
                parser_message = "mock recovered a partial result from timed-out stream frames"
            if "empty_response" in task.operators_applied:
                decision.update({"status": "error", "retryable": True, "failure_type": "empty_response", "final_result": None})
                parser_message = "mock diagnosed an empty response as retryable"
            if "service_unavailable" in task.operators_applied:
                decision.update({"status": "error", "retryable": True, "failure_type": "service_unavailable", "final_result": None})
                parser_message = "mock diagnosed service unavailability as retryable"
            if "timeout" in task.operators_applied:
                decision.update({"status": "error", "retryable": True, "failure_type": "timeout", "final_result": None})
                parser_message = "mock diagnosed timeout as retryable"
            if "auth_denied" in task.operators_applied:
                decision.update({"status": "error", "retryable": False, "failure_type": "auth_denied", "final_result": None})
                parser_message = "mock diagnosed authorization failure as non-retryable"
        elif subtype == "D2 Runtime Variability":
            if "jittered_metadata" in task.operators_applied:
                decision.update({"status": "success", "retryable": False, "failure_type": None, "final_result": deepcopy(clean_result)})
                parser_message = "mock ignored jittered metadata and kept the clean result"
            if "delayed_fresh_partial_with_stale_full" in task.operators_applied:
                partial = self._runtime_fragment(runtime_content)
                failure_type = "runtime_variability" if "contaminated_flaky_attempts" in task.operators_applied else "shape_mismatch" if "flaky_shape_conflicting_attempts" in task.operators_applied else "partial_response"
                decision.update({"status": "partial", "retryable": True, "failure_type": failure_type, "final_result": deepcopy(partial)})
                parser_message = "mock selected the fresh partial result over stale runtime attempts"
            if "unstable_optional_field" in task.operators_applied:
                decision.update({"status": "success", "retryable": False, "failure_type": None, "final_result": deepcopy(clean_result)})
                parser_message = "mock ignored unstable optional fields and kept the clean result"
            if "delayed_partial_response" in task.operators_applied:
                partial = None
                if isinstance(runtime_content, dict):
                    partial = runtime_content.get("fragment")
                    compat_payload = runtime_content.get("compat_envelope")
                    if partial is None and isinstance(compat_payload, dict):
                        partial = compat_payload.get("data")
                decision.update({"status": "partial", "retryable": True, "failure_type": "partial_response", "final_result": deepcopy(partial)})
                parser_message = "mock diagnosed delayed partial output as a retryable partial result"
            if "flaky_response_shape" in task.operators_applied:
                if "delayed_partial_response" in task.operators_applied:
                    decision.update({"status": "partial", "retryable": True, "failure_type": "shape_mismatch", "final_result": deepcopy(partial)})
                    parser_message = "mock recovered a harder partial result from a compatibility envelope"
                else:
                    decision.update({"status": "error", "retryable": True, "failure_type": "shape_mismatch", "final_result": None})
                    parser_message = "mock diagnosed an unstable envelope without a usable fragment"
        elif subtype == "D3 Failure Feedback Quality":
            if "vague_retry_hint" in task.operators_applied:
                decision.update({"status": "error", "retryable": True, "failure_type": "vague_error", "final_result": None})
                parser_message = "mock used the weak retry hint in vague feedback"
            if "wrong_failure_label_with_evidence" in task.operators_applied:
                decision.update({"status": "error", "retryable": True, "failure_type": "wrong_failure_label", "final_result": None})
                parser_message = "mock detected the contradictory failure label evidence"
            if "hidden_partial_wrong_label" in task.operators_applied:
                partial = deepcopy(self._runtime_fragment(runtime_content))
                decision.update({"status": "partial", "retryable": True, "failure_type": "wrong_failure_label", "final_result": partial})
                parser_message = "mock recovered a hidden partial fragment under the wrong label"
            if "vague_error" in task.operators_applied:
                decision.update({"status": "error", "retryable": True, "failure_type": "vague_error", "final_result": None})
                parser_message = "mock treated the vague runtime error as retryable"
            if "wrong_failure_label" in task.operators_applied and "partial_success_no_status" not in task.operators_applied:
                decision.update({"status": "error", "retryable": True, "failure_type": "wrong_failure_label", "final_result": None})
                parser_message = "mock ignored the surface label and kept the decision retryable"
            if "misleading_error" in task.operators_applied:
                decision.update({"status": "error", "retryable": False, "failure_type": "misleading_error", "final_result": None})
                parser_message = "mock diagnosed misleading runtime feedback"
            if "partial_success_no_status" in task.operators_applied:
                partial = None
                if isinstance(runtime_content, dict) and isinstance(runtime_content.get("raw"), dict):
                    partial = deepcopy(runtime_content["raw"].get("fragment"))
                elif isinstance(runtime_content, dict):
                    partial = deepcopy(runtime_content.get("fragment"))
                decision.update({"status": "partial", "retryable": True, "failure_type": "wrong_failure_label", "final_result": partial})
                parser_message = "mock recovered a hidden partial fragment despite the wrong label"

        return decision, format_valid, parser_message

    @staticmethod
    def _runtime_fragment(runtime_content: Any) -> Any:
        if not isinstance(runtime_content, dict):
            return None
        direct = runtime_content.get("fragment")
        if direct is not None:
            return direct
        raw = runtime_content.get("raw")
        if isinstance(raw, dict):
            fragment = raw.get("fragment")
            if fragment is not None:
                return fragment
            envelope = raw.get("envelope")
            if isinstance(envelope, dict) and envelope.get("data") is not None:
                return envelope.get("data")
        return None

    @classmethod
    def _drop_first_result_field(cls, value: Any) -> Any:
        if not isinstance(value, dict) or not value:
            return value
        mutated = dict(value)
        mutated.pop(next(iter(mutated)))
        return mutated

    @classmethod
    def _truncate_result(cls, value: Any) -> Any:
        if isinstance(value, dict):
            mutated = dict(value)
            if not mutated:
                return mutated
            first_key = next(iter(mutated))
            mutated[first_key] = cls._truncate_result(mutated[first_key])
            return mutated
        if isinstance(value, list):
            return value[:1]
        if isinstance(value, str):
            return value[:8]
        return value

    @classmethod
    def _sparsify_result(cls, value: Any) -> Any:
        if not isinstance(value, dict) or not value:
            return value
        first_key = next(iter(value))
        return {first_key: deepcopy(value[first_key])}

    @classmethod
    def _corrupt_first_result_value(cls, value: Any, *, stale: bool) -> Any:
        if not isinstance(value, dict) or not value:
            return value
        mutated = dict(value)
        first_key = next(iter(mutated))
        mutated[first_key] = cls._stale_value(mutated[first_key]) if stale else cls._wrong_value(mutated[first_key])
        return mutated

    @classmethod
    def _corrupt_secondary_result_value(cls, value: Any, *, stale: bool) -> Any:
        if not isinstance(value, dict) or not value:
            return value
        mutated = dict(value)
        keys = list(mutated)
        target_key = keys[1] if len(keys) > 1 else keys[0]
        mutated[target_key] = cls._stale_value(mutated[target_key]) if stale else cls._wrong_value(mutated[target_key])
        return mutated

    @classmethod
    def _corrupt_two_result_values(cls, value: Any) -> Any:
        if not isinstance(value, dict) or not value:
            return value
        mutated = dict(value)
        for key in list(mutated.keys())[:2]:
            mutated[key] = cls._wrong_value(mutated[key])
        return mutated

    @classmethod
    def _corrupt_unit_like_result_field(cls, value: Any) -> Any:
        if not isinstance(value, dict) or not value:
            return value
        mutated = dict(value)
        for key in ("scale", "target", "source", "profile"):
            if key in mutated:
                mutated[key] = cls._wrong_value(mutated[key])
                return mutated
        first_key = next(iter(mutated))
        mutated[first_key] = cls._wrong_value(mutated[first_key])
        return mutated

    @classmethod
    def _camelize_result(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {cls._camelize_key(key): cls._camelize_result(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._camelize_result(item) for item in value]
        return value

    @classmethod
    def _reorder_result(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: cls._reorder_result(value[key]) for key in reversed(list(value.keys()))}
        if isinstance(value, list):
            return [cls._reorder_result(item) for item in reversed(value)]
        return value

    @staticmethod
    def _camelize_key(value: str) -> str:
        parts = value.split("_")
        if len(parts) == 1:
            return value
        return parts[0] + "".join(part.capitalize() for part in parts[1:])

    @staticmethod
    def _wrong_value(value: object) -> object:
        if isinstance(value, str):
            return f"wrong_{value}"
        if isinstance(value, (int, float)):
            return value + 1
        if isinstance(value, list):
            reversed_list = list(reversed(value))
            return reversed_list if reversed_list != list(value) else list(value) + ["wrong_value"]
        return "wrong_value"

    @staticmethod
    def _stale_value(value: object) -> object:
        if isinstance(value, str):
            return "tomorrow" if value == "today" else "today" if value == "tomorrow" else f"stale_{value}"
        if isinstance(value, (int, float)):
            return value - 1
        if isinstance(value, list):
            return value[1:] + value[:1] if len(value) > 1 else list(value) + ["stale_value"]
        return value

    @staticmethod
    def _preferred_arg_key(arguments: dict[str, Any]) -> str:
        return sorted(arguments)[0]

    @staticmethod
    def _unit_like_key(arguments: dict[str, Any]) -> str:
        for key in ("scale", "target", "source", "profile"):
            if key in arguments:
                return key
        return sorted(arguments)[0]

    @staticmethod
    def _sibling_tool_name(task: TaskSample, tool_registry: list[ToolSpec]) -> str | None:
        for group in TOOL_GROUPS.values():
            if task.canonical_tool_id in group:
                siblings = [tool_id for tool_id in group if tool_id != task.canonical_tool_id]
                if not siblings:
                    return None
                pick = siblings[int(task.metadata.get("scenario_index", 0)) % len(siblings)]
                for tool in tool_registry:
                    if tool.canonical_tool_id == pick:
                        return tool.exposed_name
        return None
