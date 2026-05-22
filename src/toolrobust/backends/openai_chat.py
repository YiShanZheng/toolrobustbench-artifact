"""OpenAI chat-completions backend using the official SDK."""

from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from toolrobust.backends.base import BackendBase
from toolrobust.domain.enums import BenchmarkFamily
from toolrobust.domain.schemas import BackendResponse, TaskSample, ToolSpec
from toolrobust.evaluation.parser import (
    parse_backend_json_content,
    parse_execution_decision_content,
    parse_backend_payload,
    parse_observation_result_content,
)
from toolrobust.execution.runtime_feedback import build_observation_target, run_perturbed_execution
from toolrobust.tools.executor import ToolExecutor
from toolrobust.tools.registry import RegistryBuilder
from toolrobust.tools.schema_utils import validate_tool_registry


OBSERVATION_STAGE2_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "observation_stage2_result",
        "strict": False,
        "schema": {
            "type": "object",
            "properties": {
                "final_result": {
                    "anyOf": [
                        {"type": "object", "additionalProperties": True},
                        {"type": "null"},
                    ]
                }
            },
            "required": ["final_result"],
            "additionalProperties": False,
        },
    },
}

EXECUTION_STAGE2_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "execution_stage2_decision",
        "strict": False,
        "schema": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["success", "partial", "error"]},
                "retryable": {"type": "boolean"},
                "failure_type": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "null"},
                    ]
                },
                "final_result": {
                    "anyOf": [
                        {"type": "object", "additionalProperties": True},
                        {"type": "null"},
                    ]
                },
            },
            "required": ["status", "retryable", "failure_type", "final_result"],
            "additionalProperties": False,
        },
    },
}


class OpenAIChatBackend(BackendBase):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.executor = ToolExecutor()

    @property
    def name(self) -> str:
        return "openai"

    def predict(self, task: TaskSample, tool_registry: list[ToolSpec]) -> BackendResponse:
        api_key = self._resolve_env(self.config.api_key_env, "OPENAI_API_KEY")
        if not api_key:
            expected_env = self.config.api_key_env or "OPENAI_API_KEY"
            raise RuntimeError(f"{expected_env} is not configured")
        normalized_registry = validate_tool_registry(tool_registry)
        client = self._build_client(api_key)
        if task.family == BenchmarkFamily.EXECUTION_ENVIRONMENT.value:
            return self._predict_execution_stage(task, normalized_registry, client)
        if task.has_family(BenchmarkFamily.EXECUTION_ENVIRONMENT.value):
            return self._predict_execution_stage(task, normalized_registry, client)
        if task.has_family(BenchmarkFamily.OBSERVATION.value):
            return self._predict_observation_stage(task, normalized_registry, client)
        return self._predict_single_stage(task, normalized_registry, client)

    def _build_client(self, api_key: str) -> OpenAI:
        return OpenAI(
            api_key=api_key,
            base_url=self._resolve_base_url(),
            timeout=self.config.request_timeout_sec,
            max_retries=self.config.request_max_retries if self.config.request_max_retries is not None else 0,
        )

    def _resolve_base_url(self) -> str | None:
        return self._resolve_env(self.config.base_url_env, "OPENAI_BASE_URL") or self.config.base_url or None

    @staticmethod
    def _resolve_env(*names: str | None) -> str:
        for name in names:
            if not name:
                continue
            value = os.getenv(name)
            if value:
                return value
        return ""

    def _predict_single_stage(
        self,
        task: TaskSample,
        tool_registry: list[ToolSpec],
        client: OpenAI,
    ) -> BackendResponse:
        completion = client.chat.completions.create(
            **self._create_tool_call_kwargs(
                messages=[
                    {"role": "system", "content": "You are a precise single-step tool caller."},
                    {"role": "user", "content": task.user_instruction},
                ],
                tool_registry=tool_registry,
            )
        )
        choice = completion.choices[0].message
        raw = completion.model_dump(mode="json")
        parsed = self._parse_tool_call(choice)
        return BackendResponse(
            backend_name=self.name,
            model_name=self.config.model_name,
            raw_response=raw,
            parsed=parsed,
        )

    def _predict_observation_stage(
        self,
        task: TaskSample,
        tool_registry: list[ToolSpec],
        client: OpenAI,
    ) -> BackendResponse:
        stage1_completion = client.chat.completions.create(
            **self._create_tool_call_kwargs(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise tool caller. First call exactly one tool for the user request. "
                            "After runtime feedback appears, you will return a final JSON decision."
                        ),
                    },
                    {"role": "user", "content": task.user_instruction},
                ],
                tool_registry=tool_registry,
            )
        )
        stage1_choice = stage1_completion.choices[0].message
        stage1_raw = stage1_completion.model_dump(mode="json")
        stage1_parsed = self._parse_tool_call(stage1_choice)
        registry = RegistryBuilder.build_from_specs(tool_registry)
        stage1_resolved = registry.resolve_tool_name(stage1_parsed.predicted_tool_raw)
        stage1_normalized_args = (
            registry.normalize_arguments(stage1_resolved, stage1_parsed.predicted_args_raw)
            if stage1_resolved
            else {}
        )

        auxiliary: dict[str, Any] = {
            "stage1_predicted_tool_raw": stage1_parsed.predicted_tool_raw,
            "stage1_predicted_args_raw": stage1_parsed.predicted_args_raw,
            "stage1_predicted_tool_resolved": stage1_resolved,
            "stage1_normalized_arguments": stage1_normalized_args,
        }
        if not stage1_parsed.call_present or not stage1_parsed.format_valid:
            return BackendResponse(
                backend_name=self.name,
                model_name=self.config.model_name,
                raw_response={"stage1": stage1_raw, "stage2": None},
                parsed=stage1_parsed,
                auxiliary=auxiliary,
            )

        runtime_feedback = run_perturbed_execution(
            task,
            stage1_resolved,
            stage1_normalized_args,
            executor=self.executor,
        )
        public_runtime_feedback = runtime_feedback.as_public_dict()
        observation_target = build_observation_target(stage1_resolved, runtime_feedback.clean_result)
        auxiliary.update(
            {
                "runtime_execution_success": runtime_feedback.execution_success,
                "runtime_status": runtime_feedback.status,
                "runtime_feedback": runtime_feedback.content,
                "runtime_note": runtime_feedback.note,
                "runtime_clean_result": runtime_feedback.clean_result,
                "observation_target": observation_target,
            }
        )

        stage2_completion = self._create_stage2_completion(
            client,
            response_format=OBSERVATION_STAGE2_RESPONSE_FORMAT,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You recover the clean tool result from noisy runtime feedback. "
                        "Return only JSON. Prefer the shape {\"final_result\": <object-or-null>}. "
                        "Do not emit tool_name or arguments."
                    ),
                },
                {
                    "role": "user",
                    "content": self._build_observation_prompt(
                        stage1_parsed.predicted_tool_raw,
                        stage1_normalized_args,
                        public_runtime_feedback,
                        observation_target,
                    ),
                },
            ],
        )
        stage2_choice = stage2_completion.choices[0].message
        stage2_raw = stage2_completion.model_dump(mode="json")
        stage2_format_valid, stage2_result, stage2_parser_message = parse_observation_result_content(stage2_choice.content)
        auxiliary.update(
            {
                "observation_stage2_format_valid": int(stage2_format_valid),
                "observation_stage2_result": stage2_result,
                "observation_stage2_parser_message": stage2_parser_message,
            }
        )
        return BackendResponse(
            backend_name=self.name,
            model_name=self.config.model_name,
            raw_response={"stage1": stage1_raw, "stage2": stage2_raw},
            parsed=stage1_parsed,
            auxiliary=auxiliary,
        )

    def _predict_execution_stage(
        self,
        task: TaskSample,
        tool_registry: list[ToolSpec],
        client: OpenAI,
    ) -> BackendResponse:
        stage1_completion = client.chat.completions.create(
            **self._create_tool_call_kwargs(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise tool caller. Call exactly one tool for the user request. "
                            "Do not explain your choice in prose."
                        ),
                    },
                    {"role": "user", "content": task.user_instruction},
                ],
                tool_registry=tool_registry,
            )
        )
        stage1_choice = stage1_completion.choices[0].message
        stage1_raw = {"attempt_1": stage1_completion.model_dump(mode="json")}
        stage1_parsed = self._parse_tool_call(stage1_choice)
        if not stage1_parsed.call_present or not stage1_parsed.format_valid:
            retry_completion = client.chat.completions.create(
                **self._create_tool_call_kwargs(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You must respond with exactly one tool call. "
                                "Do not answer with natural language. Do not skip the tool call."
                            ),
                        },
                        {"role": "user", "content": task.user_instruction},
                    ],
                    tool_registry=tool_registry,
                )
            )
            retry_choice = retry_completion.choices[0].message
            retry_parsed = self._parse_tool_call(retry_choice)
            stage1_raw["attempt_2"] = retry_completion.model_dump(mode="json")
            if retry_parsed.call_present and retry_parsed.format_valid:
                stage1_parsed = retry_parsed

        registry = RegistryBuilder.build_from_specs(tool_registry)
        stage1_resolved = registry.resolve_tool_name(stage1_parsed.predicted_tool_raw)
        stage1_normalized_args = (
            registry.normalize_arguments(stage1_resolved, stage1_parsed.predicted_args_raw)
            if stage1_resolved
            else {}
        )

        auxiliary: dict[str, Any] = {
            "stage1_predicted_tool_raw": stage1_parsed.predicted_tool_raw,
            "stage1_predicted_args_raw": stage1_parsed.predicted_args_raw,
            "stage1_predicted_tool_resolved": stage1_resolved,
            "stage1_normalized_arguments": stage1_normalized_args,
        }
        if not stage1_parsed.call_present or not stage1_parsed.format_valid:
            return BackendResponse(
                backend_name=self.name,
                model_name=self.config.model_name,
                raw_response={"stage1": stage1_raw, "stage2": None},
                parsed=stage1_parsed,
                auxiliary=auxiliary,
            )

        runtime_feedback = run_perturbed_execution(
            task,
            stage1_resolved,
            stage1_normalized_args,
            executor=self.executor,
        )
        public_runtime_feedback = runtime_feedback.as_public_dict()
        auxiliary.update(
            {
                "runtime_execution_success": runtime_feedback.execution_success,
                "runtime_status": runtime_feedback.status,
                "runtime_feedback": runtime_feedback.content,
                "runtime_note": runtime_feedback.note,
                "runtime_clean_result": runtime_feedback.clean_result,
            }
        )

        stage2_completion = self._create_stage2_completion(
            client,
            response_format=EXECUTION_STAGE2_RESPONSE_FORMAT,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You evaluate runtime feedback after a tool call. "
                        "Return only JSON with keys `status`, `retryable`, `failure_type`, and `final_result`. "
                        "`status` must be one of `success`, `partial`, or `error`."
                    ),
                },
                {
                    "role": "user",
                    "content": self._build_execution_prompt(
                        task,
                        stage1_parsed.predicted_tool_raw,
                        stage1_normalized_args,
                        public_runtime_feedback,
                    ),
                },
            ],
        )
        stage2_choice = stage2_completion.choices[0].message
        stage2_raw = stage2_completion.model_dump(mode="json")
        stage2_format_valid, stage2_decision, stage2_parser_message = parse_execution_decision_content(stage2_choice.content)
        auxiliary.update(
            {
                "execution_stage2_format_valid": int(stage2_format_valid),
                "execution_stage2_decision": stage2_decision,
                "execution_stage2_parser_message": stage2_parser_message,
            }
        )
        return BackendResponse(
            backend_name=self.name,
            model_name=self.config.model_name,
            raw_response={"stage1": stage1_raw, "stage2": stage2_raw},
            parsed=stage1_parsed,
            auxiliary=auxiliary,
        )

    @staticmethod
    def _build_observation_prompt(
        stage1_tool_name: str | None,
        stage1_args: dict[str, Any],
        runtime_feedback: dict[str, Any],
        observation_target: Any,
    ) -> str:
        prompt_feedback = {
            "execution_success": runtime_feedback.get("execution_success"),
            "status": runtime_feedback.get("status"),
            "content": runtime_feedback.get("content"),
        }
        invocation_context = {
            "tool_name": stage1_tool_name,
            "argument_keys": sorted(stage1_args),
            "target_complexity": _result_complexity_hint(observation_target),
        }
        return (
            f"Initial invocation context:\n{json.dumps(invocation_context, ensure_ascii=False, sort_keys=True)}\n\n"
            f"Runtime feedback:\n{json.dumps(prompt_feedback, ensure_ascii=False, sort_keys=True)}\n\n"
            "Recover only the clean observation target from the noisy runtime feedback. "
            "Use the invocation context only to understand the tool family. "
            "Do not reconstruct missing values from the original arguments unless the runtime feedback itself supports them. "
            "Return only JSON and prefer the wrapper {\"final_result\": ...}. "
            "Do not emit tool_name or arguments."
        )

    def _create_stage2_completion(
        self,
        client: OpenAI,
        *,
        messages: list[dict[str, str]],
        response_format: dict[str, Any],
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.config.model_name,
            "temperature": self.config.temperature,
            "max_tokens": self._stage2_max_tokens(),
            "messages": messages,
        }
        if self._use_structured_stage2_response():
            kwargs["response_format"] = response_format
        if self.config.extra_body:
            kwargs["extra_body"] = self.config.extra_body
        return client.chat.completions.create(**kwargs)

    def _create_tool_call_kwargs(
        self,
        *,
        messages: list[dict[str, str]],
        tool_registry: list[ToolSpec],
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.config.model_name,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "messages": messages,
            "tools": [self._to_openai_tool(tool) for tool in tool_registry],
        }
        if self.config.tool_choice is not None:
            kwargs["tool_choice"] = self.config.tool_choice
        if self.config.extra_body:
            kwargs["extra_body"] = self.config.extra_body
        return kwargs

    def _use_structured_stage2_response(self) -> bool:
        configured = self.config.structured_stage2_response
        if configured is not None:
            return configured
        return self.config.model_name.lower().startswith("gemini-")

    def _stage2_max_tokens(self) -> int:
        if self.config.stage2_max_tokens is not None:
            return self.config.stage2_max_tokens
        if self.config.model_name.lower().startswith("gemini-"):
            return max(int(self.config.max_tokens), 4096)
        return int(self.config.max_tokens)

    @staticmethod
    def _build_execution_prompt(
        task: TaskSample,
        stage1_tool_name: str | None,
        stage1_args: dict[str, Any],
        runtime_feedback: dict[str, Any],
    ) -> str:
        prompt_feedback = {
            "execution_success": runtime_feedback.get("execution_success"),
            "status": runtime_feedback.get("status"),
            "content": runtime_feedback.get("content"),
        }
        return (
            f"Original user request:\n{task.user_instruction}\n\n"
            f"Initial tool invocation:\n{json.dumps({'tool_name': stage1_tool_name, 'arguments': stage1_args}, ensure_ascii=False, sort_keys=True)}\n\n"
            f"Runtime feedback:\n{json.dumps(prompt_feedback, ensure_ascii=False, sort_keys=True)}\n\n"
            "Guidance:\nJudge the runtime outcome from the evidence in the packet. "
            "Some packets contain degraded transport metadata, stale candidates, or partial fragments; separate usable evidence from diagnostic noise.\n"
            "- Keep the initial tool invocation fixed. Do not switch tools.\n"
            "- If a packet contains a `result` object plus delivery metadata, use only `result` as the final payload.\n"
            "- Use `success` only when the runtime feedback supports a fully usable final payload.\n"
            "- Use `partial` only when the runtime feedback contains a concrete usable subset of the final payload.\n"
            "- Use `error` when the runtime output is unavailable, contradictory, or too malformed to trust.\n"
            "- Choose `retryable` from the runtime condition itself, not from the user's goal.\n"
            "- If a surface error label conflicts with audit or transport evidence, summarize it as a wrong failure label instead of echoing the contradicted label.\n"
            "- If a concrete usable fragment appears under a contradicted failure label, `partial` is allowed and `final_result` should contain only that fragment.\n"
            "- Use `failure_type` as a concise summary of the runtime condition, or `null` for clean success.\n"
            "- Do not invent missing fields in `final_result`.\n\n"
            "Decide whether the runtime outcome is a usable success, a partial result, or an error. "
            "Return only JSON with keys `status`, `retryable`, `failure_type`, and `final_result`. "
            "Use `final_result` for the cleaned usable payload when possible, otherwise null."
        )

    @staticmethod
    def _parse_tool_call(message: Any):
        payload: dict[str, Any]
        if getattr(message, "tool_calls", None):
            first_call = message.tool_calls[0]
            payload = {
                "tool_name": first_call.function.name,
                "arguments": first_call.function.arguments or "{}",
            }
        else:
            payload = {"tool_name": None, "arguments": {}, "content": message.content}
        return parse_backend_payload(payload)

    @staticmethod
    def _to_openai_tool(tool: ToolSpec) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.exposed_name,
                "description": tool.description,
                "parameters": tool.parameters.model_dump(),
            },
        }


def _result_shape_hint(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _result_shape_hint(item) for key, item in value.items()}
    if isinstance(value, list):
        if not value:
            return []
        return [_result_shape_hint(value[0])]
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    return "string"


def _result_complexity_hint(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {
            "kind": "object",
            "field_count": len(value),
            "contains_list": any(isinstance(item, list) for item in value.values()),
        }
    if isinstance(value, list):
        return {"kind": "list", "length_hint": len(value)}
    return {"kind": type(value).__name__}
