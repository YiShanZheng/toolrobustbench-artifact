"""Parse backend outputs into normalized tool invocations."""

from __future__ import annotations

import json
import re
from typing import Any

from toolrobust.domain.schemas import ParsedInvocation


def parse_backend_payload(payload: dict[str, Any]) -> ParsedInvocation:
    """Parse a backend payload without enforcing tool-name correctness.

    Tool-name resolution happens later through the registry so we can separate
    interface-name correctness from canonical-tool correctness.
    """

    predicted_tool = payload.get("tool_name")
    predicted_args = payload.get("arguments", {})
    call_present = predicted_tool is not None
    format_valid = False
    parser_message = ""

    if isinstance(predicted_args, str):
        try:
            predicted_args = json.loads(predicted_args)
        except json.JSONDecodeError:
            parser_message = "invalid json arguments"
            predicted_args = {}
    if call_present and isinstance(predicted_args, dict):
        format_valid = True
        parser_message = parser_message or "parsed backend payload"
    elif call_present:
        parser_message = parser_message or "arguments are not an object"
    else:
        parser_message = "no tool call"

    return ParsedInvocation(
        predicted_tool_raw=predicted_tool,
        predicted_args_raw=predicted_args if isinstance(predicted_args, dict) else {},
        call_present=call_present,
        format_valid=format_valid,
        parser_message=parser_message,
    )


def parse_backend_json_content(content: Any) -> ParsedInvocation:
    text = _flatten_content(content).strip()
    if not text:
        return ParsedInvocation(
            predicted_tool_raw=None,
            predicted_args_raw={},
            call_present=False,
            format_valid=False,
            parser_message="empty final json content",
        )
    payload = _extract_json_object(text)
    if not isinstance(payload, dict):
        return ParsedInvocation(
            predicted_tool_raw=None,
            predicted_args_raw={},
            call_present=False,
            format_valid=False,
            parser_message="final content is not valid json object",
        )
    tool_name = payload.get("tool_name", payload.get("selected_tool"))
    arguments = payload.get("arguments", payload.get("args", {}))
    return parse_backend_payload({"tool_name": tool_name, "arguments": arguments})


def parse_observation_result_content(content: Any) -> tuple[bool, Any, str]:
    text = _flatten_content(content).strip()
    if not text:
        return False, None, "empty observation result content"
    payload = _extract_json_object(text)
    if payload is None:
        return False, None, "observation content is not valid json object"
    if "final_result" in payload:
        return True, payload.get("final_result"), "parsed observation final_result wrapper"
    if set(payload).issubset({"tool_name", "arguments", "selected_tool", "args"}):
        return False, None, "observation content used tool-call schema instead of final_result"
    return True, payload, "parsed bare observation result object"


def parse_execution_decision_content(content: Any) -> tuple[bool, dict[str, Any] | None, str]:
    text = _flatten_content(content).strip()
    if not text:
        return False, None, "empty execution decision content"
    payload = _extract_json_object(text)
    if not isinstance(payload, dict):
        return False, None, "execution decision is not valid json object"
    required_keys = {"status", "retryable", "final_result", "failure_type"}
    if not required_keys.issubset(payload):
        return False, None, "execution decision missing required keys"
    status = payload.get("status")
    retryable = payload.get("retryable")
    if status not in {"success", "partial", "error"}:
        return False, None, "execution decision used invalid status"
    if not isinstance(retryable, bool):
        return False, None, "execution decision retryable must be boolean"
    return True, {
        "status": status,
        "retryable": retryable,
        "final_result": payload.get("final_result"),
        "failure_type": payload.get("failure_type"),
    }, "parsed execution decision json"


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.DOTALL)
    for snippet in (candidate, *_braced_candidates(candidate)):
        try:
            parsed = json.loads(snippet)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _braced_candidates(text: str) -> list[str]:
    matches = re.findall(r"\{.*\}", text, flags=re.DOTALL)
    return matches[:3]
