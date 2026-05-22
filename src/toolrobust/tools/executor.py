"""Tool execution layer for benchmark v2."""

from __future__ import annotations

from typing import Any

from toolrobust.tools.base import ToolExecutionError
from toolrobust.tools.local_tools import get_default_local_tools
from toolrobust.tools.registry import ToolRegistry


class ToolExecutor:
    """Execute deterministic local tools by canonical id."""

    def __init__(self) -> None:
        self._tools = {tool.spec.canonical_tool_id: tool for tool in get_default_local_tools()}

    def execute(self, canonical_tool_id: str, args: dict[str, Any]) -> Any:
        tool = self._tools.get(canonical_tool_id)
        if tool is None:
            raise ToolExecutionError(f"Unknown canonical tool: {canonical_tool_id}")
        return tool.execute(args)

    def execute_from_registry(self, registry: ToolRegistry, predicted_tool_name: str | None, predicted_args: dict[str, Any]) -> Any:
        canonical_tool_id = registry.resolve_tool_name(predicted_tool_name)
        if canonical_tool_id is None:
            raise ToolExecutionError(f"Unable to resolve tool name: {predicted_tool_name}")
        normalized_args = registry.normalize_arguments(canonical_tool_id, predicted_args)
        return self.execute(canonical_tool_id, normalized_args)
