"""Registry construction and name-resolution for benchmark v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from toolrobust.domain.schemas import ToolSpec
from toolrobust.tools.local_tools import get_default_local_tools
from toolrobust.tools.schema_utils import validate_tool_registry


@dataclass(slots=True)
class ToolRegistry:
    """Registry with canonical identity and exposed interface mappings."""

    tools_by_canonical: dict[str, ToolSpec]
    name_to_canonical: dict[str, str]

    def list_tools(self) -> list[ToolSpec]:
        return list(self.tools_by_canonical.values())

    def get_tool(self, canonical_tool_id: str) -> ToolSpec | None:
        return self.tools_by_canonical.get(canonical_tool_id)

    def resolve_tool_name(self, predicted_tool_name: str | None) -> str | None:
        if not predicted_tool_name:
            return None
        return self.name_to_canonical.get(predicted_tool_name)

    def normalize_arguments(self, canonical_tool_id: str, predicted_args: dict[str, Any]) -> dict[str, Any]:
        tool = self.get_tool(canonical_tool_id)
        if tool is None:
            return {}
        normalized: dict[str, Any] = {}
        for key, value in predicted_args.items():
            if key in tool.parameter_name_map:
                normalized[tool.parameter_name_map[key]] = value
            else:
                normalized[key] = value
        return normalized

    def required_canonical_parameters(self, canonical_tool_id: str) -> list[str]:
        tool = self.get_tool(canonical_tool_id)
        if tool is None:
            return []
        return [tool.parameter_name_map.get(name, name) for name in tool.parameters.required]


class RegistryBuilder:
    """Build registries from normalized tool specs."""

    def build_clean_registry(
        self,
        enabled_tools: list[str] | None = None,
        *,
        include_similar_tool_groups: bool = False,
    ) -> ToolRegistry:
        return self.build_from_specs(
            validate_tool_registry(
                tool.spec
                for tool in get_default_local_tools(
                    enabled_tools,
                    include_similar_tool_groups=include_similar_tool_groups,
                )
            )
        )

    @staticmethod
    def build_from_specs(tool_specs: list[ToolSpec]) -> ToolRegistry:
        tools_by_canonical = {tool.canonical_tool_id: tool for tool in tool_specs}
        name_to_canonical: dict[str, str] = {}
        for tool in tool_specs:
            for candidate in {tool.canonical_tool_id, tool.clean_tool_name, tool.exposed_name, *tool.aliases}:
                name_to_canonical[candidate] = tool.canonical_tool_id
        return ToolRegistry(tools_by_canonical=tools_by_canonical, name_to_canonical=name_to_canonical)
