"""Normalization and validation helpers for tool JSON Schemas."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from toolrobust.domain.schemas import ToolSpec


class ToolSchemaValidationError(ValueError):
    """Raised when a tool schema is invalid or ambiguous."""


@dataclass(slots=True)
class _SchemaContext:
    tool_name: str
    path: str

    def message(self, rule: str) -> str:
        return f"Invalid schema for tool '{self.tool_name}' at '{self.path}': {rule}"


def normalize_tool_schema(tool: ToolSpec) -> ToolSpec:
    normalized = tool.model_copy(deep=True)
    normalized.canonical_tool_id = normalized.canonical_tool_id.strip()
    normalized.exposed_name = normalized.exposed_name.strip()
    normalized.clean_tool_name = normalized.clean_tool_name.strip()
    normalized.description = (normalized.description or "").strip() or f"Call the {normalized.exposed_name} tool."

    parameters = normalized.parameters.model_dump()
    parameters["type"] = "object"
    parameters["properties"] = parameters.get("properties") or {}
    parameters["required"] = list(dict.fromkeys(parameters.get("required") or []))
    parameters["additionalProperties"] = False
    parameters["properties"] = {
        property_name: _normalize_schema_node(property_schema, tool_name=normalized.exposed_name, path=f"parameters.properties.{property_name}")
        for property_name, property_schema in parameters["properties"].items()
    }
    normalized.parameters = normalized.parameters.model_validate(parameters)
    if not normalized.parameter_name_map:
        normalized.parameter_name_map = {name: name for name in normalized.parameters.properties}
    return normalized


def validate_tool_schema(tool: ToolSpec) -> None:
    if not tool.exposed_name.strip():
        raise ToolSchemaValidationError("Invalid schema for tool '<unknown>' at 'function.name': function.name must be non-empty")
    context = _SchemaContext(tool_name=tool.exposed_name, path="function.parameters")
    parameters = tool.parameters.model_dump()
    if parameters.get("type") != "object":
        raise ToolSchemaValidationError(context.message("function.parameters must be an object schema"))
    if not isinstance(parameters.get("properties"), dict) or not parameters["properties"]:
        raise ToolSchemaValidationError(context.message("object schema must declare non-empty properties"))
    required = parameters.get("required") or []
    missing_required = [field_name for field_name in required if field_name not in parameters["properties"]]
    if missing_required:
        raise ToolSchemaValidationError(context.message(f"required fields are missing from properties: {', '.join(missing_required)}"))
    for property_name, property_schema in parameters["properties"].items():
        _validate_schema_node(property_schema, tool_name=tool.exposed_name, path=f"function.parameters.properties.{property_name}")


def validate_tool_registry(tools: Iterable[ToolSpec]) -> list[ToolSpec]:
    normalized_tools: list[ToolSpec] = []
    seen_exposed: set[str] = set()
    for tool in tools:
        normalized = normalize_tool_schema(tool)
        if normalized.exposed_name in seen_exposed:
            raise ToolSchemaValidationError(f"Invalid schema for tool registry: duplicate exposed tool name '{normalized.exposed_name}'")
        validate_tool_schema(normalized)
        seen_exposed.add(normalized.exposed_name)
        normalized_tools.append(normalized)
    return normalized_tools


def _normalize_schema_node(schema: dict[str, Any], *, tool_name: str, path: str) -> dict[str, Any]:
    node = deepcopy(schema)
    if isinstance(node.get("type"), str):
        node["type"] = node["type"].strip().lower()
    if not (node.get("description") or "").strip():
        node["description"] = f"Argument for {tool_name}: {path.split('.')[-1]}."
    if node.get("type") == "object":
        node["properties"] = node.get("properties") or {}
        node["required"] = list(dict.fromkeys(node.get("required") or []))
        node["additionalProperties"] = False
        node["properties"] = {
            property_name: _normalize_schema_node(property_schema, tool_name=tool_name, path=f"{path}.properties.{property_name}")
            for property_name, property_schema in node["properties"].items()
        }
    elif node.get("type") == "array" and isinstance(node.get("items"), dict):
        node["items"] = _normalize_schema_node(node["items"], tool_name=tool_name, path=f"{path}.items")
    elif isinstance(node.get("anyOf"), list):
        node["anyOf"] = [
            _normalize_schema_node(option, tool_name=tool_name, path=f"{path}.anyOf[{index}]")
            for index, option in enumerate(node["anyOf"])
        ]
    if isinstance(node.get("enum"), list):
        node["enum"] = list(dict.fromkeys(node["enum"]))
    return node


def _validate_schema_node(schema: dict[str, Any], *, tool_name: str, path: str) -> None:
    context = _SchemaContext(tool_name=tool_name, path=path)
    node_type = schema.get("type")
    has_any_of = isinstance(schema.get("anyOf"), list) and len(schema["anyOf"]) > 0
    if node_type is None and not has_any_of:
        raise ToolSchemaValidationError(context.message("schema node must declare either 'type' or 'anyOf'"))
    if node_type == "array":
        if not isinstance(schema.get("items"), dict) or not schema["items"]:
            raise ToolSchemaValidationError(context.message("array schema must declare a non-empty 'items' schema"))
        _validate_schema_node(schema["items"], tool_name=tool_name, path=f"{path}.items")
    if node_type == "object":
        properties = schema.get("properties")
        if not isinstance(properties, dict) or not properties:
            raise ToolSchemaValidationError(context.message("object schema must declare non-empty properties"))
        required = schema.get("required") or []
        missing_required = [field_name for field_name in required if field_name not in properties]
        if missing_required:
            raise ToolSchemaValidationError(context.message(f"required fields are missing from properties: {', '.join(missing_required)}"))
        for property_name, property_schema in properties.items():
            _validate_schema_node(property_schema, tool_name=tool_name, path=f"{path}.properties.{property_name}")
    if has_any_of:
        for index, option in enumerate(schema["anyOf"]):
            if not isinstance(option, dict) or not option:
                raise ToolSchemaValidationError(_SchemaContext(tool_name, f"{path}.anyOf[{index}]").message("anyOf option must be a non-empty schema object"))
            _validate_schema_node(option, tool_name=tool_name, path=f"{path}.anyOf[{index}]")
    if "enum" in schema and (not isinstance(schema["enum"], list) or not schema["enum"]):
        raise ToolSchemaValidationError(context.message("enum must be a non-empty list"))
