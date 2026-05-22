from __future__ import annotations

import pytest

from toolrobust.tools.local_tools import get_default_local_tools, list_available_tool_ids
from toolrobust.tools.schema_utils import ToolSchemaValidationError, normalize_tool_schema, validate_tool_registry, validate_tool_schema


def test_all_builtin_tool_schemas_are_valid() -> None:
    for tool in get_default_local_tools():
        validate_tool_schema(normalize_tool_schema(tool.spec))


def test_sort_items_schema_declares_array_items_explicitly() -> None:
    sort_tool = next(tool for tool in get_default_local_tools() if tool.spec.canonical_tool_id == "sort_items")
    normalized = normalize_tool_schema(sort_tool.spec)
    items_schema = normalized.parameters.properties["items"]
    assert items_schema["type"] == "array"
    assert "items" in items_schema
    assert "anyOf" in items_schema["items"]


def test_invalid_array_schema_is_rejected() -> None:
    broken_tool = normalize_tool_schema(get_default_local_tools()[0].spec)
    broken_tool.exposed_name = "broken_array_tool"
    broken_tool.parameters.properties["place"] = {"type": "array", "description": "Broken array schema"}
    with pytest.raises(ToolSchemaValidationError, match="array schema must declare a non-empty 'items' schema"):
        validate_tool_schema(broken_tool)


def test_registry_output_is_validated() -> None:
    normalized = validate_tool_registry(tool.spec for tool in get_default_local_tools())
    assert len(normalized) == len(get_default_local_tools())


def test_builtin_tool_count_expanded_for_v21() -> None:
    assert len(list_available_tool_ids()) >= 10
