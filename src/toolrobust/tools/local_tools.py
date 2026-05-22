"""Deterministic local tools used by benchmark v2.1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from toolrobust.domain.schemas import ToolParameterSchema, ToolSpec
from toolrobust.tools.base import LocalTool, ToolExecutionError

_WEATHER_DATA = {
    ("beijing", "today"): {"condition": "sunny", "temperature_c": 26},
    ("beijing", "tomorrow"): {"condition": "hazy", "temperature_c": 24},
    ("shanghai", "today"): {"condition": "cloudy", "temperature_c": 23},
    ("shanghai", "tomorrow"): {"condition": "rain", "temperature_c": 20},
    ("hangzhou", "today"): {"condition": "overcast", "temperature_c": 21},
    ("hangzhou", "tomorrow"): {"condition": "rain", "temperature_c": 19},
    ("london", "today"): {"condition": "drizzle", "temperature_c": 13},
    ("london", "tomorrow"): {"condition": "windy", "temperature_c": 14},
    ("new york", "today"): {"condition": "clear", "temperature_c": 18},
    ("new york", "tomorrow"): {"condition": "cloudy", "temperature_c": 17},
    ("singapore", "today"): {"condition": "humid", "temperature_c": 31},
    ("singapore", "tomorrow"): {"condition": "storm", "temperature_c": 29},
}

_AIR_QUALITY_DATA = {
    ("beijing", "today"): {"aqi": 118, "category": "unhealthy_for_sensitive_groups"},
    ("beijing", "tomorrow"): {"aqi": 106, "category": "unhealthy_for_sensitive_groups"},
    ("shanghai", "today"): {"aqi": 74, "category": "moderate"},
    ("shanghai", "tomorrow"): {"aqi": 68, "category": "moderate"},
    ("hangzhou", "today"): {"aqi": 63, "category": "moderate"},
    ("hangzhou", "tomorrow"): {"aqi": 59, "category": "moderate"},
    ("london", "today"): {"aqi": 41, "category": "good"},
    ("london", "tomorrow"): {"aqi": 46, "category": "good"},
    ("new york", "today"): {"aqi": 52, "category": "moderate"},
    ("new york", "tomorrow"): {"aqi": 49, "category": "good"},
    ("singapore", "today"): {"aqi": 82, "category": "moderate"},
    ("singapore", "tomorrow"): {"aqi": 91, "category": "moderate"},
}

_DOCUMENTS = {
    "finance": [
        "Quarterly earnings summary and revenue outlook",
        "Budget allocation memo for regional teams",
        "Foreign exchange risk guidance document",
        "Treasury cash policy handbook",
    ],
    "engineering": [
        "API migration playbook for backend services",
        "Incident review for routing latency regression",
        "Tool calling interface design notes",
        "Search index rollback checklist",
    ],
    "travel": [
        "City metro pass comparison guide",
        "Weekend train itinerary recommendations",
        "Airport shuttle policy reference",
        "Passport document checklist for transit stops",
    ],
}

_POLICIES = {
    "finance": [
        "Foreign exchange approval policy",
        "Budget revision approval matrix",
        "Expense reimbursement timing standard",
    ],
    "hr": [
        "Remote work eligibility policy",
        "Parental leave accrual guideline",
        "Travel reimbursement for recruiting trips",
    ],
    "it": [
        "Laptop replacement lifecycle policy",
        "Password rotation exception process",
        "VPN access review procedure",
    ],
}

_FAQS = {
    "cloud_app": [
        "How do I reset my workspace password?",
        "Where can I find billing invoices?",
        "Can I export audit logs from the admin console?",
    ],
    "router_hub": [
        "How do I update the router firmware?",
        "Where can I change the guest Wi-Fi password?",
        "Why is the router blinking amber after reboot?",
    ],
    "payroll_portal": [
        "How do I download my pay stub?",
        "Where can I update my bank account details?",
        "Why does my tax form still show pending?",
    ],
}

_ROUTES = {
    ("home", "office", "drive"): {"distance_km": 12, "duration_min": 28},
    ("station", "museum", "walk"): {"distance_km": 2, "duration_min": 24},
    ("campus", "airport", "taxi"): {"distance_km": 32, "duration_min": 40},
    ("hotel", "expo", "metro"): {"distance_km": 9, "duration_min": 22},
    ("library", "stadium", "bus"): {"distance_km": 7, "duration_min": 26},
    ("harbor", "gallery", "bike"): {"distance_km": 4, "duration_min": 18},
    ("mall", "theater", "drive"): {"distance_km": 14, "duration_min": 31},
    ("pier", "market", "walk"): {"distance_km": 3, "duration_min": 35},
    ("clinic", "campus", "metro"): {"distance_km": 11, "duration_min": 27},
    ("plaza", "airport", "bus"): {"distance_km": 18, "duration_min": 44},
}

_UNIT_TO_BASE = {
    "c": ("temperature", lambda x: x),
    "f": ("temperature", lambda x: (x - 32) * 5 / 9),
    "km": ("distance", lambda x: x),
    "m": ("distance", lambda x: x / 1000),
    "kg": ("weight", lambda x: x),
    "g": ("weight", lambda x: x / 1000),
    "l": ("volume", lambda x: x),
    "ml": ("volume", lambda x: x / 1000),
}

_CURRENCY_TO_USD = {
    "usd": 1.0,
    "eur": 1.08,
    "cny": 0.14,
    "jpy": 0.0068,
    "gbp": 1.27,
}

_GENERIC_TOOL_GROUPS = {
    "calendar_ops": (
        "create_calendar_event",
        "update_calendar_event",
        "check_calendar_availability",
        "cancel_calendar_event",
    ),
    "email_ops": (
        "draft_email",
        "summarize_email",
        "search_email",
        "classify_email",
    ),
    "file_ops": (
        "search_file",
        "summarize_file",
        "convert_file_format",
        "extract_file_metadata",
    ),
    "commerce_ops": (
        "lookup_product",
        "compare_product_price",
        "track_order",
        "estimate_shipping",
    ),
    "crm_ops": (
        "lookup_customer",
        "create_support_ticket",
        "update_ticket_status",
        "lookup_invoice",
    ),
    "database_ops": (
        "query_table",
        "aggregate_table",
        "lookup_record",
        "validate_record",
    ),
    "security_ops": (
        "check_permission",
        "rotate_api_key",
        "audit_login_events",
        "revoke_session",
    ),
}

_GENERIC_TOOL_DESCRIPTIONS = {
    "create_calendar_event": "Create a calendar event for an entity within a workspace scope.",
    "update_calendar_event": "Update a calendar event field for an entity within a workspace scope.",
    "check_calendar_availability": "Check calendar availability for an entity within a workspace scope.",
    "cancel_calendar_event": "Cancel a calendar event for an entity within a workspace scope.",
    "draft_email": "Draft an email for a recipient or thread within a mailbox scope.",
    "summarize_email": "Summarize an email thread within a mailbox scope.",
    "search_email": "Search email messages within a mailbox scope.",
    "classify_email": "Classify an email thread within a mailbox scope.",
    "search_file": "Search files within a document workspace.",
    "summarize_file": "Summarize a file within a document workspace.",
    "convert_file_format": "Convert a file representation within a document workspace.",
    "extract_file_metadata": "Extract metadata for a file within a document workspace.",
    "lookup_product": "Look up product information within a commerce catalog.",
    "compare_product_price": "Compare product price information within a commerce catalog.",
    "track_order": "Track an order within a commerce system.",
    "estimate_shipping": "Estimate shipping information within a commerce system.",
    "lookup_customer": "Look up customer information within a support workspace.",
    "create_support_ticket": "Create a support ticket within a support workspace.",
    "update_ticket_status": "Update a support ticket status within a support workspace.",
    "lookup_invoice": "Look up invoice information within a billing workspace.",
    "query_table": "Query rows from a database table within a data workspace.",
    "aggregate_table": "Aggregate a database table within a data workspace.",
    "lookup_record": "Look up a database record within a data workspace.",
    "validate_record": "Validate a database record within a data workspace.",
    "check_permission": "Check a permission state within a security workspace.",
    "rotate_api_key": "Rotate an API key within a security workspace.",
    "audit_login_events": "Audit login events within a security workspace.",
    "revoke_session": "Revoke a session within a security workspace.",
}

TOOL_GROUPS = {
    "weather_env": ("get_weather", "get_air_quality", "get_temperature"),
    "search_retrieval": ("search_document", "search_policy", "search_faq"),
    "route_travel": ("calculate_route", "estimate_travel_time"),
    "list_ops": ("sort_items", "filter_items"),
    "conversion": ("convert_unit", "convert_currency"),
    **_GENERIC_TOOL_GROUPS,
}

_TOOL_GROUP_BY_ID = {tool_id: group for group, ids in TOOL_GROUPS.items() for tool_id in ids}


def _base_tool_spec(
    *,
    canonical_tool_id: str,
    description: str,
    properties: dict[str, dict[str, Any]],
    required: list[str],
    aliases: list[str],
) -> ToolSpec:
    return ToolSpec(
        canonical_tool_id=canonical_tool_id,
        exposed_name=canonical_tool_id,
        clean_tool_name=canonical_tool_id,
        description=description,
        aliases=aliases,
        parameter_name_map={name: name for name in properties},
        parameters=ToolParameterSchema(properties=properties, required=required),
        metadata={"tool_group": _TOOL_GROUP_BY_ID.get(canonical_tool_id, "misc")},
    )


def _from_base(unit: str, value: float) -> float:
    if unit == "c":
        return value
    if unit == "f":
        return value * 9 / 5 + 32
    if unit == "km":
        return value
    if unit == "m":
        return value * 1000
    if unit == "kg":
        return value
    if unit == "g":
        return value * 1000
    if unit == "l":
        return value
    if unit == "ml":
        return value * 1000
    raise ToolExecutionError(f"Unsupported unit: {unit}")


@dataclass(slots=True)
class ConfiguredLocalTool(LocalTool):
    _spec: ToolSpec
    _handler: Callable[[dict[str, Any]], Any]

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def execute(self, args: dict[str, Any]) -> Any:
        return self._handler(args)


def _execute_weather(args: dict[str, Any]) -> Any:
    key = (str(args["place"]).lower(), str(args["slot"]).lower())
    if key not in _WEATHER_DATA:
        raise ToolExecutionError(f"No weather data for {key}")
    return {"place": args["place"], "slot": args["slot"], **_WEATHER_DATA[key]}


def _execute_air_quality(args: dict[str, Any]) -> Any:
    key = (str(args["place"]).lower(), str(args["slot"]).lower())
    if key not in _AIR_QUALITY_DATA:
        raise ToolExecutionError(f"No air-quality data for {key}")
    return {"place": args["place"], "slot": args["slot"], **_AIR_QUALITY_DATA[key]}


def _execute_temperature(args: dict[str, Any]) -> Any:
    weather = _execute_weather({"place": args["place"], "slot": args["slot"]})
    scale = str(args["scale"]).lower()
    if scale not in {"c", "f"}:
        raise ToolExecutionError("Temperature unit must be c or f")
    value = weather["temperature_c"] if scale == "c" else round(weather["temperature_c"] * 9 / 5 + 32, 2)
    return {"place": args["place"], "slot": args["slot"], "scale": scale, "temperature": value}


def _execute_convert_unit(args: dict[str, Any]) -> Any:
    value = float(args["value"])
    source = str(args["source"]).lower()
    target = str(args["target"]).lower()
    if source not in _UNIT_TO_BASE or target not in _UNIT_TO_BASE:
        raise ToolExecutionError("Unsupported unit in conversion request")
    from_family, to_base = _UNIT_TO_BASE[source]
    to_family, _ = _UNIT_TO_BASE[target]
    if from_family != to_family:
        raise ToolExecutionError("Unit families do not match")
    converted = round(_from_base(target, to_base(value)), 4)
    return {"value": value, "source": source, "target": target, "converted_value": converted}


def _execute_convert_currency(args: dict[str, Any]) -> Any:
    value = float(args["value"])
    source = str(args["source"]).lower()
    target = str(args["target"]).lower()
    if source not in _CURRENCY_TO_USD or target not in _CURRENCY_TO_USD:
        raise ToolExecutionError("Unsupported currency in conversion request")
    usd_value = value * _CURRENCY_TO_USD[source]
    converted = round(usd_value / _CURRENCY_TO_USD[target], 4)
    return {
        "value": value,
        "source": source,
        "target": target,
        "converted_value": converted,
    }


def _execute_search_document(args: dict[str, Any]) -> Any:
    scope = str(args["scope"]).lower()
    query = str(args["query"]).lower()
    if scope not in _DOCUMENTS:
        raise ToolExecutionError(f"Unknown scope: {scope}")
    return {"scope": scope, "query": query, "matches": [title for title in _DOCUMENTS[scope] if query in title.lower()]}


def _execute_search_policy(args: dict[str, Any]) -> Any:
    scope = str(args["scope"]).lower()
    query = str(args["query"]).lower()
    if scope not in _POLICIES:
        raise ToolExecutionError(f"Unknown scope: {scope}")
    return {"scope": scope, "query": query, "matches": [title for title in _POLICIES[scope] if query in title.lower()]}


def _execute_search_faq(args: dict[str, Any]) -> Any:
    scope = str(args["scope"]).lower()
    query = str(args["query"]).lower()
    if scope not in _FAQS:
        raise ToolExecutionError(f"Unknown scope: {scope}")
    return {"scope": scope, "query": query, "matches": [title for title in _FAQS[scope] if query in title.lower()]}


def _execute_calculate_route(args: dict[str, Any]) -> Any:
    key = (str(args["origin"]).lower(), str(args["target"]).lower(), str(args["profile"]).lower())
    if key not in _ROUTES:
        raise ToolExecutionError(f"Unknown route: {key}")
    return {"origin": args["origin"], "target": args["target"], "profile": args["profile"], **_ROUTES[key]}


def _execute_estimate_travel_time(args: dict[str, Any]) -> Any:
    route = _execute_calculate_route(args)
    return {"origin": route["origin"], "target": route["target"], "profile": route["profile"], "duration_min": route["duration_min"]}


def _execute_sort_items(args: dict[str, Any]) -> Any:
    items = list(args["items"])
    rule = str(args["rule"]).lower()
    if rule not in {"asc", "desc"}:
        raise ToolExecutionError("Order must be asc or desc")
    return {"sorted_items": sorted(items, reverse=rule == "desc"), "rule": rule}


def _execute_filter_items(args: dict[str, Any]) -> Any:
    items = list(args["items"])
    rule = str(args["rule"]).lower()
    filtered = [item for item in items if rule in str(item).lower()]
    return {"filtered_items": filtered, "rule": rule}


def _execute_generic_structured_tool(tool_id: str, args: dict[str, Any]) -> Any:
    entity = str(args["entity"]).strip().lower()
    scope = str(args["scope"]).strip().lower()
    attribute = str(args["attribute"]).strip().lower()
    if not entity or not scope or not attribute:
        raise ToolExecutionError("Generic structured tools require entity, scope, and attribute")
    return {
        "entity": entity,
        "scope": scope,
        "attribute": attribute,
        "result": f"{tool_id}:{scope}:{entity}:{attribute}",
    }


def _generic_tool_spec(tool_id: str) -> ConfiguredLocalTool:
    return ConfiguredLocalTool(
        _base_tool_spec(
            canonical_tool_id=tool_id,
            description=_GENERIC_TOOL_DESCRIPTIONS[tool_id],
            aliases=[
                tool_id.replace("_", "-"),
                tool_id.replace("_", ""),
                f"{tool_id}_tool",
            ],
            properties={
                "entity": {"type": "string", "description": "Primary object or record label.", "examples": ["alpha"]},
                "scope": {"type": "string", "description": "Workspace, collection, or domain scope.", "examples": ["ops"]},
                "attribute": {"type": "string", "description": "Requested field, action, or facet.", "examples": ["status"]},
            },
            required=["entity", "scope", "attribute"],
        ),
        lambda args, tool_id=tool_id: _execute_generic_structured_tool(tool_id, args),
    )


def _build_all_tools() -> list[LocalTool]:
    return [
        ConfiguredLocalTool(
            _base_tool_spec(
                canonical_tool_id="get_weather",
                description="Retrieve an outdoor-condition snapshot for a place and time reference.",
                aliases=["weather_lookup", "wx_lookup", "fetchWeather"],
                properties={
                    "place": {"type": "string", "description": "Target location label.", "examples": ["Beijing"]},
                    "slot": {"type": "string", "description": "Requested time reference.", "enum": ["today", "tomorrow"], "examples": ["today"]},
                },
                required=["place", "slot"],
            ),
            _execute_weather,
        ),
        ConfiguredLocalTool(
            _base_tool_spec(
                canonical_tool_id="get_air_quality",
                description="Retrieve an environment-condition snapshot for a place and time reference.",
                aliases=["air_quality_lookup", "aqi_lookup", "fetchAirQuality"],
                properties={
                    "place": {"type": "string", "description": "Target location label.", "examples": ["Shanghai"]},
                    "slot": {"type": "string", "description": "Requested time reference.", "enum": ["today", "tomorrow"], "examples": ["tomorrow"]},
                },
                required=["place", "slot"],
            ),
            _execute_air_quality,
        ),
        ConfiguredLocalTool(
            _base_tool_spec(
                canonical_tool_id="get_temperature",
                description="Retrieve a numeric environment reading for a place and time reference.",
                aliases=["temperature_lookup", "temp_read", "fetchTemperature"],
                properties={
                    "place": {"type": "string", "description": "Target location label.", "examples": ["London"]},
                    "slot": {"type": "string", "description": "Requested time reference.", "enum": ["today", "tomorrow"], "examples": ["today"]},
                    "scale": {"type": "string", "description": "Requested reading format.", "enum": ["c", "f"], "examples": ["f"]},
                },
                required=["place", "slot", "scale"],
            ),
            _execute_temperature,
        ),
        ConfiguredLocalTool(
            _base_tool_spec(
                canonical_tool_id="convert_unit",
                description="Transform a scalar between supported representations.",
                aliases=["unit_converter", "unitx", "convertUnit"],
                properties={
                    "value": {"type": "number", "description": "Primary numeric input.", "examples": [5]},
                    "source": {"type": "string", "enum": ["c", "f", "km", "m", "kg", "g", "l", "ml"], "description": "Starting representation.", "examples": ["km"]},
                    "target": {"type": "string", "enum": ["c", "f", "km", "m", "kg", "g", "l", "ml"], "description": "Requested representation.", "examples": ["m"]},
                },
                required=["value", "source", "target"],
            ),
            _execute_convert_unit,
        ),
        ConfiguredLocalTool(
            _base_tool_spec(
                canonical_tool_id="convert_currency",
                description="Transform a scalar between supported monetary representations.",
                aliases=["currency_converter", "fx_convert", "convertCurrency"],
                properties={
                    "value": {"type": "number", "description": "Primary numeric input.", "examples": [120]},
                    "source": {"type": "string", "enum": ["usd", "eur", "cny", "jpy", "gbp"], "description": "Starting representation.", "examples": ["usd"]},
                    "target": {"type": "string", "enum": ["usd", "eur", "cny", "jpy", "gbp"], "description": "Requested representation.", "examples": ["cny"]},
                },
                required=["value", "source", "target"],
            ),
            _execute_convert_currency,
        ),
        ConfiguredLocalTool(
            _base_tool_spec(
                canonical_tool_id="search_document",
                description="Query one content collection with a term and scope.",
                aliases=["doc_search", "docfind", "searchDocs"],
                properties={
                    "query": {"type": "string", "description": "Lookup term.", "examples": ["routing"]},
                    "scope": {"type": "string", "enum": ["finance", "engineering", "travel"], "description": "Collection selector.", "examples": ["engineering"]},
                },
                required=["query", "scope"],
            ),
            _execute_search_document,
        ),
        ConfiguredLocalTool(
            _base_tool_spec(
                canonical_tool_id="search_policy",
                description="Query one governed content collection with a term and scope.",
                aliases=["policy_search", "policyfind", "searchPolicies"],
                properties={
                    "query": {"type": "string", "description": "Lookup term.", "examples": ["budget"]},
                    "scope": {"type": "string", "enum": ["finance", "hr", "it"], "description": "Collection selector.", "examples": ["finance"]},
                },
                required=["query", "scope"],
            ),
            _execute_search_policy,
        ),
        ConfiguredLocalTool(
            _base_tool_spec(
                canonical_tool_id="search_faq",
                description="Query one support-oriented content collection with a term and scope.",
                aliases=["faq_search", "faqfinder", "searchFaqs"],
                properties={
                    "query": {"type": "string", "description": "Lookup term.", "examples": ["password"]},
                    "scope": {"type": "string", "enum": ["cloud_app", "router_hub", "payroll_portal"], "description": "Collection selector.", "examples": ["cloud_app"]},
                },
                required=["query", "scope"],
            ),
            _execute_search_faq,
        ),
        ConfiguredLocalTool(
            _base_tool_spec(
                canonical_tool_id="calculate_route",
                description="Produce a movement-planning result from two endpoints and a profile.",
                aliases=["route_planner", "routecalc", "calculateRoute"],
                properties={
                    "origin": {"type": "string", "description": "Boundary label.", "examples": ["home"]},
                    "target": {"type": "string", "description": "Boundary label.", "examples": ["office"]},
                    "profile": {"type": "string", "enum": ["drive", "walk", "taxi", "metro", "bus", "bike"], "description": "Requested travel profile.", "examples": ["drive"]},
                },
                required=["origin", "target", "profile"],
            ),
            _execute_calculate_route,
        ),
        ConfiguredLocalTool(
            _base_tool_spec(
                canonical_tool_id="estimate_travel_time",
                description="Produce a movement-duration result from two endpoints and a profile.",
                aliases=["travel_time_estimator", "eta_calc", "estimateEta"],
                properties={
                    "origin": {"type": "string", "description": "Boundary label.", "examples": ["station"]},
                    "target": {"type": "string", "description": "Boundary label.", "examples": ["museum"]},
                    "profile": {"type": "string", "enum": ["drive", "walk", "taxi", "metro", "bus", "bike"], "description": "Requested travel profile.", "examples": ["walk"]},
                },
                required=["origin", "target", "profile"],
            ),
            _execute_estimate_travel_time,
        ),
        ConfiguredLocalTool(
            _base_tool_spec(
                canonical_tool_id="sort_items",
                description="Return an ordered version of a provided set under a rule.",
                aliases=["item_sorter", "sortseq", "sortItems"],
                properties={
                    "items": {
                        "type": "array",
                        "items": {"anyOf": [{"type": "string", "description": "Entry."}, {"type": "number", "description": "Entry."}]},
                        "description": "Provided set.",
                        "examples": [[3, 1, 2]],
                    },
                    "rule": {"type": "string", "enum": ["asc", "desc"], "description": "Applied rule value.", "examples": ["asc"]},
                },
                required=["items", "rule"],
            ),
            _execute_sort_items,
        ),
        ConfiguredLocalTool(
            _base_tool_spec(
                canonical_tool_id="filter_items",
                description="Return a reduced version of a provided set under a rule.",
                aliases=["item_filter", "sift_items", "filterItems"],
                properties={
                    "items": {
                        "type": "array",
                        "items": {"anyOf": [{"type": "string", "description": "Entry."}, {"type": "number", "description": "Entry."}]},
                        "description": "Provided set.",
                        "examples": [["alpha", "beta", "alphabet"]],
                    },
                    "rule": {"type": "string", "description": "Applied rule value.", "examples": ["alp"]},
                },
                required=["items", "rule"],
            ),
            _execute_filter_items,
        ),
        *[
            _generic_tool_spec(tool_id)
            for group_tools in _GENERIC_TOOL_GROUPS.values()
            for tool_id in group_tools
        ],
    ]


_ALL_LOCAL_TOOLS = _build_all_tools()


def list_available_tool_ids() -> list[str]:
    return [tool.spec.canonical_tool_id for tool in _ALL_LOCAL_TOOLS]


def expand_enabled_tool_ids(enabled_tools: list[str] | None = None, *, include_similar_tool_groups: bool = False) -> list[str]:
    available = set(list_available_tool_ids())
    if not enabled_tools:
        return sorted(available)
    requested = {tool_id for tool_id in enabled_tools if tool_id in available}
    if include_similar_tool_groups:
        for group_name, group_tools in TOOL_GROUPS.items():
            if requested.intersection(group_tools):
                requested.update(group_tools)
    return sorted(requested)


def get_default_local_tools(
    enabled_tools: list[str] | None = None,
    *,
    include_similar_tool_groups: bool = False,
) -> list[LocalTool]:
    enabled = set(expand_enabled_tool_ids(enabled_tools, include_similar_tool_groups=include_similar_tool_groups))
    return [tool for tool in _ALL_LOCAL_TOOLS if tool.spec.canonical_tool_id in enabled]
