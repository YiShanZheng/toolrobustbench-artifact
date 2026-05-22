"""Dataset builders for ToolRobust v3."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Any

from toolrobust.datasets.io import write_task_samples
from toolrobust.datasets.validators import validate_task_samples
from toolrobust.domain.enums import BenchmarkFamily, DatasetSplit, SeverityLevel, StageTarget
from toolrobust.domain.schemas import BenchmarkV3Config, TaskSample, default_clean_axes
from toolrobust.perturbation.catalog import SUBTYPE_STAGE_TARGET
from toolrobust.perturbation.pipeline import CompositeVariantSpec, VariantSpec, PerturbationPipeline
from toolrobust.tools.executor import ToolExecutor
from toolrobust.tools.local_tools import TOOL_GROUPS, list_available_tool_ids
from toolrobust.tools.registry import RegistryBuilder

_SAMPLE_ID_PREFIX = {"get_weather": "weather", "get_air_quality": "air", "get_temperature": "temp", "convert_unit": "unit", "convert_currency": "currency", "search_document": "searchdoc", "search_policy": "policy", "search_faq": "faq", "calculate_route": "route", "estimate_travel_time": "eta", "sort_items": "sort", "filter_items": "filter"}

_PATTERNS = {
    "get_weather": ["For {place} {slot}, what's the overall outside picture?", "I'm checking {place} for {slot}; I want the general weather, not the air reading.", "Tell me how it looks outside in {place} on {slot}."],
    "get_air_quality": ["For {place} {slot}, how is the air side looking?", "I'm checking {place} for {slot}; focus on the air reading, not the general weather.", "Tell me what the air situation is in {place} on {slot}."],
    "get_temperature": ["For {place} {slot}, what is the temperature in {scale_upper}?", "I'm checking {place} for {slot}; I only want the numeric reading in {scale_upper}.", "Tell me the degree reading for {place} on {slot}, using {scale_upper}."],
    "convert_unit": ["Take {value} in {source} and express it in {target}. This is the measurement side, not money.", "I need {value} moved from {source} to {target} as a measurement conversion.", "What's the equivalent of {value} from {source} into {target} on the non-currency side?"],
    "convert_currency": ["Take {value} in {source_upper} and express it in {target_upper}. This is the money side, not a measurement conversion.", "I need {value} moved from {source_upper} to {target_upper} as a money conversion.", "What's the equivalent of {value} from {source_upper} into {target_upper} on the currency side?"],
    "search_document": ["Find the regular document in the {scope} collection about the topic '{query}'; not a rule page or help answer.", "Look through the {scope} materials for the topic '{query}'. I want the general document entry.", "I need the normal document in {scope} about '{query}', not the policy item or FAQ answer."],
    "search_policy": ["Find the governing policy in the {scope} collection about the topic '{query}'; not general docs or help answers.", "Look through the {scope} materials for the topic '{query}'. I want the policy entry.", "I need the policy item in {scope} about '{query}', not the general document or FAQ answer."],
    "search_faq": ["Find the FAQ-style answer in the {scope} collection about the topic '{query}'; not general docs or rules.", "Look through the {scope} materials for the topic '{query}'. I want the FAQ entry.", "I need the help-answer entry in {scope} about '{query}', not the policy item or general document."],
    "calculate_route": ["I need to get from {origin} to {target} using {profile}; give me the way to go, not just the time.", "Plan the {profile} trip from {origin} to {target}. I want the route itself.", "Work out how to go from {origin} to {target} via {profile}; don't reduce it to duration only."],
    "estimate_travel_time": ["I need to get from {origin} to {target} using {profile}; just tell me how long, not the route details.", "Estimate the {profile} trip time from {origin} to {target}. I only want the duration.", "Work out how long going from {origin} to {target} via {profile} should take; no route breakdown needed."],
    "sort_items": ["Take {items} and arrange them with rule {rule}; I want the whole set reordered, not filtered down.", "Rework {items} according to {rule}. Keep everything, just change the order.", "Return {items} in {rule} order; don't throw any entries away."],
    "filter_items": ["Take {items} and apply rule {rule}; I only want the matching subset, not a reordered full list.", "Check {items} against {rule}. Keep only the entries that line up with it.", "Return just the items from {items} that match {rule}; this is not an ordering request."],
}

_SPECIALIZED_TOOL_IDS = set(_PATTERNS)

_GENERIC_SCENARIOS = [
    {"entity": "alpha", "scope": "ops", "attribute": "status"},
    {"entity": "beta", "scope": "sales", "attribute": "owner"},
    {"entity": "gamma", "scope": "support", "attribute": "priority"},
    {"entity": "delta", "scope": "finance", "attribute": "amount"},
    {"entity": "epsilon", "scope": "security", "attribute": "state"},
    {"entity": "zeta", "scope": "product", "attribute": "summary"},
]

for _group_name, _group_tools in TOOL_GROUPS.items():
    if _group_name in {"weather_env", "search_retrieval", "route_travel", "list_ops", "conversion"}:
        continue
    for _tool_id in _group_tools:
        _SAMPLE_ID_PREFIX.setdefault(_tool_id, _tool_id.replace("_", "")[:18])
        _PATTERNS.setdefault(
            _tool_id,
            [
                "Use {tool_label} for entity {entity} in {scope}; return the {attribute} field.",
                "For {scope}, handle {entity} with {tool_label}. I need {attribute}, not a neighboring workflow.",
                "Please run the {tool_label} operation on {entity} in {scope} and focus on {attribute}.",
                "Inside {scope}, the target is {entity}; the needed facet is {attribute} via {tool_label}.",
                "I need {attribute} for {entity} under {scope}; choose the {tool_label} style operation.",
                "Work on {entity} in {scope}; use {tool_label} and keep the requested facet as {attribute}.",
            ],
        )

_CLUSTER_SAMPLING_PRIORITY = ("search_retrieval", "route_travel", "list_ops", "conversion", "weather_env")


@dataclass(slots=True)
class DatasetBuilder:
    config: BenchmarkV3Config

    def build(self, progress: object | None = None) -> tuple[list[TaskSample], list[TaskSample]]:
        clean_templates = self._build_template_pool()
        if progress is not None:
            progress.start("generating_clean_samples", total_items=len(clean_templates))
        clean = self._build_clean_samples(clean_templates, progress=progress)
        validate_task_samples(clean)

        pipeline = PerturbationPipeline(self.config)
        self.config.generated_dir.mkdir(parents=True, exist_ok=True)
        pipeline.preflight_gate_report_frame().to_csv(self.config.generated_dir / "preflight_severity_gate_report.csv", index=False)
        if progress is not None:
            progress.finish_stage("[progress] completed stage=generating_clean_samples")
            single_count = 0
            if not self.config.cross_family.enabled or self.config.cross_family.include_single_family_baseline:
                single_count = len(clean) * len(pipeline.variant_specs()) * self.config.noise.samples_per_bucket
            perturbation_count = single_count + len(clean) * len(pipeline.composite_variant_specs()) * self.config.cross_family.samples_per_combo_bucket
            progress.start("generating_perturbations", total_items=perturbation_count)
        perturbed = self._build_perturbed_samples(clean, pipeline, progress=progress)
        validate_task_samples(perturbed)

        if progress is not None:
            progress.finish_stage("[progress] completed stage=generating_perturbations")
            progress.start("writing_dataset_outputs", total_items=2)
        write_task_samples(self.config.clean_dataset_path, clean)
        if progress is not None:
            progress.advance(current_item=self.config.clean_dataset_path.name)
        write_task_samples(self.config.perturbed_dataset_path, perturbed)
        if progress is not None:
            progress.advance(current_item=self.config.perturbed_dataset_path.name)
            progress.finish_stage("[progress] completed stage=writing_dataset_outputs")
        return clean, perturbed

    def selected_tool_ids(self) -> list[str]:
        available = set(list_available_tool_ids())
        requested = self.config.dataset.sample_tool_count
        if requested is None or requested >= len(available):
            return sorted(available)

        selected: list[str] = []
        for group_name in _CLUSTER_SAMPLING_PRIORITY:
            group_tools = [tool_id for tool_id in TOOL_GROUPS[group_name] if tool_id in available]
            if not group_tools:
                continue
            remaining = requested - len(selected)
            if remaining <= 0:
                break
            if len(group_tools) <= remaining:
                selected.extend(group_tools)
                continue
            selected.extend(group_tools[:remaining])
            break

        if len(selected) < requested:
            leftovers = sorted(available.difference(selected))
            rng = Random(self.config.dataset.seed)
            selected.extend(rng.sample(leftovers, requested - len(selected)))
        return sorted(selected)

    def _build_clean_samples(self, templates: list[dict[str, Any]], progress: object | None = None) -> list[TaskSample]:
        executor = ToolExecutor()
        registry = RegistryBuilder().build_clean_registry(self.selected_tool_ids(), include_similar_tool_groups=False)
        axes = default_clean_axes()
        samples: list[TaskSample] = []
        for template in templates:
            tool = registry.get_tool(template["canonical_tool_id"])
            assert tool is not None
            expected_result = executor.execute(template["canonical_tool_id"], template["expected_arguments"])
            samples.append(
                TaskSample(
                    sample_id=template["sample_id"],
                    split=DatasetSplit.CLEAN,
                    family=axes["family"],
                    subtype=axes["subtype"],
                    operator=axes["operator"],
                    severity=SeverityLevel.CLEAN,
                    stage_target=axes["stage_target"],
                    registry_variant_id="clean",
                    canonical_tool_id=tool.canonical_tool_id,
                    exposed_tool_name=tool.exposed_name,
                    clean_tool_name=tool.clean_tool_name,
                    task_text=template["task_text"],
                    user_instruction=template["task_text"],
                    expected_arguments=template["expected_arguments"],
                    expected_result=expected_result,
                    operators_applied=["clean"],
                    metadata=template["metadata"],
                )
            )
            if progress is not None:
                progress.advance(current_item=template["sample_id"])
        return samples

    def _build_perturbed_samples(self, clean_samples: list[TaskSample], pipeline: PerturbationPipeline, progress: object | None = None) -> list[TaskSample]:
        registry_index = pipeline.registry_index()
        perturbed: list[TaskSample] = []
        for clean_sample in clean_samples:
            if not self.config.cross_family.enabled or self.config.cross_family.include_single_family_baseline:
                for spec in pipeline.variant_specs():
                    tool = registry_index[spec.registry_key].get_tool(clean_sample.canonical_tool_id)
                    assert tool is not None
                    for bucket_idx in range(self.config.noise.samples_per_bucket):
                        rewritten = self._rewrite_task_text(clean_sample, spec, tool.exposed_name)
                        perturbed.append(
                            TaskSample(
                                sample_id=f"{clean_sample.sample_id}__{self._slug(spec.subtype)}__{spec.severity}__{bucket_idx + 1}",
                                split=DatasetSplit.PERTURBED,
                                family=spec.family,
                                subtype=spec.subtype,
                                operator=" + ".join(operator.operator_name for operator in spec.operators),
                                severity=SeverityLevel(spec.severity),
                                stage_target=spec.stage_target,
                                registry_variant_id=spec.registry_key,
                                canonical_tool_id=clean_sample.canonical_tool_id,
                                exposed_tool_name=tool.exposed_name,
                                clean_tool_name=clean_sample.clean_tool_name,
                                task_text=rewritten,
                                user_instruction=rewritten,
                                expected_arguments=clean_sample.expected_arguments,
                                expected_result=clean_sample.expected_result,
                                operators_applied=[operator.operator_name for operator in spec.operators],
                                metadata={**clean_sample.metadata, "source_sample_id": clean_sample.sample_id, "bucket_index": bucket_idx},
                            )
                        )
                        if progress is not None:
                            progress.advance(current_item=f"{clean_sample.sample_id}_{self._slug(spec.subtype)}_{spec.severity}_{bucket_idx + 1}")
            for spec in pipeline.composite_variant_specs():
                tool = registry_index[spec.registry_key].get_tool(clean_sample.canonical_tool_id)
                assert tool is not None
                for bucket_idx in range(self.config.cross_family.samples_per_combo_bucket):
                    rewritten = self._rewrite_task_text(clean_sample, spec, tool.exposed_name)
                    perturbed.append(
                        TaskSample(
                            sample_id=f"{clean_sample.sample_id}__xf__{self._slug(spec.composition_id)}__{spec.severity}__{bucket_idx + 1}",
                            split=DatasetSplit.PERTURBED,
                            family=BenchmarkFamily.CROSS_FAMILY.value,
                            subtype=spec.composition_id,
                            operator=" + ".join(operator.operator_name for operator in spec.operators),
                            severity=SeverityLevel(spec.severity),
                            stage_target=StageTarget.S5.value,
                            registry_variant_id=spec.registry_key,
                            canonical_tool_id=clean_sample.canonical_tool_id,
                            exposed_tool_name=tool.exposed_name,
                            clean_tool_name=clean_sample.clean_tool_name,
                            task_text=rewritten,
                            user_instruction=rewritten,
                            expected_arguments=clean_sample.expected_arguments,
                            expected_result=clean_sample.expected_result,
                            operators_applied=[operator.operator_name for operator in spec.operators],
                            composition_id=spec.composition_id,
                            families_applied=list(spec.families_applied),
                            subtypes_applied=list(spec.subtypes_applied),
                            stage_targets_applied=list(spec.stage_targets_applied),
                            primary_family=spec.families_applied[0] if spec.families_applied else None,
                            metadata={
                                **clean_sample.metadata,
                                "source_sample_id": clean_sample.sample_id,
                                "bucket_index": bucket_idx,
                                "composition_id": spec.composition_id,
                                "families_applied": list(spec.families_applied),
                                "subtypes_applied": list(spec.subtypes_applied),
                                "stage_targets_applied": list(spec.stage_targets_applied),
                            },
                        )
                    )
                    if progress is not None:
                        progress.advance(current_item=f"{clean_sample.sample_id}_{self._slug(spec.composition_id)}_{spec.severity}_{bucket_idx + 1}")
        return perturbed

    def _build_template_pool(self) -> list[dict[str, Any]]:
        templates: list[dict[str, Any]] = []
        for tool_id in self.selected_tool_ids():
            for index, args in enumerate(self._scenario_bank()[tool_id][: self.config.dataset.max_clean_per_tool], start=1):
                templates.append(
                    {
                        "sample_id": f"{_SAMPLE_ID_PREFIX[tool_id]}_{index:02d}",
                        "canonical_tool_id": tool_id,
                        "task_text": self._render_request(tool_id, args, index - 1),
                        "expected_arguments": args,
                        "metadata": {"scenario_index": index - 1, "tool_group": self._tool_group_for(tool_id)},
                    }
                )
        return templates

    @staticmethod
    def _scenario_bank() -> dict[str, list[dict[str, Any]]]:
        scenarios = {
            "get_weather": [{"place": "Beijing", "slot": "today"}, {"place": "Shanghai", "slot": "tomorrow"}, {"place": "London", "slot": "today"}],
            "get_air_quality": [{"place": "Beijing", "slot": "today"}, {"place": "Shanghai", "slot": "tomorrow"}, {"place": "London", "slot": "today"}],
            "get_temperature": [{"place": "Beijing", "slot": "today", "scale": "c"}, {"place": "Shanghai", "slot": "tomorrow", "scale": "f"}, {"place": "London", "slot": "today", "scale": "c"}],
            "convert_unit": [{"value": 68, "source": "f", "target": "c"}, {"value": 20, "source": "c", "target": "f"}, {"value": 5, "source": "km", "target": "m"}],
            "convert_currency": [{"value": 100, "source": "usd", "target": "cny"}, {"value": 250, "source": "eur", "target": "usd"}, {"value": 75, "source": "gbp", "target": "eur"}],
            "search_document": [{"query": "budget", "scope": "finance"}, {"query": "routing", "scope": "engineering"}, {"query": "train", "scope": "travel"}],
            "search_policy": [{"query": "budget", "scope": "finance"}, {"query": "remote", "scope": "hr"}, {"query": "vpn", "scope": "it"}],
            "search_faq": [{"query": "password", "scope": "cloud_app"}, {"query": "firmware", "scope": "router_hub"}, {"query": "stub", "scope": "payroll_portal"}],
            "calculate_route": [{"origin": "hotel", "target": "expo", "profile": "metro"}, {"origin": "station", "target": "museum", "profile": "walk"}, {"origin": "home", "target": "office", "profile": "drive"}],
            "estimate_travel_time": [{"origin": "hotel", "target": "expo", "profile": "metro"}, {"origin": "station", "target": "museum", "profile": "walk"}, {"origin": "home", "target": "office", "profile": "drive"}],
            "sort_items": [{"items": [10, 7, 9], "rule": "desc"}, {"items": ["pear", "apple", "banana"], "rule": "desc"}, {"items": [3, 1, 2], "rule": "asc"}],
            "filter_items": [{"items": ["router", "route", "branch"], "rule": "rout"}, {"items": ["alpha", "beta", "alphabet"], "rule": "alp"}, {"items": ["budget", "ledger", "bud"], "rule": "bud"}],
        }
        for group_name, group_tools in TOOL_GROUPS.items():
            if group_name in {"weather_env", "search_retrieval", "route_travel", "list_ops", "conversion"}:
                continue
            for tool_id in group_tools:
                scenarios[tool_id] = list(_GENERIC_SCENARIOS)
        return scenarios

    @staticmethod
    def _tool_group_for(tool_id: str) -> str:
        for group_name, group_tools in TOOL_GROUPS.items():
            if tool_id in group_tools:
                return group_name
        return "conversion"

    @classmethod
    def _rewrite_task_text(cls, clean_sample: TaskSample, spec: VariantSpec | CompositeVariantSpec, exposed_tool_name: str) -> str:
        scenario_index = int(clean_sample.metadata.get("scenario_index", 0))
        base = cls._render_request(clean_sample.canonical_tool_id, clean_sample.expected_arguments, scenario_index)
        if isinstance(spec, CompositeVariantSpec):
            user_intent_component = next(
                (component for component in spec.component_specs if component.family == BenchmarkFamily.USER_INTENT.value),
                None,
            )
            if user_intent_component is None:
                return base
            return cls._rewrite_user_intent_task(
                clean_sample.canonical_tool_id,
                clean_sample.expected_arguments,
                scenario_index,
                [operator.operator_name for operator in user_intent_component.operators],
            )
        if spec.family in {
            BenchmarkFamily.INTERFACE_EXPOSURE.value,
            BenchmarkFamily.OBSERVATION.value,
            BenchmarkFamily.EXECUTION_ENVIRONMENT.value,
        }:
            return base
        if spec.family == BenchmarkFamily.USER_INTENT.value:
            return cls._rewrite_user_intent_task(
                clean_sample.canonical_tool_id,
                clean_sample.expected_arguments,
                scenario_index,
                [operator.operator_name for operator in spec.operators],
            )
        return base

    @classmethod
    def _rewrite_user_intent_task(cls, tool_id: str, args: dict[str, Any], variant_index: int, operators: list[str]) -> str:
        base = cls._render_request(tool_id, args, variant_index)
        if tool_id not in _SPECIALIZED_TOOL_IDS:
            return cls._generic_user_intent_request(base, args, operators)
        signature = tuple(operators)
        if signature == ("style_transfer", "constraint_reposition"):
            return cls._medium_indirect_request(tool_id, args)
        if signature == ("style_transfer", "cross_sentence_scatter", "implication_rewrite"):
            return cls._heavy_indirect_request(tool_id, args)
        if signature == ("slot_implicitization", "partial_constraint_drop"):
            return cls._heavy_underspecified_request(tool_id, args)
        if signature == ("near_semantic_distractor", "side_goal_insertion"):
            return cls._medium_distraction_request(tool_id, args)
        if signature == ("near_semantic_distractor", "side_goal_insertion", "soft_topic_drift"):
            return cls._heavy_distraction_request(tool_id, args)
        if signature == ("contradictory_constraint", "late_revision", "intra_prompt_conflict"):
            return cls._heavy_conflict_request(tool_id, args)

        text = base
        for operator_name in operators:
            text = cls._apply_user_intent_operator(text, tool_id, args, operator_name)
        return text

    @classmethod
    def _generic_user_intent_request(cls, base: str, args: dict[str, Any], operators: list[str]) -> str:
        signature = tuple(operators)
        entity = args.get("entity", "the target")
        scope = args.get("scope", "the workspace")
        attribute = args.get("attribute", "the requested field")
        if signature == ("style_transfer", "constraint_reposition"):
            return f"Could you help with {entity} in {scope}? The part that matters, once the request settles, is the {attribute} field for the intended operation."
        if signature == ("style_transfer", "cross_sentence_scatter", "implication_rewrite"):
            return f"{entity} is the target. {scope} is the workspace. What I need is the {attribute} facet, and the operation should stay on that target rather than a neighboring workflow."
        if signature == ("slot_implicitization", "partial_constraint_drop"):
            return f"I need the relevant {attribute} handling in {scope}; the target is implied by the surrounding note as {entity}, though not stated in the usual slot order."
        if signature == ("near_semantic_distractor", "side_goal_insertion"):
            return f"{base} A nearby workflow in the same workspace also sounds plausible, but it should not override the requested operation."
        if signature == ("near_semantic_distractor", "side_goal_insertion", "soft_topic_drift"):
            return f"I keep circling around adjacent workflows in {scope}, then the request for {attribute} on {entity} comes back into view, and the two keep blending together."
        if signature == ("contradictory_constraint", "late_revision", "intra_prompt_conflict"):
            return f"I need {attribute} for {entity} in {scope}, except I also mention a neighboring operation, and by the end I make the scope sound like {cls._alternate_value(scope)} instead."

        text = base
        for operator_name in operators:
            text = cls._apply_operator(text, operator_name, args, "")
        return text

    @classmethod
    def _apply_user_intent_operator(cls, text: str, tool_id: str, args: dict[str, Any], operator_name: str) -> str:
        if operator_name == "paraphrase":
            return cls._simple_paraphrase(text)
        if operator_name == "style_transfer":
            return cls._style_transfer_request(tool_id, args)
        if operator_name == "constraint_reposition":
            return cls._constraint_reposition_request(tool_id, args)
        if operator_name == "cross_sentence_scatter":
            return cls._cross_sentence_scatter_request(tool_id, args)
        if operator_name == "implication_rewrite":
            return cls._implication_rewrite_request(tool_id, args)
        if operator_name == "slot_implicitization":
            return cls._slot_implicitization_request(tool_id, args)
        if operator_name == "vague_quantity":
            return cls._vague_quantity_request(tool_id, args)
        if operator_name == "omitted_modifier":
            return cls._omitted_modifier_request(tool_id, args)
        if operator_name == "partial_constraint_drop":
            return cls._partial_constraint_drop_request(tool_id, args)
        if operator_name == "redundant_context":
            return cls._redundant_context_request(tool_id, args)
        if operator_name == "near_semantic_distractor":
            return cls._near_semantic_distractor_request(tool_id, args)
        if operator_name == "side_goal_insertion":
            return cls._side_goal_insertion_request(tool_id, args)
        if operator_name == "soft_topic_drift":
            return cls._soft_topic_drift_request(tool_id, args)
        if operator_name == "contradictory_constraint":
            return cls._contradictory_constraint_request(tool_id, args)
        if operator_name == "late_revision":
            return cls._late_revision_request(tool_id, args)
        if operator_name == "intra_prompt_conflict":
            return cls._intra_prompt_conflict_request(tool_id, args)
        if operator_name == "ambiguity_probe":
            return cls._ambiguity_probe_request(tool_id, args)
        raise ValueError(f"Unsupported user intent operator: {operator_name}")

    @staticmethod
    def _format_items(items: list[Any]) -> str:
        return ", ".join(str(item) for item in items)

    @staticmethod
    def _alternate_value(value: Any) -> Any:
        replacements = {
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
            "drive": "walk",
            "walk": "metro",
            "metro": "drive",
            "bus": "bike",
            "bike": "bus",
            "c": "f",
            "f": "c",
            "usd": "eur",
            "eur": "usd",
            "cny": "usd",
            "gbp": "usd",
            "km": "m",
            "m": "km",
            "kg": "g",
            "g": "kg",
            "l": "ml",
            "ml": "l",
        }
        return replacements.get(str(value), value)

    @classmethod
    def _style_transfer_request(cls, tool_id: str, args: dict[str, Any]) -> str:
        if tool_id == "search_document":
            return f"Could you point me to something in the {args['scope']} collection about {args['query']} that reads like regular documentation, not a policy memo or a FAQ answer?"
        if tool_id == "search_policy":
            return f"Could you point me to the {args['scope']} rule or policy that covers {args['query']}? I mean the governing material, not a general document or a FAQ."
        if tool_id == "search_faq":
            return f"Could you find the help-style answer in {args['scope']} about {args['query']}? I am after the FAQ-type guidance, not the formal policy or a general document."
        if tool_id == "calculate_route":
            return f"Could you help me figure out how to get from {args['origin']} to {args['target']} by {args['profile']}? I need the route itself, not only the time."
        if tool_id == "estimate_travel_time":
            return f"Could you help me gauge how long it takes to get from {args['origin']} to {args['target']} by {args['profile']}? I only need the time."
        if tool_id == "sort_items":
            return f"Could you put [{cls._format_items(args['items'])}] into {args['rule']} order for me? Keep the whole list; I do not want a filtered subset."
        if tool_id == "filter_items":
            return f"Could you keep only the entries in [{cls._format_items(args['items'])}] that match '{args['rule']}'? I want a subset, not a reorder of everything."
        if tool_id == "convert_unit":
            return f"Could you convert {args['value']} {args['source']} into {args['target']} for me on the measurement side, not the money side?"
        if tool_id == "convert_currency":
            return f"Could you convert {args['value']} {str(args['source']).upper()} into {str(args['target']).upper()} for me as money, not as a measurement conversion?"
        if tool_id == "get_weather":
            return f"Could you tell me what the overall weather is like in {args['place']} {args['slot']}? I am after the outdoor conditions, not the air-quality number or just the temperature."
        if tool_id == "get_air_quality":
            return f"Could you tell me what the air quality is like in {args['place']} {args['slot']}? I am after the AQI-style reading, not the overall weather or the temperature alone."
        if tool_id == "get_temperature":
            return f"Could you tell me the temperature in {str(args['scale']).upper()} for {args['place']} {args['slot']}? I only need the numeric reading, not the broader weather or AQI summary."
        raise ValueError(f"Unsupported tool for style transfer: {tool_id}")

    @classmethod
    def _medium_indirect_request(cls, tool_id: str, args: dict[str, Any]) -> str:
        if tool_id == "search_document":
            return f"Could you help with something in the {args['scope']} collection about {args['query']}? The part that matters, once the request settles, is that I mean the regular document rather than the rule page or FAQ."
        if tool_id == "search_policy":
            return f"Could you help with something in the {args['scope']} collection about {args['query']}? The part that matters, once the request settles, is that I mean the policy rather than the general document or FAQ."
        if tool_id == "search_faq":
            return f"Could you help with something in the {args['scope']} collection about {args['query']}? The part that matters, once the request settles, is that I mean the FAQ-style answer rather than the policy or general document."
        if tool_id == "calculate_route":
            return f"Could you help with getting from {args['origin']} to {args['target']} by {args['profile']}? The part that matters, once the request settles, is that I mean the route itself rather than only the travel time."
        if tool_id == "estimate_travel_time":
            return f"Could you help with getting from {args['origin']} to {args['target']} by {args['profile']}? The part that matters, once the request settles, is that I mean the travel time rather than the route details."
        if tool_id == "sort_items":
            return f"Could you help with [{cls._format_items(args['items'])}]? The part that matters, once the request settles, is that I want the full set back in {args['rule']} order rather than a filtered subset."
        if tool_id == "filter_items":
            return f"Could you help with [{cls._format_items(args['items'])}]? The part that matters, once the request settles, is that I want only the entries that match '{args['rule']}' rather than a full reorder."
        if tool_id == "convert_unit":
            return f"Could you help with {args['value']} {args['source']} into {args['target']}? The part that matters, once the request settles, is that I mean the measurement conversion rather than the money version."
        if tool_id == "convert_currency":
            return f"Could you help with {args['value']} {str(args['source']).upper()} into {str(args['target']).upper()}? The part that matters, once the request settles, is that I mean the money conversion rather than the measurement version."
        if tool_id == "get_weather":
            return f"Could you help with {args['place']} {args['slot']}? The part that matters, once the request settles, is that I mean the overall weather rather than only AQI or temperature."
        if tool_id == "get_air_quality":
            return f"Could you help with {args['place']} {args['slot']}? The part that matters, once the request settles, is that I mean the AQI-style reading rather than the broader weather summary."
        if tool_id == "get_temperature":
            return f"Could you help with {args['place']} {args['slot']}? The part that matters, once the request settles, is that I mean the temperature in {str(args['scale']).upper()} rather than the broader weather summary."
        raise ValueError(f"Unsupported tool for medium indirect rewrite: {tool_id}")

    @classmethod
    def _heavy_indirect_request(cls, tool_id: str, args: dict[str, Any]) -> str:
        if tool_id == "search_document":
            return f"{str(args['query']).capitalize()} is the topic. {str(args['scope']).capitalize()} is where I am looking. I need the ordinary write-up, not the rule page and not the help-answer version."
        if tool_id == "search_policy":
            return f"{str(args['query']).capitalize()} is the topic. {str(args['scope']).capitalize()} is the part of the org I am looking inside. I need the thing that tells people what the rule is, not a general explainer or a help answer."
        if tool_id == "search_faq":
            return f"{str(args['query']).capitalize()} is the issue. {str(args['scope']).capitalize()} is where it lives. I want the answer somebody would follow, not the policy text and not the general write-up."
        if tool_id == "calculate_route":
            return f"The trip is from {args['origin']} to {args['target']}. By {args['profile']}. What I need is the actual way to go, not merely the duration."
        if tool_id == "estimate_travel_time":
            return f"The trip is from {args['origin']} to {args['target']}. By {args['profile']}. What matters is how long it should take, not the route breakdown."
        if tool_id == "sort_items":
            return f"The items are [{cls._format_items(args['items'])}]. Use the {args['rule']} rule. I need the whole set rearranged rather than reduced."
        if tool_id == "filter_items":
            return f"The items are [{cls._format_items(args['items'])}]. The clue is '{args['rule']}'. I only want the matching entries, not the full list rearranged."
        if tool_id == "convert_unit":
            return f"{args['value']} is the amount. {args['source']} is where it starts. {args['target']} is where it should land. This is the measurement conversion side, not money."
        if tool_id == "convert_currency":
            return f"{args['value']} is the amount. {str(args['source']).upper()} is where it starts. {str(args['target']).upper()} is where it should land. This is the money-conversion side, not the measurement one."
        if tool_id == "get_weather":
            return f"{args['place']} is the place. {args['slot']} is the time. I need the overall outside picture rather than the AQI reading or the isolated temperature number."
        if tool_id == "get_air_quality":
            return f"{args['place']} is the place. {args['slot']} is the time. I need the AQI-style reading rather than the overall weather summary or the isolated temperature number."
        if tool_id == "get_temperature":
            return f"{args['place']} is the place. {args['slot']} is the time. I need the numeric temperature in {str(args['scale']).upper()}, not the broader weather summary or the AQI reading."
        raise ValueError(f"Unsupported tool for heavy indirect rewrite: {tool_id}")

    @classmethod
    def _constraint_reposition_request(cls, tool_id: str, args: dict[str, Any]) -> str:
        if tool_id == "search_document":
            return f"I am looking into {args['query']} in the {args['scope']} collection. What matters, once you reach the end of the request, is that it should read like regular documentation."
        if tool_id == "search_policy":
            return f"I am looking into {args['query']} in the {args['scope']} collection. What matters, once you reach the end of the request, is that it should be the governing rule or policy."
        if tool_id == "search_faq":
            return f"I am looking into {args['query']} in the {args['scope']} collection. What matters, once you reach the end of the request, is that it should be the help-answer version."
        if tool_id == "calculate_route":
            return f"I need to get from {args['origin']} to {args['target']}. The part that matters at the end is that I want the route itself, using {args['profile']}."
        if tool_id == "estimate_travel_time":
            return f"I need to get from {args['origin']} to {args['target']}. The part that matters at the end is that I only want the duration, using {args['profile']}."
        if tool_id == "sort_items":
            return f"I need something done with [{cls._format_items(args['items'])}]. The important part at the end is that they should come back in {args['rule']} order."
        if tool_id == "filter_items":
            return f"I need something done with [{cls._format_items(args['items'])}]. The important part at the end is that only entries matching '{args['rule']}' should remain."
        if tool_id == "convert_unit":
            return f"I need something done with {args['value']} {args['source']}. The important part at the end is that it should land in {args['target']} as a measurement conversion."
        if tool_id == "convert_currency":
            return f"I need something done with {args['value']} {str(args['source']).upper()}. The important part at the end is that it should land in {str(args['target']).upper()} as a money conversion."
        if tool_id == "get_weather":
            return f"I need something about {args['place']} {args['slot']}. The important part at the end is that I want the overall weather, not the AQI or the isolated temperature."
        if tool_id == "get_air_quality":
            return f"I need something about {args['place']} {args['slot']}. The important part at the end is that I want the AQI reading, not the broader weather or just the temperature."
        if tool_id == "get_temperature":
            return f"I need something about {args['place']} {args['slot']}. The important part at the end is that I want the temperature in {str(args['scale']).upper()}."
        raise ValueError(f"Unsupported tool for constraint reposition: {tool_id}")

    @classmethod
    def _cross_sentence_scatter_request(cls, tool_id: str, args: dict[str, Any]) -> str:
        if tool_id == "search_document":
            return f"{str(args['query']).capitalize()} is the topic. The collection is {args['scope']}. I want the ordinary document, not the policy page or the FAQ."
        if tool_id == "search_policy":
            return f"{str(args['query']).capitalize()} is the topic. The collection is {args['scope']}. I want the governing rule, not the general document and not the FAQ."
        if tool_id == "search_faq":
            return f"{str(args['query']).capitalize()} is the topic. The collection is {args['scope']}. I want the help answer, not the policy and not the general document."
        if tool_id == "calculate_route":
            return f"The trip starts at {args['origin']}. It ends at {args['target']}. {str(args['profile']).capitalize()} is the mode. I need the route itself."
        if tool_id == "estimate_travel_time":
            return f"The trip starts at {args['origin']}. It ends at {args['target']}. {str(args['profile']).capitalize()} is the mode. I only need the duration."
        if tool_id == "sort_items":
            return f"The items are [{cls._format_items(args['items'])}]. The rule is {args['rule']}. Keep every item, just reorder them."
        if tool_id == "filter_items":
            return f"The items are [{cls._format_items(args['items'])}]. The clue is '{args['rule']}'. Keep only the entries that match."
        if tool_id == "convert_unit":
            return f"The amount is {args['value']}. It starts in {args['source']}. It should end in {args['target']}. This is the measurement version."
        if tool_id == "convert_currency":
            return f"The amount is {args['value']}. It starts in {str(args['source']).upper()}. It should end in {str(args['target']).upper()}. This is the money version."
        if tool_id == "get_weather":
            return f"{args['place']} is the place. {args['slot']} is the time. I want the overall weather."
        if tool_id == "get_air_quality":
            return f"{args['place']} is the place. {args['slot']} is the time. I want the AQI-style reading."
        if tool_id == "get_temperature":
            return f"{args['place']} is the place. {args['slot']} is the time. The reading should be in {str(args['scale']).upper()}. I only want the temperature."
        raise ValueError(f"Unsupported tool for cross sentence scatter: {tool_id}")

    @classmethod
    def _implication_rewrite_request(cls, tool_id: str, args: dict[str, Any]) -> str:
        if tool_id == "search_document":
            return f"I need the version someone would read as the normal write-up on {args['query']} in {args['scope']}, not the rule text and not the help answer."
        if tool_id == "search_policy":
            return f"I need the thing in {args['scope']} that settles what the rule is for {args['query']}, not a general explainer and not a FAQ."
        if tool_id == "search_faq":
            return f"I need the answer someone would follow in {args['scope']} for {args['query']}, not the formal policy text and not the general write-up."
        if tool_id == "calculate_route":
            return f"I am trying to figure out which way to go from {args['origin']} to {args['target']} on {args['profile']}, not just how many minutes it takes."
        if tool_id == "estimate_travel_time":
            return f"I am trying to gauge how long the {args['profile']} trip from {args['origin']} to {args['target']} should take, not map the route."
        if tool_id == "sort_items":
            return f"I need [{cls._format_items(args['items'])}] to come back arranged under the {args['rule']} rule, with nothing removed."
        if tool_id == "filter_items":
            return f"I need only the entries from [{cls._format_items(args['items'])}] that fit '{args['rule']}', not the whole set rearranged."
        if tool_id == "convert_unit":
            return f"I need to see what {args['value']} {args['source']} turns into in {args['target']} on the measurement side."
        if tool_id == "convert_currency":
            return f"I need to see what {args['value']} {str(args['source']).upper()} turns into in {str(args['target']).upper()} on the money side."
        if tool_id == "get_weather":
            return f"I need the outside picture for {args['place']} {args['slot']}, not the AQI reading or the isolated temperature number."
        if tool_id == "get_air_quality":
            return f"I need the AQI-style reading for {args['place']} {args['slot']}, not the overall weather summary or the isolated temperature number."
        if tool_id == "get_temperature":
            return f"I need the numeric temperature in {str(args['scale']).upper()} for {args['place']} {args['slot']}, not the broader weather or AQI summary."
        raise ValueError(f"Unsupported tool for implication rewrite: {tool_id}")

    @classmethod
    def _slot_implicitization_request(cls, tool_id: str, args: dict[str, Any]) -> str:
        if tool_id == "search_document":
            return f"I am looking for the ordinary write-up about {args['query']} on the {args['scope']} side."
        if tool_id == "search_policy":
            return f"I am looking for the rule that governs {args['query']} on the {args['scope']} side."
        if tool_id == "search_faq":
            return f"I am looking for the answer people follow for {args['query']} in {args['scope']}."
        if tool_id == "calculate_route":
            return f"I need the way to get from {args['origin']} to {args['target']}; I mean the {args['profile']} option rather than another mode."
        if tool_id == "estimate_travel_time":
            return f"About how long should the {args['profile']} run from {args['origin']} to {args['target']} take?"
        if tool_id == "sort_items":
            return f"Please line [{cls._format_items(args['items'])}] up in {args['rule']} order."
        if tool_id == "filter_items":
            return f"Please keep only the entries from [{cls._format_items(args['items'])}] that fit '{args['rule']}'."
        if tool_id == "convert_unit":
            return f"What does {args['value']} {args['source']} come out to in {args['target']} on the measurement side?"
        if tool_id == "convert_currency":
            return f"What does {args['value']} {str(args['source']).upper()} come out to in {str(args['target']).upper()} on the money side?"
        if tool_id == "get_weather":
            return f"What is it like outside in {args['place']} {args['slot']}?"
        if tool_id == "get_air_quality":
            return f"How is the air side in {args['place']} {args['slot']}?"
        if tool_id == "get_temperature":
            return f"What is the {str(args['scale']).upper()} reading for {args['place']} {args['slot']}?"
        raise ValueError(f"Unsupported tool for slot implicitization: {tool_id}")

    @classmethod
    def _vague_quantity_request(cls, tool_id: str, args: dict[str, Any]) -> str:
        if tool_id == "convert_unit":
            return f"Convert roughly {args['value']} {args['source']} into {args['target']} for me on the measurement side."
        if tool_id == "convert_currency":
            return f"Convert about {args['value']} {str(args['source']).upper()} into {str(args['target']).upper()} for me on the money side."
        if tool_id == "sort_items":
            return f"Please put [{cls._format_items(args['items'])}] into a clearly {args['rule']}-leaning order without dropping anything."
        if tool_id == "filter_items":
            return f"Please keep the entries from [{cls._format_items(args['items'])}] that roughly fit '{args['rule']}'."
        return cls._render_request(tool_id, args, 0)

    @classmethod
    def _omitted_modifier_request(cls, tool_id: str, args: dict[str, Any]) -> str:
        if tool_id in {"search_document", "search_policy", "search_faq"}:
            return f"Find something in the {args['scope']} collection about {args['query']}."
        if tool_id in {"calculate_route", "estimate_travel_time"}:
            return f"Help me with the {args['profile']} trip from {args['origin']} to {args['target']}."
        if tool_id == "sort_items":
            return f"Handle [{cls._format_items(args['items'])}] with the {args['rule']} rule."
        if tool_id == "filter_items":
            return f"Handle [{cls._format_items(args['items'])}] with the '{args['rule']}' rule."
        if tool_id == "convert_unit":
            return f"Convert {args['value']} {args['source']} into {args['target']}."
        if tool_id == "convert_currency":
            return f"Convert {args['value']} {str(args['source']).upper()} into {str(args['target']).upper()}."
        if tool_id == "get_temperature":
            return f"Tell me about {args['place']} {args['slot']} in {str(args['scale']).upper()}."
        if tool_id in {"get_weather", "get_air_quality"}:
            return f"Tell me about {args['place']} {args['slot']}."
        raise ValueError(f"Unsupported tool for omitted modifier: {tool_id}")

    @classmethod
    def _partial_constraint_drop_request(cls, tool_id: str, args: dict[str, Any]) -> str:
        if tool_id in {"search_document", "search_policy", "search_faq"}:
            return f"I need something in {args['scope']} about {args['query']}."
        if tool_id in {"calculate_route", "estimate_travel_time"}:
            return f"I need to get from {args['origin']} to {args['target']} by {args['profile']}."
        if tool_id == "sort_items":
            return f"Please handle [{cls._format_items(args['items'])}] with the {args['rule']} rule."
        if tool_id == "filter_items":
            return f"Please handle [{cls._format_items(args['items'])}] with the '{args['rule']}' rule."
        if tool_id == "convert_unit":
            return f"Convert {args['value']} {args['source']} into {args['target']} for me."
        if tool_id == "convert_currency":
            return f"Convert {args['value']} {str(args['source']).upper()} into {str(args['target']).upper()} for me."
        return f"Tell me about {args['place']} {args['slot']}."

    @classmethod
    def _heavy_underspecified_request(cls, tool_id: str, args: dict[str, Any]) -> str:
        if tool_id == "search_document":
            return f"I need something on {args['query']} from the {args['scope']} side."
        if tool_id == "search_policy":
            return f"I need whatever in {args['scope']} governs {args['query']}."
        if tool_id == "search_faq":
            return f"I need the answer on {args['query']} from {args['scope']}."
        if tool_id in {"calculate_route", "estimate_travel_time"}:
            return f"I need to get from {args['origin']} to {args['target']} on {args['profile']}."
        if tool_id == "sort_items":
            return f"Please do the {args['rule']} handling on [{cls._format_items(args['items'])}]."
        if tool_id == "filter_items":
            return f"Please do the '{args['rule']}' handling on [{cls._format_items(args['items'])}]."
        if tool_id == "convert_unit":
            return f"Handle {args['value']} {args['source']} into {args['target']} for me."
        if tool_id == "convert_currency":
            return f"Handle {args['value']} {str(args['source']).upper()} into {str(args['target']).upper()} for me."
        return f"How is {args['place']} {args['slot']} looking?"

    @classmethod
    def _redundant_context_request(cls, tool_id: str, args: dict[str, Any]) -> str:
        return f"I am organizing a quick working note before a meeting. {cls._style_transfer_request(tool_id, args)}"

    @classmethod
    def _near_semantic_distractor_request(cls, tool_id: str, args: dict[str, Any]) -> str:
        if tool_id == "search_document":
            return f"I started by thinking about the policy page on '{args['query']}' in {args['scope']}, but the item I need returned is the regular document."
        if tool_id == "search_policy":
            return f"I started by thinking about the general document on '{args['query']}' in {args['scope']}, but the item I need returned is the policy that governs it."
        if tool_id == "search_faq":
            return f"I started by thinking about the policy text on '{args['query']}' in {args['scope']}, but the item I need returned is the FAQ-style answer."
        if tool_id == "calculate_route":
            return f"I started by thinking about the ETA for getting from {args['origin']} to {args['target']} by {args['profile']}, but the item I need returned is the route itself."
        if tool_id == "estimate_travel_time":
            return f"I started by thinking about the route for going from {args['origin']} to {args['target']} by {args['profile']}, but the item I need returned is only the time."
        if tool_id == "sort_items":
            return f"I started by thinking about filtering [{cls._format_items(args['items'])}], but the item I need returned is the full list in {args['rule']} order."
        if tool_id == "filter_items":
            return f"I started by thinking about reordering [{cls._format_items(args['items'])}], but the item I need returned is only the entries that match '{args['rule']}'."
        if tool_id == "convert_unit":
            return f"I started by thinking about the money-conversion version, but the item I need returned is {args['value']} {args['source']} into {args['target']} as a measurement."
        if tool_id == "convert_currency":
            return f"I started by thinking about the measurement-conversion version, but the item I need returned is {args['value']} {str(args['source']).upper()} into {str(args['target']).upper()} as money."
        if tool_id == "get_weather":
            return f"I started by thinking about the AQI in {args['place']} {args['slot']}, but the item I need returned is the overall weather."
        if tool_id == "get_air_quality":
            return f"I started by thinking about the broader weather in {args['place']} {args['slot']}, but the item I need returned is the AQI-style reading."
        if tool_id == "get_temperature":
            return f"I started by thinking about the broader weather in {args['place']} {args['slot']}, but the item I need returned is the temperature in {str(args['scale']).upper()}."
        raise ValueError(f"Unsupported tool for semantic distractor: {tool_id}")

    @classmethod
    def _side_goal_insertion_request(cls, tool_id: str, args: dict[str, Any]) -> str:
        if tool_id == "search_document":
            return f"I need the regular document on {args['query']} in {args['scope']}. If there is a policy or FAQ nearby, that can stay secondary."
        if tool_id == "search_policy":
            return f"I need the policy on {args['query']} in {args['scope']}. If there is a readable general document nearby, that can stay secondary."
        if tool_id == "search_faq":
            return f"I need the FAQ answer on {args['query']} in {args['scope']}. If there is a policy or document nearby, that can stay secondary."
        if tool_id == "calculate_route":
            return f"I need the route from {args['origin']} to {args['target']} by {args['profile']}. If a rough ETA also comes to mind, that part is secondary."
        if tool_id == "estimate_travel_time":
            return f"I need the travel time from {args['origin']} to {args['target']} by {args['profile']}. If the route shape comes to mind, that part is secondary."
        if tool_id == "sort_items":
            return f"I need [{cls._format_items(args['items'])}] in {args['rule']} order. If a matching subset is obvious, that part stays secondary."
        if tool_id == "filter_items":
            return f"I need the matching subset from [{cls._format_items(args['items'])}] using '{args['rule']}'. If a full reorder is obvious, that part stays secondary."
        if tool_id == "convert_unit":
            return f"I need {args['value']} {args['source']} into {args['target']} as a measurement conversion. If the money analogue crosses your mind, that part is secondary."
        if tool_id == "convert_currency":
            return f"I need {args['value']} {str(args['source']).upper()} into {str(args['target']).upper()} as a money conversion. If the measurement analogue crosses your mind, that part is secondary."
        if tool_id == "get_weather":
            return f"I need the overall weather in {args['place']} {args['slot']}. If the AQI or exact temperature comes to mind, that stays secondary."
        if tool_id == "get_air_quality":
            return f"I need the AQI-style reading in {args['place']} {args['slot']}. If the broader weather or exact temperature comes to mind, that stays secondary."
        if tool_id == "get_temperature":
            return f"I need the temperature in {str(args['scale']).upper()} for {args['place']} {args['slot']}. If the broader weather or AQI comes to mind, that stays secondary."
        raise ValueError(f"Unsupported tool for side goal insertion: {tool_id}")

    @classmethod
    def _medium_distraction_request(cls, tool_id: str, args: dict[str, Any]) -> str:
        if tool_id == "search_document":
            return f"I started by thinking about the policy page on '{args['query']}' in {args['scope']}, and if it gives quick context it can stay secondary. The item I need returned is the regular document."
        if tool_id == "search_policy":
            return f"I started by thinking about the general document on '{args['query']}' in {args['scope']}, and if it gives quick context it can stay secondary. The item I need returned is the policy that governs it."
        if tool_id == "search_faq":
            return f"I started by thinking about the policy text on '{args['query']}' in {args['scope']}, and if it gives quick context it can stay secondary. The item I need returned is the FAQ-style answer."
        if tool_id == "calculate_route":
            return f"I started by thinking about the ETA for getting from {args['origin']} to {args['target']} by {args['profile']}, and if that estimate comes to mind it can stay secondary. The item I need returned is the route itself."
        if tool_id == "estimate_travel_time":
            return f"I started by thinking about the route from {args['origin']} to {args['target']} by {args['profile']}, and if that outline comes to mind it can stay secondary. The item I need returned is only the time."
        if tool_id == "sort_items":
            return f"I started by thinking about filtering [{cls._format_items(args['items'])}], and if a matching subset is obvious it can stay secondary. The item I need returned is the full list in {args['rule']} order."
        if tool_id == "filter_items":
            return f"I started by thinking about reordering [{cls._format_items(args['items'])}], and if a clean order is obvious it can stay secondary. The item I need returned is only the entries that match '{args['rule']}'."
        if tool_id == "convert_unit":
            return f"I started by thinking about the money-conversion version, and if that analogy comes to mind it can stay secondary. The item I need returned is {args['value']} {args['source']} into {args['target']} as a measurement."
        if tool_id == "convert_currency":
            return f"I started by thinking about the measurement-conversion version, and if that analogy comes to mind it can stay secondary. The item I need returned is {args['value']} {str(args['source']).upper()} into {str(args['target']).upper()} as money."
        if tool_id == "get_weather":
            return f"I started by thinking about the AQI in {args['place']} {args['slot']}, and if that comes up it can stay secondary. The item I need returned is the overall weather."
        if tool_id == "get_air_quality":
            return f"I started by thinking about the broader weather in {args['place']} {args['slot']}, and if that comes up it can stay secondary. The item I need returned is the AQI-style reading."
        if tool_id == "get_temperature":
            return f"I started by thinking about the broader weather in {args['place']} {args['slot']}, and if that comes up it can stay secondary. The item I need returned is the temperature in {str(args['scale']).upper()}."
        raise ValueError(f"Unsupported tool for medium distraction: {tool_id}")

    @classmethod
    def _soft_topic_drift_request(cls, tool_id: str, args: dict[str, Any]) -> str:
        if tool_id == "search_document":
            return f"I drifted through the policy and FAQ material on {args['query']} in {args['scope']}, and the regular document stayed mixed into the same thread instead of standing apart."
        if tool_id == "search_policy":
            return f"I drifted through the general docs and FAQ material on {args['query']} in {args['scope']}, and the policy stayed mixed into the same thread instead of standing apart."
        if tool_id == "search_faq":
            return f"I drifted through the document and policy material on {args['query']} in {args['scope']}, and the FAQ answer stayed mixed into the same thread instead of standing apart."
        if tool_id == "calculate_route":
            return f"I drifted toward how long the trip from {args['origin']} to {args['target']} by {args['profile']} would take, and the route itself stayed mixed into the same thread instead of standing apart."
        if tool_id == "estimate_travel_time":
            return f"I drifted toward mapping the trip from {args['origin']} to {args['target']} by {args['profile']}, and the time estimate stayed mixed into the same thread instead of standing apart."
        if tool_id == "sort_items":
            return f"I drifted toward filtering [{cls._format_items(args['items'])}], and the full reorder in {args['rule']} stayed mixed into the same thread instead of standing apart."
        if tool_id == "filter_items":
            return f"I drifted toward reordering [{cls._format_items(args['items'])}], and the matching subset for '{args['rule']}' stayed mixed into the same thread instead of standing apart."
        if tool_id == "convert_unit":
            return f"I drifted toward the currency version of this conversion, and the measurement reading of {args['value']} {args['source']} into {args['target']} stayed mixed into the same thread instead of standing apart."
        if tool_id == "convert_currency":
            return f"I drifted toward the measurement version of this conversion, and the money reading of {args['value']} {str(args['source']).upper()} into {str(args['target']).upper()} stayed mixed into the same thread instead of standing apart."
        if tool_id == "get_weather":
            return f"I drifted toward the AQI and temperature for {args['place']} {args['slot']}, and the overall weather stayed mixed into the same thread instead of standing apart."
        if tool_id == "get_air_quality":
            return f"I drifted toward the broader weather and temperature for {args['place']} {args['slot']}, and the AQI-style reading stayed mixed into the same thread instead of standing apart."
        if tool_id == "get_temperature":
            return f"I drifted toward the broader weather and AQI for {args['place']} {args['slot']}, and the temperature in {str(args['scale']).upper()} stayed mixed into the same thread instead of standing apart."
        raise ValueError(f"Unsupported tool for soft topic drift: {tool_id}")

    @classmethod
    def _heavy_distraction_request(cls, tool_id: str, args: dict[str, Any]) -> str:
        if tool_id == "search_document":
            return f"I keep circling around the policy page on {args['query']} in {args['scope']}, then the FAQ seems useful too, and the regular document is mixed in as another piece for the same note."
        if tool_id == "search_policy":
            return f"I keep circling around the general docs on {args['query']} in {args['scope']}, then the FAQ seems useful too, and the policy is mixed in as another piece for the same note."
        if tool_id == "search_faq":
            return f"I keep circling around the general docs on {args['query']} in {args['scope']}, then the policy text seems useful too, and the FAQ answer is mixed in as another piece for the same note."
        if tool_id == "calculate_route":
            return f"I keep circling around how long the {args['profile']} trip from {args['origin']} to {args['target']} would take, then the route itself becomes relevant again, and the two keep blending together."
        if tool_id == "estimate_travel_time":
            return f"I keep circling around sketching the route from {args['origin']} to {args['target']} by {args['profile']}, then the time estimate becomes relevant again, and the two keep blending together."
        if tool_id == "sort_items":
            return f"I keep circling around filtering [{cls._format_items(args['items'])}], then the full reorder in {args['rule']} comes back into view, and the two keep blending together."
        if tool_id == "filter_items":
            return f"I keep circling around reordering [{cls._format_items(args['items'])}], then the matching subset for '{args['rule']}' comes back into view, and the two keep blending together."
        if tool_id == "convert_unit":
            return f"I keep circling around the money-conversion version of this request, then the measurement reading of {args['value']} {args['source']} into {args['target']} comes back into view, and the two keep blending together."
        if tool_id == "convert_currency":
            return f"I keep circling around the measurement-conversion version of this request, then the money reading of {args['value']} {str(args['source']).upper()} into {str(args['target']).upper()} comes back into view, and the two keep blending together."
        if tool_id == "get_weather":
            return f"I keep circling around the AQI and the temperature for {args['place']} {args['slot']}, and the overall weather stays mixed into the same thread."
        if tool_id == "get_air_quality":
            return f"I keep circling around the broader weather and the temperature for {args['place']} {args['slot']}, and the AQI-style reading stays mixed into the same thread."
        if tool_id == "get_temperature":
            return f"I keep circling around the broader weather and the AQI for {args['place']} {args['slot']}, and the temperature in {str(args['scale']).upper()} stays mixed into the same thread."
        raise ValueError(f"Unsupported tool for heavy distraction: {tool_id}")

    @classmethod
    def _ambiguity_probe_request(cls, tool_id: str, args: dict[str, Any]) -> str:
        if tool_id in {"search_document", "search_policy", "search_faq"}:
            if tool_id == "search_document":
                return f"I need something in {args['scope']} about '{args['query']}', probably the regular document rather than the rule page or FAQ."
            if tool_id == "search_policy":
                return f"I need something in {args['scope']} about '{args['query']}', probably the policy rather than the general document or FAQ."
            return f"I need something in {args['scope']} about '{args['query']}', probably the FAQ-style answer rather than the policy or general document."
        if tool_id in {"calculate_route", "estimate_travel_time"}:
            if tool_id == "calculate_route":
                return f"Help me with getting from {args['origin']} to {args['target']} by {args['profile']}, probably the route itself rather than just the time."
            return f"Help me with getting from {args['origin']} to {args['target']} by {args['profile']}, probably just the time rather than the route details."
        if tool_id == "sort_items":
            return f"Please do the {args['rule']} handling on [{cls._format_items(args['items'])}], probably as a full reorder rather than a filtered subset."
        if tool_id == "filter_items":
            return f"Please do the '{args['rule']}' handling on [{cls._format_items(args['items'])}], probably as a filtered subset rather than a full reorder."
        if tool_id == "convert_unit":
            return f"Handle {args['value']} {args['source']} into {args['target']} for me, probably as a measurement conversion rather than the money version."
        if tool_id == "convert_currency":
            return f"Handle {args['value']} {str(args['source']).upper()} into {str(args['target']).upper()} for me, probably as a money conversion rather than the measurement version."
        if tool_id in {"get_weather", "get_air_quality", "get_temperature"}:
            if tool_id == "get_weather":
                return f"How is {args['place']} {args['slot']} looking, probably in the overall weather sense rather than just AQI or temperature?"
            if tool_id == "get_air_quality":
                return f"How is {args['place']} {args['slot']} looking, probably in the AQI sense rather than the broader weather summary?"
            return f"How is {args['place']} {args['slot']} looking, probably in terms of the temperature reading rather than the broader weather summary?"
        raise ValueError(f"Unsupported tool for ambiguity probe: {tool_id}")

    @classmethod
    def _contradictory_constraint_request(cls, tool_id: str, args: dict[str, Any]) -> str:
        if tool_id == "search_document":
            return f"I need the regular document on {args['query']} in {args['scope']}, but another part of the request keeps asking for the rule page instead."
        if tool_id == "search_policy":
            return f"I need the policy on {args['query']} in {args['scope']}, but another part of the request keeps asking for the general document instead."
        if tool_id == "search_faq":
            return f"I need the FAQ-style answer on {args['query']} in {args['scope']}, but another part of the request keeps asking for the formal policy instead."
        if tool_id == "calculate_route":
            return f"I need the route from {args['origin']} to {args['target']} by {args['profile']}, but another part of the request says just the travel time is enough."
        if tool_id == "estimate_travel_time":
            return f"I need the travel time from {args['origin']} to {args['target']} by {args['profile']}, but another part of the request says the route itself is what matters."
        if tool_id == "sort_items":
            return f"Reorder [{cls._format_items(args['items'])}] in {args['rule']} order, but another part of the request says to keep only the matching subset."
        if tool_id == "filter_items":
            return f"Keep only the entries from [{cls._format_items(args['items'])}] that match '{args['rule']}', but another part of the request says to reorder the whole list instead."
        if tool_id == "convert_unit":
            return f"Convert {args['value']} {args['source']} into {args['target']}, but another part of the request makes it sound like the money-conversion version."
        if tool_id == "convert_currency":
            return f"Convert {args['value']} {str(args['source']).upper()} into {str(args['target']).upper()}, but another part of the request makes it sound like the measurement-conversion version."
        if tool_id == "get_weather":
            return f"Tell me the overall weather in {args['place']} {args['slot']}, but another part of the request sounds like I only want the AQI."
        if tool_id == "get_air_quality":
            return f"Tell me the AQI-style reading in {args['place']} {args['slot']}, but another part of the request sounds like I only want the broader weather."
        if tool_id == "get_temperature":
            return f"Tell me the temperature in {str(args['scale']).upper()} for {args['place']} {args['slot']}, but another part of the request sounds like I only want the broader weather."
        raise ValueError(f"Unsupported tool for contradictory constraint: {tool_id}")

    @classmethod
    def _late_revision_request(cls, tool_id: str, args: dict[str, Any]) -> str:
        if tool_id in {"search_document", "search_policy", "search_faq"}:
            return f"{cls._ambiguity_probe_request(tool_id, args)} Actually, near the end I switch the collection to {cls._alternate_value(args['scope'])}."
        if tool_id in {"calculate_route", "estimate_travel_time"}:
            return f"{cls._ambiguity_probe_request(tool_id, args)} Actually, near the end I switch the starting point to {cls._alternate_value(args['origin'])}."
        if tool_id == "sort_items":
            return f"Please sort [{cls._format_items(args['items'])}] in {args['rule']} order. Actually, near the end I say to keep only part of the list."
        if tool_id == "filter_items":
            return f"Please keep only the entries from [{cls._format_items(args['items'])}] that match '{args['rule']}'. Actually, near the end I say to keep everything and just reorder it."
        if tool_id == "convert_unit":
            return f"Convert {args['value']} {args['source']} into {args['target']}. Actually, near the end I make it sound like the money-conversion version."
        if tool_id == "convert_currency":
            return f"Convert {args['value']} {str(args['source']).upper()} into {str(args['target']).upper()}. Actually, near the end I make it sound like the measurement-conversion version."
        return f"{cls._ambiguity_probe_request(tool_id, args)} Actually, near the end I switch what kind of reading I seem to want."

    @classmethod
    def _intra_prompt_conflict_request(cls, tool_id: str, args: dict[str, Any]) -> str:
        if tool_id == "search_document":
            return f"I need something in {args['scope']} about {args['query']}, and I say both that it should read like regular documentation and that it should be the formal rule page."
        if tool_id == "search_policy":
            return f"I need something in {args['scope']} about {args['query']}, and I say both that it should be the formal rule page and that it should read like an ordinary document."
        if tool_id == "search_faq":
            return f"I need something in {args['scope']} about {args['query']}, and I say both that it should be the help-answer version and that it should be the formal policy."
        if tool_id == "calculate_route":
            return f"I ask for the route from {args['origin']} to {args['target']} by {args['profile']}, and in the same breath I also say I only want the duration."
        if tool_id == "estimate_travel_time":
            return f"I ask for the duration from {args['origin']} to {args['target']} by {args['profile']}, and in the same breath I also say I want the route itself."
        if tool_id == "sort_items":
            return f"I ask for [{cls._format_items(args['items'])}] in {args['rule']} order, and in the same breath I also say to keep only a matching subset."
        if tool_id == "filter_items":
            return f"I ask to keep only the entries from [{cls._format_items(args['items'])}] that match '{args['rule']}', and in the same breath I also say to keep every item and just reorder them."
        if tool_id == "convert_unit":
            return f"I ask for {args['value']} {args['source']} into {args['target']}, and in the same breath I also frame it as the money-conversion version."
        if tool_id == "convert_currency":
            return f"I ask for {args['value']} {str(args['source']).upper()} into {str(args['target']).upper()}, and in the same breath I also frame it as the measurement-conversion version."
        if tool_id == "get_weather":
            return f"I ask for the overall weather in {args['place']} {args['slot']}, and in the same breath I also say I only want the AQI."
        if tool_id == "get_air_quality":
            return f"I ask for the AQI-style reading in {args['place']} {args['slot']}, and in the same breath I also say I only want the broader weather."
        if tool_id == "get_temperature":
            return f"I ask for the temperature in {str(args['scale']).upper()} for {args['place']} {args['slot']}, and in the same breath I also say I only want the broader weather."
        raise ValueError(f"Unsupported tool for intra-prompt conflict: {tool_id}")

    @classmethod
    def _heavy_conflict_request(cls, tool_id: str, args: dict[str, Any]) -> str:
        if tool_id == "search_document":
            return f"I need the regular document on '{args['query']}' in {args['scope']}, except I also say the rule page is the one to use, and by the end I am talking about the {cls._alternate_value(args['scope'])} collection as if that had been the scope all along."
        if tool_id == "search_policy":
            return f"I need the policy on '{args['query']}' in {args['scope']}, except I also say the general document is the one to use, and by the end I am talking about the {cls._alternate_value(args['scope'])} collection as if that had been the scope all along."
        if tool_id == "search_faq":
            return f"I need the FAQ-style answer on '{args['query']}' in {args['scope']}, except I also say the formal policy is the one to use, and by the end I am talking about the {cls._alternate_value(args['scope'])} collection as if that had been the scope all along."
        if tool_id == "calculate_route":
            return f"I need the route from {args['origin']} to {args['target']} by {args['profile']}, except I also say only the travel time matters, and by the end I swap {args['origin']} for {cls._alternate_value(args['origin'])} as though that had been the start all along."
        if tool_id == "estimate_travel_time":
            return f"I need the travel time from {args['origin']} to {args['target']} by {args['profile']}, except I also say the route itself is what matters, and by the end I swap {args['origin']} for {cls._alternate_value(args['origin'])} as though that had been the start all along."
        if tool_id == "sort_items":
            return f"I need [{cls._format_items(args['items'])}] in {args['rule']} order, except I also say to keep only a matching subset, and by the end I talk as though every item must stay."
        if tool_id == "filter_items":
            return f"I need only the entries from [{cls._format_items(args['items'])}] that match '{args['rule']}', except I also say to reorder the whole list, and by the end I talk as though every item must stay."
        if tool_id == "convert_unit":
            return f"I need {args['value']} {args['source']} into {args['target']} as a measurement conversion, except I also frame it as money, and by the end I switch the target to {cls._alternate_value(args['target'])} as though that had been the unit all along."
        if tool_id == "convert_currency":
            return f"I need {args['value']} {str(args['source']).upper()} into {str(args['target']).upper()} as a money conversion, except I also frame it as a measurement conversion, and by the end I switch the target to {str(cls._alternate_value(args['target'])).upper()} as though that had been the currency all along."
        if tool_id == "get_weather":
            return f"I need the overall weather in {args['place']} {args['slot']}, except I also say to return only the AQI, and by the end I make it sound as though the temperature is what matters."
        if tool_id == "get_air_quality":
            return f"I need the AQI-style reading in {args['place']} {args['slot']}, except I also say to return only the broader weather, and by the end I make it sound as though the temperature is what matters."
        if tool_id == "get_temperature":
            return f"I need the temperature in {str(args['scale']).upper()} for {args['place']} {args['slot']}, except I also say to return only the broader weather, and by the end I make it sound as though the AQI is what matters."
        raise ValueError(f"Unsupported tool for heavy conflict: {tool_id}")

    @classmethod
    def _apply_operator(cls, text: str, operator_name: str, args: dict[str, Any], exposed_tool_name: str) -> str:
        if operator_name == "paraphrase":
            return cls._simple_paraphrase(text)
        if operator_name == "style_transfer":
            return f"Handle this in a more conversational, indirect style: {text}"
        if operator_name == "constraint_reposition":
            return f"{text} The key constraint may appear later than usual."
        if operator_name == "cross_sentence_scatter":
            return f"{text} One part of the constraint is stated separately. Another part is implied in the surrounding wording."
        if operator_name == "implication_rewrite":
            return f"I'm trying to get the right end result without naming each requirement directly. {text}"
        if operator_name == "slot_implicitization":
            return f"{text} One slot is only implied by context rather than stated directly."
        if operator_name == "vague_quantity":
            return f"{text} Treat one numeric cue as vaguely specified."
        if operator_name == "omitted_modifier":
            return f"{text} A modifier that would normally narrow interpretation is omitted."
        if operator_name == "partial_constraint_drop":
            return f"{text} One constraint is partially dropped from the prompt surface."
        if operator_name == "redundant_context":
            return f"Background note that is not actually needed. {text}"
        if operator_name == "near_semantic_distractor":
            return f"{text} A nearby semantic distractor makes a sibling action look plausible."
        if operator_name == "side_goal_insertion":
            return f"{text} There is also an inserted side goal that should not override the main one."
        if operator_name == "soft_topic_drift":
            return f"{text} The topic drifts softly before coming back to the real request."
        if operator_name == "contradictory_constraint":
            return f"{text} Another sentence contradicts one of the constraints."
        if operator_name == "late_revision":
            return f"{text} A late revision changes one detail near the end."
        if operator_name == "intra_prompt_conflict":
            return f"{text} Different parts of the prompt conflict with each other."
        if operator_name == "ambiguity_probe":
            return f"{text} The wording intentionally probes whether the caller can resolve ambiguity."
        if operator_name in {"key_rename_soft", "key_rename", "camel_snake_shift", "field_reorder", "nesting_shift", "wrapper_insertion"}:
            return f"Return structure will be perturbed via `{operator_name}`. {text}"
        if operator_name in {"missing_field", "partial_truncation", "dropped_metadata", "sparse_result"}:
            return f"Evidence loss may occur via `{operator_name}`. {text}"
        if operator_name in {"stale_value", "supporting_conflict", "wrong_fact_field", "unit_mismatch", "inconsistent_pair"}:
            return f"Evidence corruption may occur via `{operator_name}`. {text}"
        if operator_name in {"misleading_note", "distractor_candidate_soft", "distractor_candidate", "low_utility_redundancy_soft", "low_utility_redundancy", "spurious_auxiliary_field"}:
            return f"Competing evidence may appear via `{operator_name}`. {text}"
        if operator_name in {"cached_availability_fallback", "service_unavailable_with_stale_cache", "timeout_after_partial_stream", "jittered_metadata", "delayed_fresh_partial_with_stale_full", "flaky_shape_conflicting_attempts", "contaminated_flaky_attempts", "vague_retry_hint", "wrong_failure_label_with_evidence", "hidden_partial_wrong_label", "conflicting_diagnostic_decoy", "timeout", "service_unavailable", "auth_denied", "empty_response", "nondeterministic_order", "delayed_partial_response", "unstable_optional_field", "flaky_response_shape", "vague_error", "misleading_error", "partial_success_no_status", "wrong_failure_label"}:
            return f"Execution environment noise may appear via `{operator_name}`. {text}"
        return text

    @staticmethod
    def _simple_paraphrase(text: str) -> str:
        return text.replace("I need", "I want").replace("Please", "Could you")

    @classmethod
    def _render_request(cls, tool_id: str, args: dict[str, Any], variant_index: int) -> str:
        values = dict(args)
        if "scale" in values:
            values["scale_upper"] = str(values["scale"]).upper()
        if "source" in values:
            values["source_upper"] = str(values["source"]).upper()
            values["target_upper"] = str(values["target"]).upper()
        values.setdefault("tool_label", tool_id.replace("_", " "))
        return _PATTERNS[tool_id][variant_index % len(_PATTERNS[tool_id])].format(**values)

    @staticmethod
    def _slug(value: str) -> str:
        return value.lower().replace(" ", "_")
