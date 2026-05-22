"""Static ToolRobust v3 operator catalog and mapping tables."""

from __future__ import annotations

from collections import OrderedDict

from toolrobust.domain.enums import BenchmarkFamily, FailureTag, OperatorStrength, StageTarget
from toolrobust.domain.schemas import OperatorSpec

OPERATOR_CATALOG_VERSION = "v3.8"

SUBTYPE_STAGE_TARGET = {
    "A1 Name Surface Shift": StageTarget.S1.value,
    "A2 Description Abstraction": StageTarget.S2.value,
    "A3 Family Similarity Pressure": StageTarget.S2.value,
    "B1 Indirectness": StageTarget.S2.value,
    "B2 Underspecification": StageTarget.S3.value,
    "B3 Distraction": StageTarget.S1.value,
    "B4 Conflict": StageTarget.S1.value,
    "C1 Return Structure Noise": StageTarget.S4.value,
    "C2 Return Evidence Loss": StageTarget.S4.value,
    "C3 Return Evidence Corruption": StageTarget.S4.value,
    "C4 Return Evidence Competition": StageTarget.S4.value,
    "D1 Availability Failure": StageTarget.S4.value,
    "D2 Runtime Variability": StageTarget.S4.value,
    "D3 Failure Feedback Quality": StageTarget.S4.value,
}

SUBTYPE_FAILURE_RISKS = {
    "A1 Name Surface Shift": [FailureTag.WRONG_TOOL.value],
    "A2 Description Abstraction": [FailureTag.WRONG_TOOL.value],
    "A3 Family Similarity Pressure": [FailureTag.WRONG_TOOL.value],
    "B1 Indirectness": [FailureTag.WRONG_TOOL.value, FailureTag.WRONG_PARAMETER.value],
    "B2 Underspecification": [FailureTag.MISSING_PARAMETER.value, FailureTag.WRONG_PARAMETER.value],
    "B3 Distraction": [FailureTag.WRONG_TOOL.value, FailureTag.WRONG_PARAMETER.value],
    "B4 Conflict": [FailureTag.NO_CALL.value, FailureTag.WRONG_TOOL.value, FailureTag.MISSING_PARAMETER.value, FailureTag.WRONG_PARAMETER.value],
    "C1 Return Structure Noise": [FailureTag.INVALID_FORMAT.value, FailureTag.WRONG_PARAMETER.value],
    "C2 Return Evidence Loss": [FailureTag.MISSING_PARAMETER.value, FailureTag.EXECUTION_FAIL.value, FailureTag.WRONG_PARAMETER.value],
    "C3 Return Evidence Corruption": [FailureTag.WRONG_PARAMETER.value, FailureTag.EXECUTION_FAIL.value],
    "C4 Return Evidence Competition": [FailureTag.WRONG_PARAMETER.value],
    "D1 Availability Failure": [FailureTag.EXECUTION_FAIL.value, FailureTag.NO_CALL.value],
    "D2 Runtime Variability": [FailureTag.INVALID_FORMAT.value, FailureTag.WRONG_PARAMETER.value, FailureTag.EXECUTION_FAIL.value],
    "D3 Failure Feedback Quality": [FailureTag.EXECUTION_FAIL.value, FailureTag.NO_CALL.value, FailureTag.WRONG_PARAMETER.value],
}

FAMILY_CATALOG: "OrderedDict[str, OrderedDict[str, list[OperatorSpec]]]" = OrderedDict(
    [
        (
            BenchmarkFamily.INTERFACE_EXPOSURE.value,
            OrderedDict(
                [
                    (
                        "A1 Name Surface Shift",
                        [
                            OperatorSpec(family=BenchmarkFamily.INTERFACE_EXPOSURE.value, subtype="A1 Name Surface Shift", operator_name="transparent_alias", default_strength=OperatorStrength.MILD, default_stage_target=StageTarget.S1.value, primary_failure_risks=[FailureTag.WRONG_TOOL.value], intrinsic_strength_score=0.18, subtype_affinity=["A1 Name Surface Shift"], severity_fit=["light"], exposure_axis="name", stack_role="primary", should_not_stack_with=["semi_opaque_name", "family_collision_name"], minimum_effect_expectation=0.05, maximum_effect_expectation=0.20, notes="Readable alias close to the canonical action."),
                            OperatorSpec(family=BenchmarkFamily.INTERFACE_EXPOSURE.value, subtype="A1 Name Surface Shift", operator_name="semi_opaque_name", default_strength=OperatorStrength.STRONG, default_stage_target=StageTarget.S1.value, primary_failure_risks=[FailureTag.WRONG_TOOL.value], intrinsic_strength_score=0.52, subtype_affinity=["A1 Name Surface Shift"], severity_fit=["medium", "heavy"], exposure_axis="name", stack_role="primary", can_stack_with=["opaque_internal_name"], should_not_stack_with=["transparent_alias", "family_collision_name"], minimum_effect_expectation=0.25, maximum_effect_expectation=0.60, notes="Weakly semantic name that still hints at the family."),
                            OperatorSpec(family=BenchmarkFamily.INTERFACE_EXPOSURE.value, subtype="A1 Name Surface Shift", operator_name="opaque_internal_name", default_strength=OperatorStrength.STRONG, default_stage_target=StageTarget.S1.value, primary_failure_risks=[FailureTag.WRONG_TOOL.value], intrinsic_strength_score=0.34, subtype_affinity=["A1 Name Surface Shift"], severity_fit=["heavy"], exposure_axis="name", stack_role="amplifier", can_stack_with=["semi_opaque_name"], should_not_stack_with=["transparent_alias", "family_collision_name"], minimum_effect_expectation=0.14, maximum_effect_expectation=0.38, notes="Escalate a semi-opaque name into a model-visible internal service code."),
                            OperatorSpec(family=BenchmarkFamily.INTERFACE_EXPOSURE.value, subtype="A1 Name Surface Shift", operator_name="alias_pruning", default_strength=OperatorStrength.MILD, default_stage_target=StageTarget.S2.value, primary_failure_risks=[FailureTag.WRONG_TOOL.value], intrinsic_strength_score=0.16, subtype_affinity=["A1 Name Surface Shift"], severity_fit=["heavy"], exposure_axis="name", stack_role="amplifier", can_stack_with=["semi_opaque_name"], should_not_stack_with=["transparent_alias"], minimum_effect_expectation=0.05, maximum_effect_expectation=0.18, notes="Remove alias help so a semi-opaque name has less support."),
                        ],
                    ),
                    (
                        "A2 Description Abstraction",
                        [
                            OperatorSpec(family=BenchmarkFamily.INTERFACE_EXPOSURE.value, subtype="A2 Description Abstraction", operator_name="light_description_rewrite", default_strength=OperatorStrength.MILD, default_stage_target=StageTarget.S2.value, primary_failure_risks=[FailureTag.WRONG_TOOL.value], intrinsic_strength_score=0.18, subtype_affinity=["A2 Description Abstraction"], severity_fit=["light"], exposure_axis="description", stack_role="primary", should_not_stack_with=["description_abstraction"], minimum_effect_expectation=0.05, maximum_effect_expectation=0.20, notes="Light rewrite that keeps the action legible but trims some direct cues."),
                            OperatorSpec(family=BenchmarkFamily.INTERFACE_EXPOSURE.value, subtype="A2 Description Abstraction", operator_name="description_abstraction", default_strength=OperatorStrength.STRONG, default_stage_target=StageTarget.S2.value, primary_failure_risks=[FailureTag.WRONG_TOOL.value], intrinsic_strength_score=0.55, subtype_affinity=["A2 Description Abstraction"], severity_fit=["medium", "heavy"], exposure_axis="description", stack_role="primary", can_stack_with=["family_collision_name"], should_not_stack_with=["light_description_rewrite"], minimum_effect_expectation=0.22, maximum_effect_expectation=0.68, notes="Reduce explicit discriminative phrases while preserving coarse functional meaning."),
                        ],
                    ),
                    (
                        "A3 Family Similarity Pressure",
                        [
                            OperatorSpec(family=BenchmarkFamily.INTERFACE_EXPOSURE.value, subtype="A3 Family Similarity Pressure", operator_name="sibling_similarity_hint", default_strength=OperatorStrength.MILD, default_stage_target=StageTarget.S2.value, primary_failure_risks=[FailureTag.WRONG_TOOL.value], intrinsic_strength_score=0.18, subtype_affinity=["A3 Family Similarity Pressure"], severity_fit=["light"], exposure_axis="family_similarity", stack_role="primary", should_not_stack_with=["family_collision_name", "cross_tool_description_blend"], minimum_effect_expectation=0.05, maximum_effect_expectation=0.20, notes="Make sibling tools read slightly more alike without hiding the function."),
                            OperatorSpec(family=BenchmarkFamily.INTERFACE_EXPOSURE.value, subtype="A3 Family Similarity Pressure", operator_name="family_collision_name", default_strength=OperatorStrength.STRONG, default_stage_target=StageTarget.S1.value, primary_failure_risks=[FailureTag.WRONG_TOOL.value], intrinsic_strength_score=0.50, subtype_affinity=["A1 Name Surface Shift", "A2 Description Abstraction", "A3 Family Similarity Pressure"], severity_fit=["heavy", "medium"], exposure_axis="family_similarity", stack_role="primary", can_stack_with=["alias_pruning", "cross_tool_description_blend", "description_abstraction"], should_not_stack_with=["transparent_alias", "semi_opaque_name", "sibling_similarity_hint"], minimum_effect_expectation=0.20, maximum_effect_expectation=0.62, notes="Rename sibling tools into a near-collision family style."),
                            OperatorSpec(family=BenchmarkFamily.INTERFACE_EXPOSURE.value, subtype="A3 Family Similarity Pressure", operator_name="cross_tool_description_blend", default_strength=OperatorStrength.STRONG, default_stage_target=StageTarget.S2.value, primary_failure_risks=[FailureTag.WRONG_TOOL.value], intrinsic_strength_score=0.20, subtype_affinity=["A3 Family Similarity Pressure"], severity_fit=["heavy"], exposure_axis="family_similarity", stack_role="amplifier", can_stack_with=["family_collision_name"], minimum_effect_expectation=0.10, maximum_effect_expectation=0.24, notes="Blend descriptions across sibling tools to reduce separability."),
                        ],
                    ),
                ]
            ),
        ),
        (
            BenchmarkFamily.USER_INTENT.value,
            OrderedDict(
                [
                    (
                        "B1 Indirectness",
                        [
                            OperatorSpec(family=BenchmarkFamily.USER_INTENT.value, subtype="B1 Indirectness", operator_name="paraphrase", default_strength=OperatorStrength.MILD, default_stage_target=StageTarget.S2.value, primary_failure_risks=SUBTYPE_FAILURE_RISKS["B1 Indirectness"], intrinsic_strength_score=0.08, subtype_affinity=["B1 Indirectness"], severity_fit=["light"], exposure_axis="indirectness", stack_role="primary", should_not_stack_with=["style_transfer"], minimum_effect_expectation=0.02, maximum_effect_expectation=0.10, notes="Surface-level rewording that keeps the request explicit."),
                            OperatorSpec(family=BenchmarkFamily.USER_INTENT.value, subtype="B1 Indirectness", operator_name="style_transfer", default_strength=OperatorStrength.STRONG, default_stage_target=StageTarget.S2.value, primary_failure_risks=SUBTYPE_FAILURE_RISKS["B1 Indirectness"], intrinsic_strength_score=0.34, subtype_affinity=["B1 Indirectness"], severity_fit=["medium", "heavy"], exposure_axis="indirectness", stack_role="primary", can_stack_with=["constraint_reposition", "cross_sentence_scatter", "implication_rewrite"], should_not_stack_with=["paraphrase"], minimum_effect_expectation=0.10, maximum_effect_expectation=0.34, notes="Convert direct commands into indirect conversational intent."),
                            OperatorSpec(family=BenchmarkFamily.USER_INTENT.value, subtype="B1 Indirectness", operator_name="constraint_reposition", default_strength=OperatorStrength.MILD, default_stage_target=StageTarget.S3.value, primary_failure_risks=SUBTYPE_FAILURE_RISKS["B1 Indirectness"], intrinsic_strength_score=0.18, subtype_affinity=["B1 Indirectness"], severity_fit=["light", "medium"], exposure_axis="indirectness", stack_role="amplifier", minimum_effect_expectation=0.05, maximum_effect_expectation=0.20, notes="Move the discriminative clue away from its expected surface position."),
                            OperatorSpec(family=BenchmarkFamily.USER_INTENT.value, subtype="B1 Indirectness", operator_name="cross_sentence_scatter", default_strength=OperatorStrength.STRONG, default_stage_target=StageTarget.S3.value, primary_failure_risks=SUBTYPE_FAILURE_RISKS["B1 Indirectness"], intrinsic_strength_score=0.28, subtype_affinity=["B1 Indirectness"], severity_fit=["heavy"], exposure_axis="indirectness", stack_role="amplifier", can_stack_with=["style_transfer", "implication_rewrite"], minimum_effect_expectation=0.08, maximum_effect_expectation=0.30, notes="Scatter intent cues across sentences instead of keeping them local."),
                            OperatorSpec(family=BenchmarkFamily.USER_INTENT.value, subtype="B1 Indirectness", operator_name="implication_rewrite", default_strength=OperatorStrength.STRONG, default_stage_target=StageTarget.S3.value, primary_failure_risks=SUBTYPE_FAILURE_RISKS["B1 Indirectness"], intrinsic_strength_score=0.32, subtype_affinity=["B1 Indirectness"], severity_fit=["heavy"], exposure_axis="indirectness", stack_role="amplifier", can_stack_with=["style_transfer", "cross_sentence_scatter"], minimum_effect_expectation=0.08, maximum_effect_expectation=0.34, notes="Replace explicit action labels with implied intent."),
                        ],
                    ),
                    (
                        "B2 Underspecification",
                        [
                            OperatorSpec(family=BenchmarkFamily.USER_INTENT.value, subtype="B2 Underspecification", operator_name="slot_implicitization", default_strength=OperatorStrength.STRONG, default_stage_target=StageTarget.S3.value, primary_failure_risks=SUBTYPE_FAILURE_RISKS["B2 Underspecification"], intrinsic_strength_score=0.36, subtype_affinity=["B2 Underspecification"], severity_fit=["medium", "heavy"], exposure_axis="underspecification", stack_role="primary", can_stack_with=["partial_constraint_drop"], minimum_effect_expectation=0.12, maximum_effect_expectation=0.38, notes="Turn explicit slots into contextual references."),
                            OperatorSpec(family=BenchmarkFamily.USER_INTENT.value, subtype="B2 Underspecification", operator_name="vague_quantity", default_strength=OperatorStrength.MILD, default_stage_target=StageTarget.S3.value, primary_failure_risks=SUBTYPE_FAILURE_RISKS["B2 Underspecification"], intrinsic_strength_score=0.14, subtype_affinity=["B2 Underspecification"], severity_fit=["light"], exposure_axis="underspecification", stack_role="primary", minimum_effect_expectation=0.04, maximum_effect_expectation=0.16, notes="Blur a numeric or magnitude cue without removing it."),
                            OperatorSpec(family=BenchmarkFamily.USER_INTENT.value, subtype="B2 Underspecification", operator_name="omitted_modifier", default_strength=OperatorStrength.MILD, default_stage_target=StageTarget.S3.value, primary_failure_risks=SUBTYPE_FAILURE_RISKS["B2 Underspecification"], intrinsic_strength_score=0.18, subtype_affinity=["B2 Underspecification"], severity_fit=["light"], exposure_axis="underspecification", stack_role="amplifier", minimum_effect_expectation=0.05, maximum_effect_expectation=0.22, notes="Remove a modifier that would normally disambiguate the request."),
                            OperatorSpec(family=BenchmarkFamily.USER_INTENT.value, subtype="B2 Underspecification", operator_name="partial_constraint_drop", default_strength=OperatorStrength.STRONG, default_stage_target=StageTarget.S3.value, primary_failure_risks=SUBTYPE_FAILURE_RISKS["B2 Underspecification"], intrinsic_strength_score=0.46, subtype_affinity=["B2 Underspecification"], severity_fit=["heavy"], exposure_axis="underspecification", stack_role="amplifier", can_stack_with=["slot_implicitization"], minimum_effect_expectation=0.14, maximum_effect_expectation=0.50, notes="Drop a high-value discriminative constraint from the surface form."),
                        ],
                    ),
                    (
                        "B3 Distraction",
                        [
                            OperatorSpec(family=BenchmarkFamily.USER_INTENT.value, subtype="B3 Distraction", operator_name="redundant_context", default_strength=OperatorStrength.MILD, default_stage_target=StageTarget.S3.value, primary_failure_risks=SUBTYPE_FAILURE_RISKS["B3 Distraction"], intrinsic_strength_score=0.13, subtype_affinity=["B3 Distraction"], severity_fit=["light"], exposure_axis="distraction", stack_role="primary", minimum_effect_expectation=0.03, maximum_effect_expectation=0.14, notes="Add realistic but irrelevant setup context before the request."),
                            OperatorSpec(family=BenchmarkFamily.USER_INTENT.value, subtype="B3 Distraction", operator_name="near_semantic_distractor", default_strength=OperatorStrength.STRONG, default_stage_target=StageTarget.S1.value, primary_failure_risks=SUBTYPE_FAILURE_RISKS["B3 Distraction"], intrinsic_strength_score=0.36, subtype_affinity=["B3 Distraction"], severity_fit=["medium", "heavy"], exposure_axis="distraction", stack_role="primary", can_stack_with=["side_goal_insertion", "soft_topic_drift"], minimum_effect_expectation=0.14, maximum_effect_expectation=0.40, notes="Mention a sibling action that is semantically adjacent to the gold tool."),
                            OperatorSpec(family=BenchmarkFamily.USER_INTENT.value, subtype="B3 Distraction", operator_name="side_goal_insertion", default_strength=OperatorStrength.STRONG, default_stage_target=StageTarget.S3.value, primary_failure_risks=SUBTYPE_FAILURE_RISKS["B3 Distraction"], intrinsic_strength_score=0.20, subtype_affinity=["B3 Distraction"], severity_fit=["medium", "heavy"], exposure_axis="distraction", stack_role="amplifier", can_stack_with=["near_semantic_distractor", "soft_topic_drift"], minimum_effect_expectation=0.05, maximum_effect_expectation=0.22, notes="Insert a plausible secondary objective that competes with the main task."),
                            OperatorSpec(family=BenchmarkFamily.USER_INTENT.value, subtype="B3 Distraction", operator_name="soft_topic_drift", default_strength=OperatorStrength.STRONG, default_stage_target=StageTarget.S1.value, primary_failure_risks=SUBTYPE_FAILURE_RISKS["B3 Distraction"], intrinsic_strength_score=0.20, subtype_affinity=["B3 Distraction"], severity_fit=["heavy"], exposure_axis="distraction", stack_role="amplifier", can_stack_with=["near_semantic_distractor", "side_goal_insertion"], minimum_effect_expectation=0.05, maximum_effect_expectation=0.22, notes="Let the prompt wander through a nearby topic before returning to the real ask."),
                        ],
                    ),
                    (
                        "B4 Conflict",
                        [
                            OperatorSpec(family=BenchmarkFamily.USER_INTENT.value, subtype="B4 Conflict", operator_name="contradictory_constraint", default_strength=OperatorStrength.STRONG, default_stage_target=StageTarget.S3.value, primary_failure_risks=SUBTYPE_FAILURE_RISKS["B4 Conflict"], intrinsic_strength_score=0.36, subtype_affinity=["B4 Conflict"], severity_fit=["medium", "heavy"], exposure_axis="conflict", stack_role="primary", can_stack_with=["late_revision", "intra_prompt_conflict"], minimum_effect_expectation=0.12, maximum_effect_expectation=0.40, notes="Introduce a direct contradiction against a key tool-discriminative constraint."),
                            OperatorSpec(family=BenchmarkFamily.USER_INTENT.value, subtype="B4 Conflict", operator_name="late_revision", default_strength=OperatorStrength.STRONG, default_stage_target=StageTarget.S3.value, primary_failure_risks=SUBTYPE_FAILURE_RISKS["B4 Conflict"], intrinsic_strength_score=0.24, subtype_affinity=["B4 Conflict"], severity_fit=["heavy"], exposure_axis="conflict", stack_role="amplifier", can_stack_with=["contradictory_constraint", "intra_prompt_conflict"], minimum_effect_expectation=0.06, maximum_effect_expectation=0.26, notes="Append a late edit that changes a previously stable slot value."),
                            OperatorSpec(family=BenchmarkFamily.USER_INTENT.value, subtype="B4 Conflict", operator_name="intra_prompt_conflict", default_strength=OperatorStrength.STRONG, default_stage_target=StageTarget.S1.value, primary_failure_risks=SUBTYPE_FAILURE_RISKS["B4 Conflict"], intrinsic_strength_score=0.26, subtype_affinity=["B4 Conflict"], severity_fit=["heavy"], exposure_axis="conflict", stack_role="amplifier", can_stack_with=["contradictory_constraint", "late_revision"], minimum_effect_expectation=0.06, maximum_effect_expectation=0.28, notes="Create conflicting top-level goals inside the same request."),
                            OperatorSpec(family=BenchmarkFamily.USER_INTENT.value, subtype="B4 Conflict", operator_name="ambiguity_probe", default_strength=OperatorStrength.MILD, default_stage_target=StageTarget.S1.value, primary_failure_risks=SUBTYPE_FAILURE_RISKS["B4 Conflict"], intrinsic_strength_score=0.13, subtype_affinity=["B4 Conflict"], severity_fit=["light"], exposure_axis="conflict", stack_role="primary", minimum_effect_expectation=0.03, maximum_effect_expectation=0.14, notes="Leave a mild but recoverable ambiguity between two sibling intents."),
                        ],
                    ),
                ]
            ),
        ),
        (
            BenchmarkFamily.OBSERVATION.value,
            OrderedDict(
                [
                    (
                        "C1 Return Structure Noise",
                        [
                            OperatorSpec(family=BenchmarkFamily.OBSERVATION.value, subtype="C1 Return Structure Noise", operator_name="key_rename_soft", default_strength=OperatorStrength.MILD, default_stage_target=StageTarget.S4.value, primary_failure_risks=SUBTYPE_FAILURE_RISKS["C1 Return Structure Noise"], intrinsic_strength_score=0.12, subtype_affinity=["C1 Return Structure Noise"], severity_fit=["light", "medium"], exposure_axis="structure", stack_role="primary", can_stack_with=["field_reorder", "nesting_shift"], minimum_effect_expectation=0.03, maximum_effect_expectation=0.12, notes="Rename one salient field in the observed payload without rewriting the entire envelope."),
                            OperatorSpec(family=BenchmarkFamily.OBSERVATION.value, subtype="C1 Return Structure Noise", operator_name="key_rename", default_strength=OperatorStrength.MILD, default_stage_target=StageTarget.S4.value, primary_failure_risks=SUBTYPE_FAILURE_RISKS["C1 Return Structure Noise"], intrinsic_strength_score=0.18, subtype_affinity=["C1 Return Structure Noise"], severity_fit=["heavy"], exposure_axis="structure", stack_role="primary", can_stack_with=["field_reorder", "camel_snake_shift", "wrapper_insertion", "nesting_shift"], should_not_stack_with=[], minimum_effect_expectation=0.04, maximum_effect_expectation=0.20, notes="Rename salient fields across the observed payload while also perturbing wrapper-level keys."),
                            OperatorSpec(family=BenchmarkFamily.OBSERVATION.value, subtype="C1 Return Structure Noise", operator_name="camel_snake_shift", default_strength=OperatorStrength.MILD, default_stage_target=StageTarget.S4.value, primary_failure_risks=SUBTYPE_FAILURE_RISKS["C1 Return Structure Noise"], intrinsic_strength_score=0.16, subtype_affinity=["C1 Return Structure Noise"], severity_fit=["medium", "heavy"], exposure_axis="structure", stack_role="amplifier", can_stack_with=["key_rename", "nesting_shift", "wrapper_insertion"], minimum_effect_expectation=0.03, maximum_effect_expectation=0.16, notes="Shift field names between snake_case and camelCase while leaving the evidence values intact."),
                            OperatorSpec(family=BenchmarkFamily.OBSERVATION.value, subtype="C1 Return Structure Noise", operator_name="field_reorder", default_strength=OperatorStrength.MILD, default_stage_target=StageTarget.S4.value, primary_failure_risks=SUBTYPE_FAILURE_RISKS["C1 Return Structure Noise"], intrinsic_strength_score=0.10, subtype_affinity=["C1 Return Structure Noise"], severity_fit=["light"], exposure_axis="structure", stack_role="amplifier", can_stack_with=["key_rename", "key_rename_soft"], minimum_effect_expectation=0.02, maximum_effect_expectation=0.12, notes="Reorder fields while keeping the payload otherwise intact."),
                            OperatorSpec(family=BenchmarkFamily.OBSERVATION.value, subtype="C1 Return Structure Noise", operator_name="nesting_shift", default_strength=OperatorStrength.STRONG, default_stage_target=StageTarget.S4.value, primary_failure_risks=SUBTYPE_FAILURE_RISKS["C1 Return Structure Noise"], intrinsic_strength_score=0.46, subtype_affinity=["C1 Return Structure Noise"], severity_fit=["medium", "heavy"], exposure_axis="structure", stack_role="primary", can_stack_with=["camel_snake_shift", "key_rename", "key_rename_soft"], should_not_stack_with=["wrapper_insertion"], minimum_effect_expectation=0.16, maximum_effect_expectation=0.48, notes="Move the payload into a deeper nested envelope."),
                            OperatorSpec(family=BenchmarkFamily.OBSERVATION.value, subtype="C1 Return Structure Noise", operator_name="wrapper_insertion", default_strength=OperatorStrength.STRONG, default_stage_target=StageTarget.S4.value, primary_failure_risks=SUBTYPE_FAILURE_RISKS["C1 Return Structure Noise"], intrinsic_strength_score=0.52, subtype_affinity=["C1 Return Structure Noise"], severity_fit=["heavy"], exposure_axis="structure", stack_role="amplifier", can_stack_with=["key_rename", "camel_snake_shift"], minimum_effect_expectation=0.18, maximum_effect_expectation=0.56, notes="Insert a new wrapper object around the entire observation."),
                        ],
                    ),
                    (
                        "C2 Return Evidence Loss",
                        [
                            OperatorSpec(family=BenchmarkFamily.OBSERVATION.value, subtype="C2 Return Evidence Loss", operator_name="missing_field", default_strength=OperatorStrength.STRONG, default_stage_target=StageTarget.S4.value, primary_failure_risks=SUBTYPE_FAILURE_RISKS["C2 Return Evidence Loss"], intrinsic_strength_score=0.48, subtype_affinity=["C2 Return Evidence Loss"], severity_fit=["medium", "heavy"], exposure_axis="evidence_loss", stack_role="primary", can_stack_with=["partial_truncation", "sparse_result"], minimum_effect_expectation=0.18, maximum_effect_expectation=0.52, notes="Remove a high-value evidence field from the payload."),
                            OperatorSpec(family=BenchmarkFamily.OBSERVATION.value, subtype="C2 Return Evidence Loss", operator_name="partial_truncation", default_strength=OperatorStrength.STRONG, default_stage_target=StageTarget.S4.value, primary_failure_risks=SUBTYPE_FAILURE_RISKS["C2 Return Evidence Loss"], intrinsic_strength_score=0.38, subtype_affinity=["C2 Return Evidence Loss"], severity_fit=["heavy"], exposure_axis="evidence_loss", stack_role="amplifier", can_stack_with=["missing_field"], minimum_effect_expectation=0.12, maximum_effect_expectation=0.40, notes="Truncate evidence-bearing content instead of dropping it entirely."),
                            OperatorSpec(family=BenchmarkFamily.OBSERVATION.value, subtype="C2 Return Evidence Loss", operator_name="dropped_metadata", default_strength=OperatorStrength.MILD, default_stage_target=StageTarget.S4.value, primary_failure_risks=SUBTYPE_FAILURE_RISKS["C2 Return Evidence Loss"], intrinsic_strength_score=0.18, subtype_affinity=["C2 Return Evidence Loss"], severity_fit=["light"], exposure_axis="evidence_loss", stack_role="primary", should_not_stack_with=["sparse_result"], minimum_effect_expectation=0.04, maximum_effect_expectation=0.18, notes="Drop contextual metadata while leaving the core value present."),
                            OperatorSpec(family=BenchmarkFamily.OBSERVATION.value, subtype="C2 Return Evidence Loss", operator_name="sparse_result", default_strength=OperatorStrength.MILD, default_stage_target=StageTarget.S4.value, primary_failure_risks=SUBTYPE_FAILURE_RISKS["C2 Return Evidence Loss"], intrinsic_strength_score=0.22, subtype_affinity=["C2 Return Evidence Loss"], severity_fit=["light", "medium"], exposure_axis="evidence_loss", stack_role="primary", can_stack_with=["missing_field"], minimum_effect_expectation=0.05, maximum_effect_expectation=0.22, notes="Collapse the payload to a small but still partially recoverable subset of evidence."),
                        ],
                    ),
                    (
                        "C3 Return Evidence Corruption",
                        [
                            OperatorSpec(family=BenchmarkFamily.OBSERVATION.value, subtype="C3 Return Evidence Corruption", operator_name="stale_value", default_strength=OperatorStrength.MILD, default_stage_target=StageTarget.S4.value, primary_failure_risks=SUBTYPE_FAILURE_RISKS["C3 Return Evidence Corruption"], intrinsic_strength_score=0.20, subtype_affinity=["C3 Return Evidence Corruption"], severity_fit=["light", "medium", "heavy"], exposure_axis="evidence_corruption", stack_role="amplifier", can_stack_with=["supporting_conflict", "unit_mismatch"], minimum_effect_expectation=0.05, maximum_effect_expectation=0.20, notes="Swap one supporting value with a stale-looking variant."),
                            OperatorSpec(family=BenchmarkFamily.OBSERVATION.value, subtype="C3 Return Evidence Corruption", operator_name="supporting_conflict", default_strength=OperatorStrength.MILD, default_stage_target=StageTarget.S4.value, primary_failure_risks=SUBTYPE_FAILURE_RISKS["C3 Return Evidence Corruption"], intrinsic_strength_score=0.30, subtype_affinity=["C3 Return Evidence Corruption"], severity_fit=["medium", "heavy"], exposure_axis="evidence_corruption", stack_role="primary", can_stack_with=["stale_value", "unit_mismatch"], minimum_effect_expectation=0.09, maximum_effect_expectation=0.28, notes="Corrupt multiple supporting cues while leaving the primary evidence field intact."),
                            OperatorSpec(family=BenchmarkFamily.OBSERVATION.value, subtype="C3 Return Evidence Corruption", operator_name="wrong_fact_field", default_strength=OperatorStrength.STRONG, default_stage_target=StageTarget.S4.value, primary_failure_risks=SUBTYPE_FAILURE_RISKS["C3 Return Evidence Corruption"], intrinsic_strength_score=0.44, subtype_affinity=["C3 Return Evidence Corruption"], severity_fit=["heavy"], exposure_axis="evidence_corruption", stack_role="primary", should_not_stack_with=["unit_mismatch"], can_stack_with=["stale_value", "inconsistent_pair"], minimum_effect_expectation=0.16, maximum_effect_expectation=0.46, notes="Corrupt the main fact-bearing field while keeping the surrounding structure intact."),
                            OperatorSpec(family=BenchmarkFamily.OBSERVATION.value, subtype="C3 Return Evidence Corruption", operator_name="unit_mismatch", default_strength=OperatorStrength.STRONG, default_stage_target=StageTarget.S4.value, primary_failure_risks=SUBTYPE_FAILURE_RISKS["C3 Return Evidence Corruption"], intrinsic_strength_score=0.52, subtype_affinity=["C3 Return Evidence Corruption"], severity_fit=["heavy"], exposure_axis="evidence_corruption", stack_role="primary", can_stack_with=["stale_value", "supporting_conflict", "inconsistent_pair"], minimum_effect_expectation=0.20, maximum_effect_expectation=0.54, notes="Keep a value but flip the unit or label attached to it."),
                            OperatorSpec(family=BenchmarkFamily.OBSERVATION.value, subtype="C3 Return Evidence Corruption", operator_name="inconsistent_pair", default_strength=OperatorStrength.STRONG, default_stage_target=StageTarget.S4.value, primary_failure_risks=SUBTYPE_FAILURE_RISKS["C3 Return Evidence Corruption"], intrinsic_strength_score=0.34, subtype_affinity=["C3 Return Evidence Corruption"], severity_fit=["heavy"], exposure_axis="evidence_corruption", stack_role="amplifier", can_stack_with=["unit_mismatch", "wrong_fact_field"], minimum_effect_expectation=0.12, maximum_effect_expectation=0.36, notes="Create a conflicting pair of evidence fields."),
                        ],
                    ),
                    (
                        "C4 Return Evidence Competition",
                        [
                            OperatorSpec(family=BenchmarkFamily.OBSERVATION.value, subtype="C4 Return Evidence Competition", operator_name="misleading_note", default_strength=OperatorStrength.STRONG, default_stage_target=StageTarget.S4.value, primary_failure_risks=SUBTYPE_FAILURE_RISKS["C4 Return Evidence Competition"], intrinsic_strength_score=0.40, subtype_affinity=["C4 Return Evidence Competition"], severity_fit=["heavy"], exposure_axis="evidence_competition", stack_role="amplifier", can_stack_with=["distractor_candidate", "low_utility_redundancy"], minimum_effect_expectation=0.14, maximum_effect_expectation=0.44, notes="Add an explicit note nudging toward the wrong interpretation."),
                            OperatorSpec(family=BenchmarkFamily.OBSERVATION.value, subtype="C4 Return Evidence Competition", operator_name="distractor_candidate_soft", default_strength=OperatorStrength.MILD, default_stage_target=StageTarget.S4.value, primary_failure_risks=SUBTYPE_FAILURE_RISKS["C4 Return Evidence Competition"], intrinsic_strength_score=0.28, subtype_affinity=["C4 Return Evidence Competition"], severity_fit=["medium"], exposure_axis="evidence_competition", stack_role="primary", can_stack_with=["low_utility_redundancy_soft"], minimum_effect_expectation=0.08, maximum_effect_expectation=0.26, notes="Inject a plausible alternative candidate without strongly biasing the observation toward it."),
                            OperatorSpec(family=BenchmarkFamily.OBSERVATION.value, subtype="C4 Return Evidence Competition", operator_name="distractor_candidate", default_strength=OperatorStrength.STRONG, default_stage_target=StageTarget.S4.value, primary_failure_risks=SUBTYPE_FAILURE_RISKS["C4 Return Evidence Competition"], intrinsic_strength_score=0.50, subtype_affinity=["C4 Return Evidence Competition"], severity_fit=["heavy"], exposure_axis="evidence_competition", stack_role="primary", can_stack_with=["spurious_auxiliary_field", "misleading_note", "low_utility_redundancy"], minimum_effect_expectation=0.18, maximum_effect_expectation=0.54, notes="Inject a plausible alternative candidate into the observation and bias the packet toward it."),
                            OperatorSpec(family=BenchmarkFamily.OBSERVATION.value, subtype="C4 Return Evidence Competition", operator_name="low_utility_redundancy_soft", default_strength=OperatorStrength.MILD, default_stage_target=StageTarget.S4.value, primary_failure_risks=SUBTYPE_FAILURE_RISKS["C4 Return Evidence Competition"], intrinsic_strength_score=0.13, subtype_affinity=["C4 Return Evidence Competition"], severity_fit=["light"], exposure_axis="evidence_competition", stack_role="amplifier", can_stack_with=["distractor_candidate_soft"], minimum_effect_expectation=0.03, maximum_effect_expectation=0.12, notes="Wrap the observation with mild duplicate bookkeeping without introducing a preferred wrong view."),
                            OperatorSpec(family=BenchmarkFamily.OBSERVATION.value, subtype="C4 Return Evidence Competition", operator_name="low_utility_redundancy", default_strength=OperatorStrength.MILD, default_stage_target=StageTarget.S4.value, primary_failure_risks=SUBTYPE_FAILURE_RISKS["C4 Return Evidence Competition"], intrinsic_strength_score=0.14, subtype_affinity=["C4 Return Evidence Competition"], severity_fit=["heavy"], exposure_axis="evidence_competition", stack_role="amplifier", can_stack_with=["distractor_candidate", "misleading_note"], minimum_effect_expectation=0.03, maximum_effect_expectation=0.14, notes="Duplicate low-value fields in a way that pushes the observation toward a competing evidence path."),
                            OperatorSpec(family=BenchmarkFamily.OBSERVATION.value, subtype="C4 Return Evidence Competition", operator_name="spurious_auxiliary_field", default_strength=OperatorStrength.STRONG, default_stage_target=StageTarget.S4.value, primary_failure_risks=SUBTYPE_FAILURE_RISKS["C4 Return Evidence Competition"], intrinsic_strength_score=0.32, subtype_affinity=["C4 Return Evidence Competition"], severity_fit=["heavy"], exposure_axis="evidence_competition", stack_role="amplifier", can_stack_with=["distractor_candidate"], minimum_effect_expectation=0.10, maximum_effect_expectation=0.34, notes="Attach an auxiliary field that competes with the main evidence path."),
                        ],
                    ),
                ]
            ),
        ),
        (
            BenchmarkFamily.EXECUTION_ENVIRONMENT.value,
            OrderedDict(
                [
                    (
                        "D1 Availability Failure",
                        [
                            OperatorSpec(family=BenchmarkFamily.EXECUTION_ENVIRONMENT.value, subtype="D1 Availability Failure", operator_name="cached_availability_fallback", default_strength=OperatorStrength.MILD, default_stage_target=StageTarget.S4.value, primary_failure_risks=SUBTYPE_FAILURE_RISKS["D1 Availability Failure"], intrinsic_strength_score=0.13, subtype_affinity=["D1 Availability Failure"], severity_fit=["light"], exposure_axis="availability", stack_role="primary", minimum_effect_expectation=0.02, maximum_effect_expectation=0.14, notes="Return a usable payload through a transparent fallback path."),
                            OperatorSpec(family=BenchmarkFamily.EXECUTION_ENVIRONMENT.value, subtype="D1 Availability Failure", operator_name="service_unavailable_with_stale_cache", default_strength=OperatorStrength.STRONG, default_stage_target=StageTarget.S4.value, primary_failure_risks=SUBTYPE_FAILURE_RISKS["D1 Availability Failure"], intrinsic_strength_score=0.26, subtype_affinity=["D1 Availability Failure"], severity_fit=["medium", "heavy"], exposure_axis="availability", stack_role="primary", should_not_stack_with=["cached_availability_fallback"], can_stack_with=["timeout_after_partial_stream"], minimum_effect_expectation=0.08, maximum_effect_expectation=0.30, notes="Expose a transient service outage with a stale cache snapshot."),
                            OperatorSpec(family=BenchmarkFamily.EXECUTION_ENVIRONMENT.value, subtype="D1 Availability Failure", operator_name="timeout_after_partial_stream", default_strength=OperatorStrength.STRONG, default_stage_target=StageTarget.S4.value, primary_failure_risks=SUBTYPE_FAILURE_RISKS["D1 Availability Failure"], intrinsic_strength_score=0.42, subtype_affinity=["D1 Availability Failure"], severity_fit=["heavy"], exposure_axis="availability", stack_role="primary", minimum_effect_expectation=0.14, maximum_effect_expectation=0.44, notes="Simulate a timeout after partial frames without making the packet unrecoverable."),
                        ],
                    ),
                    (
                        "D2 Runtime Variability",
                        [
                            OperatorSpec(family=BenchmarkFamily.EXECUTION_ENVIRONMENT.value, subtype="D2 Runtime Variability", operator_name="jittered_metadata", default_strength=OperatorStrength.MILD, default_stage_target=StageTarget.S4.value, primary_failure_risks=SUBTYPE_FAILURE_RISKS["D2 Runtime Variability"], intrinsic_strength_score=0.13, subtype_affinity=["D2 Runtime Variability"], severity_fit=["light"], exposure_axis="runtime_variability", stack_role="primary", minimum_effect_expectation=0.02, maximum_effect_expectation=0.14, notes="Add delivery metadata around a complete payload."),
                            OperatorSpec(family=BenchmarkFamily.EXECUTION_ENVIRONMENT.value, subtype="D2 Runtime Variability", operator_name="delayed_fresh_partial_with_stale_full", default_strength=OperatorStrength.STRONG, default_stage_target=StageTarget.S4.value, primary_failure_risks=SUBTYPE_FAILURE_RISKS["D2 Runtime Variability"], intrinsic_strength_score=0.28, subtype_affinity=["D2 Runtime Variability"], severity_fit=["medium", "heavy"], exposure_axis="runtime_variability", stack_role="primary", can_stack_with=["flaky_shape_conflicting_attempts", "contaminated_flaky_attempts"], minimum_effect_expectation=0.08, maximum_effect_expectation=0.32, notes="Return a stale complete candidate beside a fresher partial fragment."),
                            OperatorSpec(family=BenchmarkFamily.EXECUTION_ENVIRONMENT.value, subtype="D2 Runtime Variability", operator_name="flaky_shape_conflicting_attempts", default_strength=OperatorStrength.STRONG, default_stage_target=StageTarget.S4.value, primary_failure_risks=SUBTYPE_FAILURE_RISKS["D2 Runtime Variability"], intrinsic_strength_score=0.18, subtype_affinity=["D2 Runtime Variability"], severity_fit=["medium", "heavy"], exposure_axis="runtime_variability", stack_role="amplifier", can_stack_with=["delayed_fresh_partial_with_stale_full"], minimum_effect_expectation=0.06, maximum_effect_expectation=0.24, notes="Vary the runtime envelope while keeping the fresh partial path visible."),
                            OperatorSpec(family=BenchmarkFamily.EXECUTION_ENVIRONMENT.value, subtype="D2 Runtime Variability", operator_name="contaminated_flaky_attempts", default_strength=OperatorStrength.STRONG, default_stage_target=StageTarget.S4.value, primary_failure_risks=SUBTYPE_FAILURE_RISKS["D2 Runtime Variability"], intrinsic_strength_score=0.32, subtype_affinity=["D2 Runtime Variability"], severity_fit=["heavy"], exposure_axis="runtime_variability", stack_role="amplifier", can_stack_with=["delayed_fresh_partial_with_stale_full"], minimum_effect_expectation=0.12, maximum_effect_expectation=0.40, notes="Contaminate rotating attempts with a plausible stale replay and a low-confidence decoy."),
                        ],
                    ),
                    (
                        "D3 Failure Feedback Quality",
                        [
                            OperatorSpec(family=BenchmarkFamily.EXECUTION_ENVIRONMENT.value, subtype="D3 Failure Feedback Quality", operator_name="vague_retry_hint", default_strength=OperatorStrength.MILD, default_stage_target=StageTarget.S4.value, primary_failure_risks=SUBTYPE_FAILURE_RISKS["D3 Failure Feedback Quality"], intrinsic_strength_score=0.13, subtype_affinity=["D3 Failure Feedback Quality"], severity_fit=["light"], exposure_axis="failure_feedback", stack_role="primary", minimum_effect_expectation=0.02, maximum_effect_expectation=0.14, notes="Return a vague error with a retryability hint."),
                            OperatorSpec(family=BenchmarkFamily.EXECUTION_ENVIRONMENT.value, subtype="D3 Failure Feedback Quality", operator_name="wrong_failure_label_with_evidence", default_strength=OperatorStrength.STRONG, default_stage_target=StageTarget.S4.value, primary_failure_risks=SUBTYPE_FAILURE_RISKS["D3 Failure Feedback Quality"], intrinsic_strength_score=0.24, subtype_affinity=["D3 Failure Feedback Quality"], severity_fit=["medium"], exposure_axis="failure_feedback", stack_role="primary", should_not_stack_with=["vague_retry_hint"], minimum_effect_expectation=0.08, maximum_effect_expectation=0.28, notes="Expose evidence that softens a reported failure label."),
                            OperatorSpec(family=BenchmarkFamily.EXECUTION_ENVIRONMENT.value, subtype="D3 Failure Feedback Quality", operator_name="hidden_partial_wrong_label", default_strength=OperatorStrength.STRONG, default_stage_target=StageTarget.S4.value, primary_failure_risks=SUBTYPE_FAILURE_RISKS["D3 Failure Feedback Quality"], intrinsic_strength_score=0.40, subtype_affinity=["D3 Failure Feedback Quality"], severity_fit=["medium", "heavy"], exposure_axis="failure_feedback", stack_role="primary", can_stack_with=["conflicting_diagnostic_decoy"], minimum_effect_expectation=0.12, maximum_effect_expectation=0.42, notes="Hide a usable partial fragment under a wrong failure label while leaving recovery cues visible."),
                            OperatorSpec(family=BenchmarkFamily.EXECUTION_ENVIRONMENT.value, subtype="D3 Failure Feedback Quality", operator_name="conflicting_diagnostic_decoy", default_strength=OperatorStrength.STRONG, default_stage_target=StageTarget.S4.value, primary_failure_risks=SUBTYPE_FAILURE_RISKS["D3 Failure Feedback Quality"], intrinsic_strength_score=0.22, subtype_affinity=["D3 Failure Feedback Quality"], severity_fit=["heavy"], exposure_axis="failure_feedback", stack_role="amplifier", can_stack_with=["hidden_partial_wrong_label"], minimum_effect_expectation=0.08, maximum_effect_expectation=0.32, notes="Add a plausible diagnostic decoy that competes with the hidden partial evidence."),
                        ],
                    ),
                ]
            ),
        ),
    ]
)


def family_order() -> list[str]:
    return list(FAMILY_CATALOG.keys())


def subtype_order(enabled_families: list[str], enabled_subtypes: list[str] | str) -> list[str]:
    order: list[str] = []
    allowed = None if enabled_subtypes == "all" else set(enabled_subtypes)
    for family in enabled_families:
        for subtype in FAMILY_CATALOG[family]:
            if allowed is None or subtype in allowed:
                order.append(subtype)
    return order


def operator_index() -> dict[str, OperatorSpec]:
    return {
        spec.operator_name: spec
        for family in FAMILY_CATALOG.values()
        for specs in family.values()
        for spec in specs
    }
