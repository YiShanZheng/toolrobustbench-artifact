"""Enumeration types used across ToolRobust v3."""

from __future__ import annotations

from enum import Enum


class DatasetSplit(str, Enum):
    CLEAN = "clean"
    PERTURBED = "perturbed"


class BenchmarkFamily(str, Enum):
    CLEAN = "clean"
    CROSS_FAMILY = "cross_family"
    INTERFACE_EXPOSURE = "interface_exposure"
    USER_INTENT = "user_intent"
    OBSERVATION = "observation"
    EXECUTION_ENVIRONMENT = "execution_environment"


class SeverityLevel(str, Enum):
    CLEAN = "clean"
    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"


class OperatorStrength(str, Enum):
    MILD = "mild"
    STRONG = "strong"


class StageTarget(str, Enum):
    CLEAN = "clean"
    S1 = "S1 Tool Selection"
    S2 = "S2 Interface Grounding"
    S3 = "S3 Parameter Binding"
    S4 = "S4 Observation / Execution Handling"
    S5 = "S5 End-to-End Task Success"


class FailureTag(str, Enum):
    NO_CALL = "no-call"
    WRONG_TOOL = "wrong-tool"
    INVALID_FORMAT = "invalid-format"
    MISSING_PARAMETER = "missing-parameter"
    WRONG_PARAMETER = "wrong-parameter"
    EXECUTION_FAIL = "execution-fail"


class OutputProfile(str, Enum):
    LITE = "lite"
    FULL = "full"


SEVERITY_ORDER = [
    SeverityLevel.CLEAN.value,
    SeverityLevel.LIGHT.value,
    SeverityLevel.MEDIUM.value,
    SeverityLevel.HEAVY.value,
]


FAMILY_ORDER = [
    BenchmarkFamily.INTERFACE_EXPOSURE.value,
    BenchmarkFamily.USER_INTENT.value,
    BenchmarkFamily.OBSERVATION.value,
    BenchmarkFamily.EXECUTION_ENVIRONMENT.value,
]


STAGE_ORDER = [
    StageTarget.S1.value,
    StageTarget.S2.value,
    StageTarget.S3.value,
    StageTarget.S4.value,
    StageTarget.S5.value,
]
