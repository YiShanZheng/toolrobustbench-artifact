"""Base classes for local deterministic tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from toolrobust.domain.schemas import ToolSpec


class ToolExecutionError(RuntimeError):
    """Raised when a local tool invocation cannot be executed."""


class LocalTool(ABC):
    """Abstract base class for deterministic local tools."""

    @property
    @abstractmethod
    def spec(self) -> ToolSpec:
        """Return the serializable tool specification."""

    @abstractmethod
    def execute(self, args: dict[str, Any]) -> Any:
        """Execute the tool with validated arguments."""
