"""Base backend abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod

from toolrobust.domain.schemas import BackendConfig, BackendResponse, ParsedInvocation, TaskSample, ToolSpec


class BackendBase(ABC):
    def __init__(self, config: BackendConfig) -> None:
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name."""

    @abstractmethod
    def predict(self, task: TaskSample, tool_registry: list[ToolSpec]) -> BackendResponse:
        """Return a normalized prediction for a single task."""

    def build_instruction(self, task: TaskSample) -> str:
        return (
            "You are a single-step tool caller. Choose exactly one tool and emit JSON with keys "
            f"tool_name and arguments. Task: {task.user_instruction}"
        )

    def default_parsed(
        self,
        *,
        raw_response: dict[str, object],
        predicted_tool_raw: str | None = None,
        predicted_args_raw: dict[str, object] | None = None,
        call_present: bool = False,
        format_valid: bool = False,
        parser_message: str = "",
        latency_ms: float | None = None,
        cost_usd: float | None = None,
        auxiliary: dict[str, object] | None = None,
    ) -> BackendResponse:
        return BackendResponse(
            backend_name=self.name,
            model_name=self.config.model_name,
            raw_response=raw_response,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            auxiliary=auxiliary or {},
            parsed=ParsedInvocation(
                predicted_tool_raw=predicted_tool_raw,
                predicted_args_raw=predicted_args_raw or {},
                call_present=call_present,
                format_valid=format_valid,
                parser_message=parser_message,
            ),
        )
