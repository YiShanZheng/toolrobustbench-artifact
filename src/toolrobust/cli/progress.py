"""Lightweight single-line progress reporting for CLI commands.

The reporter uses ``\\r`` to refresh one terminal line so VSCode's PowerShell
terminal does not accumulate progress spam. A background heartbeat thread keeps
the line alive even when a single API request blocks for a while.
"""

from __future__ import annotations

import shutil
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any


COMMAND_STAGE_WEIGHTS: dict[str, dict[str, float]] = {
    "build-data": {
        "loading_config": 0.05,
        "generating_clean_samples": 0.35,
        "generating_perturbations": 0.45,
        "writing_dataset_outputs": 0.15,
    },
    "run": {
        "loading_config": 0.02,
        "generating_clean_samples": 0.18,
        "generating_perturbations": 0.22,
        "writing_dataset_outputs": 0.03,
        "running_inference": 0.75,
        "aggregating_metrics": 0.04,
        "writing_outputs": 0.03,
        "generating_plots": 0.03,
        "done": 0.0,
    },
    "plot": {
        "loading_config": 0.05,
        "loading_results": 0.20,
        "generating_plots": 0.70,
        "writing_outputs": 0.05,
        "done": 0.0,
    },
}


@dataclass(slots=True)
class _ProgressState:
    stage: str = "idle"
    stage_completed: int = 0
    stage_total: int = 0
    total_percent: float = 0.0
    current_item: str = ""
    started_at: float = 0.0
    last_call_seconds: float | None = None


class NullProgressReporter:
    """No-op progress reporter used when progress is disabled."""

    def start(self, stage: str, total_items: int | None = None, current_item: str | None = None) -> None:
        return

    def update(
        self,
        *,
        stage_completed: int | None = None,
        current_item: str | None = None,
        total_percent: float | None = None,
    ) -> None:
        return

    def advance(self, *, step: int = 1, current_item: str | None = None) -> None:
        return

    def finish_stage(self, message: str | None = None) -> None:
        return

    def finish_all(self, message: str | None = None) -> None:
        return

    def record_api_call(self, elapsed_seconds: float) -> None:
        return


class SingleLineProgressReporter:
    """Single-line progress reporter with periodic heartbeat refresh."""

    def __init__(
        self,
        *,
        command_name: str,
        interval_seconds: float = 5.0,
        progress_label: str | None = None,
        progress_offset: int = 0,
        progress_total: int = 0,
    ) -> None:
        self.command_name = command_name
        self.interval_seconds = interval_seconds
        self.progress_label = progress_label or ""
        self.progress_offset = max(0, int(progress_offset))
        self.progress_total = max(0, int(progress_total))
        self.stage_weights = COMMAND_STAGE_WEIGHTS[command_name]
        self.state = _ProgressState(started_at=time.time())
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._line_width = 0
        self._stage_order = list(self.stage_weights.keys())
        self._stream = sys.stderr
        self._supports_ansi = bool(getattr(self._stream, "isatty", lambda: False)())
        self._heartbeat_thread.start()

    def start(self, stage: str, total_items: int | None = None, current_item: str | None = None) -> None:
        with self._lock:
            self.state.stage = stage
            self.state.stage_completed = 0
            self.state.stage_total = total_items or 0
            self.state.current_item = current_item or ""
            self.state.total_percent = self._compute_total_percent(stage, 0, total_items or 0)
        self._render()

    def update(
        self,
        *,
        stage_completed: int | None = None,
        current_item: str | None = None,
        total_percent: float | None = None,
    ) -> None:
        with self._lock:
            if stage_completed is not None:
                self.state.stage_completed = stage_completed
            if current_item is not None:
                self.state.current_item = current_item
            if total_percent is not None:
                self.state.total_percent = total_percent
            else:
                self.state.total_percent = self._compute_total_percent(
                    self.state.stage,
                    self.state.stage_completed,
                    self.state.stage_total,
                )
        self._render()

    def advance(self, *, step: int = 1, current_item: str | None = None) -> None:
        with self._lock:
            self.state.stage_completed += step
            if current_item is not None:
                self.state.current_item = current_item
            self.state.total_percent = self._compute_total_percent(
                self.state.stage,
                self.state.stage_completed,
                self.state.stage_total,
            )
        self._render()

    def record_api_call(self, elapsed_seconds: float) -> None:
        with self._lock:
            self.state.last_call_seconds = max(0.0, float(elapsed_seconds))
        self._render()

    def finish_stage(self, message: str | None = None) -> None:
        with self._lock:
            if self.state.stage_total > 0:
                self.state.stage_completed = self.state.stage_total
            self.state.total_percent = self._compute_total_percent(
                self.state.stage,
                self.state.stage_completed,
                self.state.stage_total,
            )
        self._render()
        with self._lock:
            self.state.stage = "idle"
            self.state.stage_completed = 0
            self.state.stage_total = 0
            self.state.current_item = ""

    def finish_all(self, message: str | None = None) -> None:
        with self._lock:
            self.state.stage = "done"
            self.state.total_percent = 100.0
            self.state.current_item = "completed"
        self._render()
        self._stop_event.set()
        self._heartbeat_thread.join(timeout=self.interval_seconds + 1)
        self._stream.write("\n")
        self._stream.flush()

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            self._render()

    def _render(self) -> None:
        with self._lock:
            line = self._fit_to_terminal(self._format_line())
            padded = line + " " * max(0, self._line_width - len(line))
            self._line_width = max(self._line_width, len(line))
            prefix = "\r\x1b[2K" if self._supports_ansi else "\r"
            self._stream.write(prefix + padded)
            self._stream.flush()

    def _format_line(self) -> str:
        elapsed_seconds = int(time.time() - self.state.started_at)
        elapsed = f"{elapsed_seconds // 60:02d}:{elapsed_seconds % 60:02d}"
        stage_progress = "?"
        if self.state.stage_total > 0:
            stage_progress = f"{self.state.stage_completed}/{self.state.stage_total}"
        elif self.state.stage_completed > 0:
            stage_progress = str(self.state.stage_completed)
        if self.progress_label and self.state.stage == "running_inference":
            api_elapsed = self._format_call_elapsed(self.state.last_call_seconds)
            if self.progress_total > 0:
                global_completed = min(self.progress_total, self.progress_offset + self.state.stage_completed)
                global_progress = f"{global_completed}/{self.progress_total}"
            else:
                global_progress = "?"
            return f"【progress】{self.progress_label}|{stage_progress}|{global_progress}|{api_elapsed}"
        line = (
            f"[progress] stage={self.state.stage} | "
            f"stage_progress={stage_progress} | "
            f"total={self.state.total_percent:0.1f}% | "
            f"elapsed={elapsed}"
        )
        if self.state.current_item:
            line += f" | current={self.state.current_item}"
        return line

    @staticmethod
    def _format_call_elapsed(elapsed_seconds: float | None) -> str:
        seconds = int(elapsed_seconds or 0)
        return f"{seconds // 60:02d}：{seconds % 60:02d}"

    def _fit_to_terminal(self, line: str) -> str:
        columns = max(40, shutil.get_terminal_size(fallback=(120, 20)).columns)
        if len(line) <= columns - 1:
            return line
        max_len = max(20, columns - 1)
        if max_len <= 3:
            return line[:max_len]
        return line[: max_len - 3] + "..."

    def _compute_total_percent(self, stage: str, stage_completed: int, stage_total: int) -> float:
        completed_weight = 0.0
        for ordered_stage in self._stage_order:
            if ordered_stage == stage:
                break
            completed_weight += self.stage_weights.get(ordered_stage, 0.0)
        stage_weight = self.stage_weights.get(stage, 0.0)
        stage_fraction = 0.0
        if stage_total > 0:
            stage_fraction = min(1.0, max(0.0, stage_completed / stage_total))
        return (completed_weight + stage_weight * stage_fraction) * 100

def build_progress_reporter(
    *,
    command_name: str,
    enabled: bool,
    interval_seconds: float,
    progress_label: str | None = None,
    progress_offset: int = 0,
    progress_total: int = 0,
) -> SingleLineProgressReporter | NullProgressReporter:
    """Build either a real or no-op progress reporter."""

    if not enabled or not getattr(sys.stderr, "isatty", lambda: False)():
        return NullProgressReporter()
    return SingleLineProgressReporter(
        command_name=command_name,
        interval_seconds=interval_seconds,
        progress_label=progress_label,
        progress_offset=progress_offset,
        progress_total=progress_total,
    )
