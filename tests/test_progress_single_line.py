from __future__ import annotations

import io
import sys

from toolrobust.cli.progress import SingleLineProgressReporter


def test_progress_reporter_keeps_single_line_until_finish(monkeypatch) -> None:
    buffer = io.StringIO()
    monkeypatch.setattr(sys, "stderr", buffer)
    reporter = SingleLineProgressReporter(command_name="run", interval_seconds=60.0)
    reporter.start("generating_clean_samples", total_items=2, current_item="sample_a")
    reporter.advance(current_item="sample_b")
    reporter.finish_stage()
    reporter.start("running_inference", total_items=2, current_item="task_a")
    reporter.advance(current_item="task_b")
    reporter.finish_stage()
    reporter.finish_all()
    output = buffer.getvalue()
    assert output.count("\n") == 1


def test_progress_reporter_uses_clear_line_escape_on_tty(monkeypatch) -> None:
    class _TTYBuffer(io.StringIO):
        def isatty(self) -> bool:
            return True

    buffer = _TTYBuffer()
    monkeypatch.setattr(sys, "stderr", buffer)
    reporter = SingleLineProgressReporter(command_name="run", interval_seconds=60.0)
    reporter.start("running_inference", total_items=1, current_item="task_a")
    reporter.finish_all()
    output = buffer.getvalue()
    assert "\x1b[2K" in output


def test_progress_reporter_formats_batch_inference_line(monkeypatch) -> None:
    buffer = io.StringIO()
    monkeypatch.setattr(sys, "stderr", buffer)
    reporter = SingleLineProgressReporter(
        command_name="run",
        interval_seconds=60.0,
        progress_label="gpt-5.4-mini-interface_exposure",
        progress_offset=17,
        progress_total=2063,
    )
    reporter.start("running_inference", total_items=463, current_item="task_a")
    reporter.record_api_call(36.4)
    reporter.advance(current_item="task_b")
    reporter.finish_all()
    output = buffer.getvalue()
    assert "【progress】gpt-5.4-mini-interface_exposure|1/463|18/2063|00：36" in output
