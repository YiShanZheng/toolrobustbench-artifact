"""Markdown summary generation for ToolRobust v3."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from toolrobust.domain.schemas import RunManifest


class MarkdownReportGenerator:
    def generate(self, path: Path, *, manifest: RunManifest, tables: dict[str, pd.DataFrame]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# ToolRobust v3 Summary",
            "",
            "## Scope",
            "",
            f"- benchmark_version: `{manifest.benchmark_version}`",
            f"- operator_catalog_version: `{manifest.operator_catalog_version}`",
            f"- experiment_name: `{manifest.experiment_name}`",
            f"- models: `{', '.join(manifest.models)}`",
            f"- task_scope: `{manifest.dataset['task_scope']}`",
            f"- enabled_families: `{', '.join(manifest.enabled_families)}`",
            f"- enabled_subtypes: `{manifest.enabled_subtypes}`",
            f"- disabled_operators: `{', '.join(manifest.disabled_operators) if manifest.disabled_operators else 'none'}`",
            f"- summary_overall semantics: current scope overall only",
            "",
            "## summary_overall",
            "",
            tables["summary_overall"].to_markdown(index=False),
            "",
            "## summary_by_family",
            "",
            tables["summary_by_family"].to_markdown(index=False),
            "",
            "## summary_by_subtype",
            "",
            tables["summary_by_subtype"].to_markdown(index=False),
            "",
            "## summary_by_stage",
            "",
            tables["summary_by_stage"].to_markdown(index=False),
        ]
        if not tables["summary_by_operator"].empty:
            lines.extend(["", "## summary_by_operator", "", tables["summary_by_operator"].to_markdown(index=False)])
        if not tables["summary_by_composition"].empty:
            lines.extend(["", "## summary_by_composition", "", tables["summary_by_composition"].to_markdown(index=False)])
        if not tables["interaction_effect"].empty:
            lines.extend(["", "## interaction_effect", "", tables["interaction_effect"].to_markdown(index=False)])
        if manifest.preflight_gate_rows:
            lines.extend(["", "## preflight_severity_gate_report", "", pd.DataFrame(manifest.preflight_gate_rows).to_markdown(index=False)])
        if not tables["severity_calibration_report"].empty:
            lines.extend(["", "## severity_calibration_report", "", tables["severity_calibration_report"].to_markdown(index=False)])
        if not tables["failure_taxonomy"].empty:
            lines.extend(["", "## failure_taxonomy", "", tables["failure_taxonomy"].to_markdown(index=False)])
        if not tables["primary_failure_breakdown"].empty:
            lines.extend(["", "## primary_failure_breakdown", "", tables["primary_failure_breakdown"].to_markdown(index=False)])
        if not tables["cascade_failure_breakdown"].empty:
            lines.extend(["", "## cascade_failure_breakdown", "", tables["cascade_failure_breakdown"].to_markdown(index=False)])
        if not tables["failure_by_composition"].empty:
            lines.extend(["", "## failure_by_composition", "", tables["failure_by_composition"].to_markdown(index=False)])
        if not tables["family_boundary_report"].empty:
            lines.extend(["", "## family_boundary_report", "", tables["family_boundary_report"].to_markdown(index=False)])
        elif manifest.cross_family.get("enabled") and not manifest.cross_family.get("include_single_family_baseline", True):
            lines.extend(
                [
                    "",
                    "## family_boundary_report",
                    "",
                    "No single-family perturbed samples were included in this cross-family-only run, so the family boundary report is intentionally empty.",
                ]
            )
        path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
