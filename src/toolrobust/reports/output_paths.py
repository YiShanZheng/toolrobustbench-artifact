"""Result-directory naming and output path construction for ToolRobust v3."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class RunOutputPaths:
    run_name: str
    results_dir: Path
    plots_dir: Path
    records_csv: Path
    summary_overall_csv: Path
    summary_by_family_csv: Path
    summary_by_subtype_csv: Path
    summary_by_stage_csv: Path
    summary_by_operator_csv: Path
    summary_by_composition_csv: Path
    interaction_effect_csv: Path
    preflight_severity_gate_report_csv: Path
    severity_calibration_report_csv: Path
    failure_taxonomy_csv: Path
    primary_failure_breakdown_csv: Path
    cascade_failure_breakdown_csv: Path
    failure_by_family_csv: Path
    failure_by_composition_csv: Path
    paired_case_dump_csv: Path
    family_boundary_report_csv: Path
    run_manifest_json: Path
    summary_md: Path


def sanitize_run_name(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip().lower())
    sanitized = re.sub(r"-+", "-", sanitized).strip("-._")
    return sanitized or "unknown"


def resolve_results_dir(output_root: Path, *, experiment_name: str, explicit_results_dir: Path | None = None) -> Path:
    if explicit_results_dir is not None:
        return explicit_results_dir
    return output_root / f"{sanitize_run_name(experiment_name)}-results"


def build_run_output_paths(output_root: Path, *, experiment_name: str, explicit_results_dir: Path | None = None) -> RunOutputPaths:
    results_dir = resolve_results_dir(output_root, experiment_name=experiment_name, explicit_results_dir=explicit_results_dir)
    run_name = results_dir.name.removesuffix("-results")
    return RunOutputPaths(
        run_name=run_name,
        results_dir=results_dir,
        plots_dir=results_dir / "plots",
        records_csv=results_dir / "records.csv",
        summary_overall_csv=results_dir / "summary_overall.csv",
        summary_by_family_csv=results_dir / "summary_by_family.csv",
        summary_by_subtype_csv=results_dir / "summary_by_subtype.csv",
        summary_by_stage_csv=results_dir / "summary_by_stage.csv",
        summary_by_operator_csv=results_dir / "summary_by_operator.csv",
        summary_by_composition_csv=results_dir / "summary_by_composition.csv",
        interaction_effect_csv=results_dir / "interaction_effect.csv",
        preflight_severity_gate_report_csv=results_dir / "preflight_severity_gate_report.csv",
        severity_calibration_report_csv=results_dir / "severity_calibration_report.csv",
        failure_taxonomy_csv=results_dir / "failure_taxonomy.csv",
        primary_failure_breakdown_csv=results_dir / "primary_failure_breakdown.csv",
        cascade_failure_breakdown_csv=results_dir / "cascade_failure_breakdown.csv",
        failure_by_family_csv=results_dir / "failure_by_family.csv",
        failure_by_composition_csv=results_dir / "failure_by_composition.csv",
        paired_case_dump_csv=results_dir / "paired_case_dump.csv",
        family_boundary_report_csv=results_dir / "family_boundary_report.csv",
        run_manifest_json=results_dir / "run_manifest.json",
        summary_md=results_dir / "summary.md",
    )
