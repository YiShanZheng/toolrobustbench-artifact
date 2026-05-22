"""Metric aggregation utilities for ToolRobust v3."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from toolrobust.domain.enums import BenchmarkFamily, FailureTag, SEVERITY_ORDER, SeverityLevel, StageTarget
from toolrobust.domain.schemas import RunManifest

METRIC_COLUMNS = [
    "sample_count",
    "call_rate",
    "interface_name_accuracy",
    "canonical_tool_accuracy",
    "format_valid_rate",
    "missing_parameter_rate",
    "wrong_parameter_rate",
    "execution_success_rate",
    "overall_task_success_rate",
]

PAIR_COLUMNS = [
    "paired_clean_sample_count",
    "paired_clean_success_rate",
    "paired_success_flip_rate",
]


class BenchmarkMetricCalculator:
    def __init__(self, manifest: RunManifest, keep_missing_buckets: bool = True) -> None:
        self.manifest = manifest
        self.keep_missing_buckets = keep_missing_buckets

    def calculate(self, frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
        paired = self._build_paired_frame(frame)
        summary_overall = self._summary_overall(frame, paired)
        clean_map = (
            summary_overall.loc[summary_overall["condition"] == "clean", ["model_name", "overall_task_success_rate"]]
            .set_index("model_name")["overall_task_success_rate"]
            .to_dict()
        )
        return {
            "summary_overall": summary_overall,
            "summary_by_family": self._summary_by_family(frame, paired, clean_map),
            "summary_by_subtype": self._summary_by_subtype(frame, paired, clean_map),
            "summary_by_stage": self._summary_by_stage(frame, paired, clean_map),
            "summary_by_operator": self._summary_by_operator(frame, paired),
            "summary_by_composition": self._summary_by_composition(frame, paired),
            "interaction_effect": self._interaction_effect(frame),
            "severity_calibration_report": self._severity_calibration_report(frame, paired),
            "failure_taxonomy": self._failure_taxonomy(frame),
            "primary_failure_breakdown": self._primary_failure_breakdown(frame),
            "cascade_failure_breakdown": self._cascade_failure_breakdown(frame),
            "failure_by_family": self._failure_by_family(frame),
            "failure_by_composition": self._failure_by_composition(frame),
            "paired_case_dump": self._paired_case_dump(paired),
            "family_boundary_report": self._family_boundary_report(frame),
        }

    def _summary_overall(self, frame: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
        grouped = self._aggregate(frame, ["model_name", "severity"]).rename(columns={"severity": "condition"})
        grouped = self._order(grouped, "condition", ["model_name"])
        clean_ref = grouped.loc[grouped["condition"] == "clean", ["model_name", "overall_task_success_rate"]].rename(
            columns={"overall_task_success_rate": "clean_overall_task_success_rate"}
        )
        grouped = grouped.merge(clean_ref, on="model_name", how="left")
        paired_metrics = self._aggregate_pair_metrics(paired, ["model_name", "severity"]).rename(columns={"severity": "condition"})
        grouped = grouped.merge(paired_metrics, on=["model_name", "condition"], how="left")
        grouped = self._finalize_paired_columns(grouped)
        grouped["paired_clean_sample_count"] = grouped["paired_clean_sample_count"].astype("int64")
        clean_mask = grouped["condition"] == "clean"
        grouped.loc[clean_mask, "paired_clean_sample_count"] = grouped.loc[clean_mask, "sample_count"]
        grouped.loc[clean_mask, "paired_clean_success_rate"] = grouped.loc[clean_mask, "overall_task_success_rate"]
        grouped.loc[clean_mask, "paired_success_flip_rate"] = 0.0
        grouped["robustness_drop_vs_clean"] = grouped["paired_clean_success_rate"] - grouped["overall_task_success_rate"]
        grouped["scope_label"] = self.manifest.dataset["task_scope"]
        return grouped

    def _summary_by_family(self, frame: pd.DataFrame, paired: pd.DataFrame, clean_map: dict[str, float]) -> pd.DataFrame:
        perturbed = frame.loc[frame["family"] != BenchmarkFamily.CLEAN.value].copy()
        grouped = self._aggregate(perturbed, ["model_name", "family", "severity"])
        paired_metrics = self._aggregate_pair_metrics(paired, ["model_name", "family", "severity"])
        grouped = grouped.merge(paired_metrics, on=["model_name", "family", "severity"], how="left")
        grouped = self._fill_family_buckets(grouped)
        grouped["paired_clean_sample_count"] = grouped["paired_clean_sample_count"].astype("int64")
        clean_counts = frame.loc[frame["severity"] == "clean"].groupby("model_name")["sample_id"].count().to_dict()
        for model_name, clean_value in clean_map.items():
            mask = (grouped["model_name"] == model_name) & (grouped["severity"] == "clean")
            grouped.loc[mask, "sample_count"] = clean_counts.get(model_name, 0)
            grouped.loc[mask, ["call_rate", "interface_name_accuracy", "canonical_tool_accuracy", "format_valid_rate", "execution_success_rate", "overall_task_success_rate"]] = clean_value
            grouped.loc[mask, ["missing_parameter_rate", "wrong_parameter_rate"]] = 0.0
            grouped.loc[mask, "paired_clean_sample_count"] = clean_counts.get(model_name, 0)
            grouped.loc[mask, "paired_clean_success_rate"] = clean_value
            grouped.loc[mask, "paired_success_flip_rate"] = 0.0
        grouped = self._finalize_paired_columns(grouped)
        grouped["drop_vs_clean"] = grouped["paired_clean_success_rate"] - grouped["overall_task_success_rate"]
        grouped["scope_label"] = self.manifest.dataset["task_scope"]
        return grouped

    def _summary_by_subtype(self, frame: pd.DataFrame, paired: pd.DataFrame, clean_map: dict[str, float]) -> pd.DataFrame:
        perturbed = frame.loc[frame["family"] != BenchmarkFamily.CLEAN.value].copy()
        grouped = self._aggregate(perturbed, ["model_name", "family", "subtype", "stage_target", "severity"])
        paired_metrics = self._aggregate_pair_metrics(paired, ["model_name", "family", "subtype", "stage_target", "severity"])
        grouped = grouped.merge(paired_metrics, on=["model_name", "family", "subtype", "stage_target", "severity"], how="left")
        grouped = self._fill_subtype_buckets(grouped)
        grouped["paired_clean_sample_count"] = grouped["paired_clean_sample_count"].astype("int64")
        clean_counts = frame.loc[frame["severity"] == "clean"].groupby("model_name")["sample_id"].count().to_dict()
        for model_name, clean_value in clean_map.items():
            mask = (grouped["model_name"] == model_name) & (grouped["severity"] == "clean")
            grouped.loc[mask, "sample_count"] = clean_counts.get(model_name, 0)
            grouped.loc[mask, ["call_rate", "interface_name_accuracy", "canonical_tool_accuracy", "format_valid_rate", "execution_success_rate", "overall_task_success_rate"]] = clean_value
            grouped.loc[mask, ["missing_parameter_rate", "wrong_parameter_rate"]] = 0.0
            grouped.loc[mask, "paired_clean_sample_count"] = clean_counts.get(model_name, 0)
            grouped.loc[mask, "paired_clean_success_rate"] = clean_value
            grouped.loc[mask, "paired_success_flip_rate"] = 0.0
        grouped = self._finalize_paired_columns(grouped)
        grouped["drop_vs_clean"] = grouped["paired_clean_success_rate"] - grouped["overall_task_success_rate"]
        grouped["scope_label"] = self.manifest.dataset["task_scope"]
        return grouped

    def _summary_by_stage(self, frame: pd.DataFrame, paired: pd.DataFrame, clean_map: dict[str, float]) -> pd.DataFrame:
        grouped = self._aggregate(frame, ["model_name", "stage_target", "severity"])
        paired_metrics = self._aggregate_pair_metrics(paired, ["model_name", "stage_target", "severity"])
        grouped = grouped.merge(paired_metrics, on=["model_name", "stage_target", "severity"], how="left")
        grouped = self._fill_stage_buckets(grouped)
        grouped["paired_clean_sample_count"] = grouped["paired_clean_sample_count"].astype("int64")

        severity_overall = self._aggregate(frame, ["model_name", "severity"])
        severity_metric_rows = severity_overall.set_index(["model_name", "severity"])[METRIC_COLUMNS].to_dict(orient="index")
        severity_counts = severity_overall.set_index(["model_name", "severity"])["sample_count"].to_dict()
        paired_severity = self._aggregate_pair_metrics(paired, ["model_name", "severity"])
        paired_severity_counts = paired_severity.set_index(["model_name", "severity"])["paired_clean_sample_count"].to_dict()
        paired_severity_success = paired_severity.set_index(["model_name", "severity"])["paired_clean_success_rate"].to_dict()
        paired_severity_flip = paired_severity.set_index(["model_name", "severity"])["paired_success_flip_rate"].to_dict()

        for model_name, clean_value in clean_map.items():
            clean_mask = (grouped["model_name"] == model_name) & (grouped["severity"] == "clean")
            grouped.loc[clean_mask, "sample_count"] = severity_counts.get((model_name, "clean"), 0)
            grouped.loc[clean_mask, ["call_rate", "interface_name_accuracy", "canonical_tool_accuracy", "format_valid_rate", "execution_success_rate", "overall_task_success_rate"]] = clean_value
            grouped.loc[clean_mask, ["missing_parameter_rate", "wrong_parameter_rate"]] = 0.0
            grouped.loc[clean_mask, "paired_clean_sample_count"] = severity_counts.get((model_name, "clean"), 0)
            grouped.loc[clean_mask, "paired_clean_success_rate"] = clean_value
            grouped.loc[clean_mask, "paired_success_flip_rate"] = 0.0

        for model_name, severity in severity_metric_rows:
            s5_mask = (
                (grouped["model_name"] == model_name)
                & (grouped["severity"] == severity)
                & (grouped["stage_target"] == StageTarget.S5.value)
            )
            grouped.loc[s5_mask, "sample_count"] = severity_counts.get((model_name, severity), 0)
            metric_row = severity_metric_rows.get((model_name, severity), {})
            for column in METRIC_COLUMNS:
                grouped.loc[s5_mask, column] = metric_row.get(column, 0.0)
            grouped.loc[s5_mask, "paired_clean_sample_count"] = paired_severity_counts.get((model_name, severity), 0)
            grouped.loc[s5_mask, "paired_clean_success_rate"] = paired_severity_success.get((model_name, severity), clean_map.get(model_name, 0.0))
            grouped.loc[s5_mask, "paired_success_flip_rate"] = paired_severity_flip.get((model_name, severity), 0.0)

        grouped = self._finalize_paired_columns(grouped)
        grouped["drop_vs_clean"] = grouped["paired_clean_success_rate"] - grouped["overall_task_success_rate"]
        zero_mask = (grouped["sample_count"] == 0) & (grouped["severity"] != SeverityLevel.CLEAN.value)
        grouped["drop_vs_clean"] = grouped["drop_vs_clean"].astype(object)
        grouped.loc[zero_mask, "drop_vs_clean"] = "N/A"
        grouped["scope_label"] = self.manifest.dataset["task_scope"]
        return grouped

    def _summary_by_operator(self, frame: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
        perturbed = frame.loc[frame["split"] == "perturbed"].copy()
        if perturbed.empty:
            return pd.DataFrame(
                columns=[
                    "model_name",
                    "family",
                    "subtype",
                    "stage_target",
                    "severity",
                    "operator_name",
                    *METRIC_COLUMNS,
                    *PAIR_COLUMNS,
                    "drop_vs_paired_clean",
                    "total_failures",
                    "primary_failure_tag_distribution",
                    "observed_failure_tag_distribution",
                    "intrinsic_strength_score",
                    "severity_fit",
                    "subtype_affinity",
                    "exposure_axis",
                    "stack_role",
                    "severity_fit_match",
                    "calibration_issue_flag",
                    "calibration_issue_reason",
                    "expected_primary_match_rate",
                    "allowed_spillover_match_rate",
                    "boundary_violation_rate",
                    "recommended_status",
                    "scope_label",
                ]
            )

        exploded = perturbed.explode("operator_names").rename(columns={"operator_names": "operator_name"})
        grouped = self._aggregate(exploded, ["model_name", "family", "subtype", "stage_target", "severity", "operator_name"])
        paired_exploded = paired.explode("operator_names").rename(columns={"operator_names": "operator_name"})
        paired_metrics = self._aggregate_pair_metrics(paired_exploded, ["model_name", "family", "subtype", "stage_target", "severity", "operator_name"])
        grouped = grouped.merge(
            paired_metrics,
            on=["model_name", "family", "subtype", "stage_target", "severity", "operator_name"],
            how="left",
        )
        grouped = self._finalize_paired_columns(grouped)
        grouped["drop_vs_paired_clean"] = grouped["paired_clean_success_rate"] - grouped["overall_task_success_rate"]

        diagnostics_rows: list[dict[str, object]] = []
        for key, group in exploded.groupby(["model_name", "family", "subtype", "stage_target", "severity", "operator_name"], dropna=False):
            failures = group.loc[group["status"] == "failure"].copy()
            diagnostics_rows.append(
                {
                    "model_name": key[0],
                    "family": key[1],
                    "subtype": key[2],
                    "stage_target": key[3],
                    "severity": key[4],
                    "operator_name": key[5],
                    "total_failures": int(len(failures)),
                    "primary_failure_tag_distribution": self._distribution_json(failures["primary_failure_tag"]),
                    "observed_failure_tag_distribution": self._distribution_json(failures["observed_failure_tag"]),
                    "expected_primary_match_rate": self._failure_boolean_rate(failures, "expected_primary_match"),
                    "allowed_spillover_match_rate": self._failure_boolean_rate(failures, "allowed_spillover_match"),
                    "boundary_violation_rate": self._failure_boolean_rate(failures, "boundary_violation_flag"),
                }
            )
        diagnostics = pd.DataFrame(diagnostics_rows)
        grouped = grouped.merge(
            diagnostics,
            on=["model_name", "family", "subtype", "stage_target", "severity", "operator_name"],
            how="left",
        )
        grouped[["intrinsic_strength_score", "severity_fit", "subtype_affinity", "exposure_axis", "stack_role"]] = grouped.apply(
            lambda row: pd.Series(self._operator_metadata_series(str(row.get("operator_name")))),
            axis=1,
        )
        calibration_lookup = self._severity_calibration_lookup(frame, paired)
        grouped["severity_fit_match"] = grouped.apply(self._severity_fit_match, axis=1)
        grouped[["calibration_issue_flag", "calibration_issue_reason"]] = grouped.apply(
            lambda row: pd.Series(self._operator_calibration_issue(row, calibration_lookup)),
            axis=1,
        )
        grouped["recommended_status"] = grouped.apply(self._recommended_operator_status, axis=1)
        grouped["scope_label"] = self.manifest.dataset["task_scope"]
        return self._order(grouped, "severity", ["model_name", "family", "subtype", "operator_name"])

    def _summary_by_composition(self, frame: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
        perturbed = frame.loc[
            (frame["split"] == "perturbed")
            & (frame["family"] == BenchmarkFamily.CROSS_FAMILY.value)
            & (frame["composition_id"].notna())
        ].copy()
        columns = [
            "model_name",
            "composition_id",
            "families_applied",
            "subtypes_applied",
            "stage_targets_applied",
            "severity",
            *METRIC_COLUMNS,
            *PAIR_COLUMNS,
            "drop_vs_paired_clean",
            "scope_label",
        ]
        if perturbed.empty:
            return pd.DataFrame(columns=columns)

        grouped = self._aggregate(perturbed, ["model_name", "composition_id", "severity"])
        paired_metrics = self._aggregate_pair_metrics(paired.loc[paired["family"] == BenchmarkFamily.CROSS_FAMILY.value], ["model_name", "composition_id", "severity"])
        grouped = grouped.merge(paired_metrics, on=["model_name", "composition_id", "severity"], how="left")
        grouped = self._finalize_paired_columns(grouped)
        grouped["drop_vs_paired_clean"] = grouped["paired_clean_success_rate"] - grouped["overall_task_success_rate"]
        metadata = (
            perturbed.groupby(["model_name", "composition_id", "severity"], dropna=False)
            .agg(
                families_applied=("families_applied", self._first_jsonish),
                subtypes_applied=("subtypes_applied", self._first_jsonish),
                stage_targets_applied=("stage_targets_applied", self._first_jsonish),
            )
            .reset_index()
        )
        grouped = grouped.merge(metadata, on=["model_name", "composition_id", "severity"], how="left")
        grouped["scope_label"] = self.manifest.dataset["task_scope"]
        grouped = grouped[columns].copy()
        return self._order(grouped, "severity", ["model_name", "composition_id"])

    def _interaction_effect(self, frame: pd.DataFrame) -> pd.DataFrame:
        cross = frame.loc[
            (frame["split"] == "perturbed")
            & (frame["family"] == BenchmarkFamily.CROSS_FAMILY.value)
            & (frame["composition_id"].notna())
        ].copy()
        columns = [
            "model_name",
            "composition_id",
            "severity",
            "families_applied",
            "composition_success_rate",
            "weakest_component_success_rate",
            "mean_component_success_rate",
            "interaction_delta_vs_weakest",
            "interaction_delta_vs_mean",
            "component_baseline_status",
        ]
        if cross.empty:
            return pd.DataFrame(columns=columns)

        single = frame.loc[
            (frame["split"] == "perturbed")
            & (frame["family"] != BenchmarkFamily.CLEAN.value)
            & (frame["family"] != BenchmarkFamily.CROSS_FAMILY.value)
        ].copy()
        single_rates = single.groupby(["model_name", "family", "severity"], dropna=False)["overall_task_success"].mean().to_dict()
        rows: list[dict[str, object]] = []
        for (model_name, composition_id, severity), group in cross.groupby(["model_name", "composition_id", "severity"], dropna=False):
            families = self._coerce_jsonish_list(group["families_applied"].iloc[0])
            component_rates = [
                float(single_rates[(model_name, family, severity)])
                for family in families
                if (model_name, family, severity) in single_rates
            ]
            composition_rate = float(group["overall_task_success"].mean())
            baseline_complete = len(component_rates) == len(families) and bool(families)
            weakest = min(component_rates) if baseline_complete else None
            mean_rate = sum(component_rates) / len(component_rates) if baseline_complete else None
            rows.append(
                {
                    "model_name": model_name,
                    "composition_id": composition_id,
                    "severity": severity,
                    "families_applied": json.dumps(families, ensure_ascii=False),
                    "composition_success_rate": composition_rate,
                    "weakest_component_success_rate": weakest,
                    "mean_component_success_rate": mean_rate,
                    "interaction_delta_vs_weakest": composition_rate - weakest if weakest is not None else None,
                    "interaction_delta_vs_mean": composition_rate - mean_rate if mean_rate is not None else None,
                    "component_baseline_status": "complete" if baseline_complete else "missing_component_baseline",
                }
            )
        return self._order(pd.DataFrame(rows), "severity", ["model_name", "composition_id"])

    def _severity_calibration_report(self, frame: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
        calibrated_families = {str(row.get("family")) for row in self.manifest.preflight_gate_rows}
        perturbed = frame.loc[
            (frame["split"] == "perturbed") & (frame["family"].isin(calibrated_families))
        ].copy()
        if perturbed.empty:
            return pd.DataFrame(
                columns=[
                    "model_name",
                    "family",
                    "subtype",
                    "severity",
                    "template_name",
                    "exposure_axis",
                    "static_strength_score",
                    "monotonic_template_pass",
                    "subtype_purity_pass",
                    "light_nonzero_pass",
                    "sample_count",
                    "overall_task_success_rate",
                    "paired_success_flip_rate",
                    "dominant_primary_failure_tag",
                    "dominant_observed_failure_tag",
                    "operator_names",
                    "posthoc_model_result",
                    "calibration_status",
                    "calibration_note",
                ]
            )

        grouped = self._aggregate(perturbed, ["model_name", "family", "subtype", "severity"])
        paired_metrics = self._aggregate_pair_metrics(paired.loc[paired["family"].isin(calibrated_families)], ["model_name", "family", "subtype", "severity"])
        grouped = grouped.merge(paired_metrics, on=["model_name", "family", "subtype", "severity"], how="left")
        grouped = self._finalize_paired_columns(grouped)

        diag_rows: list[dict[str, object]] = []
        for key, group in perturbed.groupby(["model_name", "family", "subtype", "severity"], dropna=False):
            failures = group.loc[group["status"] == "failure"].copy()
            diag_rows.append(
                {
                    "model_name": key[0],
                    "family": key[1],
                    "subtype": key[2],
                    "severity": key[3],
                    "dominant_primary_failure_tag": self._dominant_value(failures["primary_failure_tag"]),
                    "dominant_observed_failure_tag": self._dominant_value(failures["observed_failure_tag"]),
                    "operator_names": self._dominant_value(group["operator"]),
                }
            )
        diagnostics = pd.DataFrame(diag_rows)
        grouped = grouped.merge(diagnostics, on=["model_name", "family", "subtype", "severity"], how="left")
        grouped[["template_name", "exposure_axis", "static_strength_score", "monotonic_template_pass", "subtype_purity_pass", "light_nonzero_pass"]] = grouped.apply(
            lambda row: pd.Series(self._preflight_series(str(row["family"]), str(row["subtype"]), str(row["severity"]))),
            axis=1,
        )

        status_rows: list[dict[str, object]] = []
        for (model_name, family, subtype), group in grouped.groupby(["model_name", "family", "subtype"], dropna=False):
            severity_map = group.set_index("severity")["overall_task_success_rate"].to_dict()
            preflight_map = group.set_index("severity")["static_strength_score"].to_dict()
            gate_flags = {
                "monotonic_template_pass": bool(group["monotonic_template_pass"].all()),
                "subtype_purity_pass": bool(group["subtype_purity_pass"].all()),
                "light_nonzero_pass": bool(group.loc[group["severity"] == SeverityLevel.LIGHT.value, "light_nonzero_pass"].all()),
            }
            status, note = self._subtype_calibration_status(subtype, severity_map, preflight_map, gate_flags)
            for severity in group["severity"].tolist():
                status_rows.append(
                    {
                        "model_name": model_name,
                        "family": family,
                        "subtype": subtype,
                        "severity": severity,
                        "posthoc_model_result": self._posthoc_model_result_label(severity_map, severity),
                        "calibration_status": status,
                        "calibration_note": note,
                    }
                )
        statuses = pd.DataFrame(status_rows)
        grouped = grouped.merge(statuses, on=["model_name", "family", "subtype", "severity"], how="left")
        grouped = grouped[
            [
                "model_name",
                "family",
                "subtype",
                "severity",
                "template_name",
                "exposure_axis",
                "static_strength_score",
                "monotonic_template_pass",
                "subtype_purity_pass",
                "light_nonzero_pass",
                "sample_count",
                "overall_task_success_rate",
                "paired_success_flip_rate",
                "dominant_primary_failure_tag",
                "dominant_observed_failure_tag",
                "operator_names",
                "posthoc_model_result",
                "calibration_status",
                "calibration_note",
            ]
        ].copy()
        return self._order(grouped, "severity", ["model_name", "family", "subtype"])

    def _failure_taxonomy(self, frame: pd.DataFrame) -> pd.DataFrame:
        failures = frame.loc[frame["status"] == "failure"].copy()
        if failures.empty:
            return pd.DataFrame(
                columns=[
                    "model_name",
                    "severity",
                    "observed_failure_tag",
                    "primary_failure_tag",
                    "is_cascade_failure",
                    "secondary_reason",
                    "count",
                    "total",
                    "ratio",
                ]
            )
        counts = (
            failures.groupby(
                ["model_name", "severity", "observed_failure_tag", "primary_failure_tag", "is_cascade_failure", "secondary_reason"],
                dropna=False,
            )
            .size()
            .reset_index(name="count")
        )
        totals = frame.groupby(["model_name", "severity"], dropna=False).size().reset_index(name="total")
        merged = counts.merge(totals, on=["model_name", "severity"], how="left")
        merged["ratio"] = merged["count"] / merged["total"]
        return self._order(merged, "severity", ["model_name", "observed_failure_tag", "primary_failure_tag", "secondary_reason"])

    def _primary_failure_breakdown(self, frame: pd.DataFrame) -> pd.DataFrame:
        failures = frame.loc[frame["status"] == "failure"].copy()
        if failures.empty:
            return pd.DataFrame(
                columns=["model_name", "family", "severity", "first_failure_stage", "primary_failure_tag", "count", "total", "ratio"]
            )
        counts = (
            failures.groupby(["model_name", "family", "severity", "first_failure_stage", "primary_failure_tag"], dropna=False)
            .size()
            .reset_index(name="count")
        )
        totals = failures.groupby(["model_name", "family", "severity"], dropna=False).size().reset_index(name="total")
        merged = counts.merge(totals, on=["model_name", "family", "severity"], how="left")
        merged["ratio"] = merged["count"] / merged["total"]
        return self._order(merged, "severity", ["model_name", "family", "first_failure_stage", "primary_failure_tag"])

    def _cascade_failure_breakdown(self, frame: pd.DataFrame) -> pd.DataFrame:
        cascades = frame.loc[(frame["status"] == "failure") & (frame["is_cascade_failure"])].copy()
        if cascades.empty:
            return pd.DataFrame(
                columns=[
                    "model_name",
                    "family",
                    "severity",
                    "cascade_from_stage",
                    "primary_failure_tag",
                    "observed_failure_tag",
                    "count",
                    "total",
                    "ratio",
                ]
            )
        counts = (
            cascades.groupby(
                ["model_name", "family", "severity", "cascade_from_stage", "primary_failure_tag", "observed_failure_tag"],
                dropna=False,
            )
            .size()
            .reset_index(name="count")
        )
        totals = cascades.groupby(["model_name", "family", "severity"], dropna=False).size().reset_index(name="total")
        merged = counts.merge(totals, on=["model_name", "family", "severity"], how="left")
        merged["ratio"] = merged["count"] / merged["total"]
        return self._order(merged, "severity", ["model_name", "family", "cascade_from_stage", "observed_failure_tag"])

    def _failure_by_family(self, frame: pd.DataFrame) -> pd.DataFrame:
        failures = frame.loc[
            (frame["status"] == "failure")
            & (frame["family"] != BenchmarkFamily.CLEAN.value)
            & (frame["family"] != BenchmarkFamily.CROSS_FAMILY.value)
        ].copy()
        if failures.empty:
            return pd.DataFrame(columns=["model_name", "family", "severity", "observed_failure_tag", "count", "total", "ratio"])
        counts = failures.groupby(["model_name", "family", "severity", "observed_failure_tag"], dropna=False).size().reset_index(name="count")
        totals = (
            frame.loc[
                (frame["family"] != BenchmarkFamily.CLEAN.value)
                & (frame["family"] != BenchmarkFamily.CROSS_FAMILY.value)
            ]
            .groupby(["model_name", "family", "severity"], dropna=False)
            .size()
            .reset_index(name="total")
        )
        merged = counts.merge(totals, on=["model_name", "family", "severity"], how="left")
        merged["ratio"] = merged["count"] / merged["total"]
        return self._order(merged, "severity", ["model_name", "family", "observed_failure_tag"])

    def _failure_by_composition(self, frame: pd.DataFrame) -> pd.DataFrame:
        failures = frame.loc[
            (frame["status"] == "failure")
            & (frame["family"] == BenchmarkFamily.CROSS_FAMILY.value)
            & (frame["composition_id"].notna())
        ].copy()
        if failures.empty:
            return pd.DataFrame(columns=["model_name", "composition_id", "severity", "observed_failure_tag", "count", "total", "ratio"])
        counts = failures.groupby(["model_name", "composition_id", "severity", "observed_failure_tag"], dropna=False).size().reset_index(name="count")
        totals = (
            frame.loc[
                (frame["family"] == BenchmarkFamily.CROSS_FAMILY.value)
                & (frame["composition_id"].notna())
            ]
            .groupby(["model_name", "composition_id", "severity"], dropna=False)
            .size()
            .reset_index(name="total")
        )
        merged = counts.merge(totals, on=["model_name", "composition_id", "severity"], how="left")
        merged["ratio"] = merged["count"] / merged["total"]
        return self._order(merged, "severity", ["model_name", "composition_id", "observed_failure_tag"])

    def _paired_case_dump(self, paired: pd.DataFrame) -> pd.DataFrame:
        if paired.empty:
            return pd.DataFrame(
                columns=[
                    "model_name",
                    "family",
                    "subtype",
                    "severity",
                    "stage_target",
                    "source_sample_id",
                    "sample_id",
                    "clean_sample_id",
                    "operator_names",
                    "clean_success",
                    "perturbed_success",
                    "paired_success_flip",
                    "observed_failure_tag",
                    "primary_failure_tag",
                    "first_failure_stage",
                    "is_cascade_failure",
                    "cascade_from_stage",
                    "expected_primary_match",
                    "allowed_spillover_match",
                    "boundary_violation_flag",
                    "boundary_violation_reason",
                    "gold_canonical_tool_id",
                    "gold_exposed_tool_name",
                    "predicted_tool_raw",
                    "predicted_tool_resolved",
                    "predicted_arguments",
                    "expected_arguments",
                    "expected_result",
                ]
            )
        export = paired[
            [
                "model_name",
                "family",
                "subtype",
                "severity",
                "stage_target",
                "source_sample_id",
                "sample_id",
                "clean_sample_id",
                "operator_names",
                "clean_overall_task_success",
                "overall_task_success",
                "paired_success_flip",
                "observed_failure_tag",
                "primary_failure_tag",
                "first_failure_stage",
                "is_cascade_failure",
                "cascade_from_stage",
                "expected_primary_match",
                "allowed_spillover_match",
                "boundary_violation_flag",
                "boundary_violation_reason",
                "gold_canonical_tool_id",
                "gold_exposed_tool_name",
                "predicted_tool_raw",
                "predicted_tool_resolved",
                "predicted_arguments",
                "expected_arguments",
                "expected_result",
            ]
        ].copy()
        export = export.rename(
            columns={
                "clean_overall_task_success": "clean_success",
                "overall_task_success": "perturbed_success",
            }
        )
        return self._order(export, "severity", ["model_name", "family", "subtype", "source_sample_id"])

    def _family_boundary_report(self, frame: pd.DataFrame) -> pd.DataFrame:
        perturbed = frame.loc[
            (frame["split"] == "perturbed")
            & (frame["family"] != BenchmarkFamily.CROSS_FAMILY.value)
        ].copy()
        if perturbed.empty:
            return pd.DataFrame(
                columns=[
                    "model_name",
                    "family",
                    "severity",
                    "sample_count",
                    "total_failures",
                    "primary_no_call_failures",
                    "primary_wrong_tool_failures",
                    "primary_missing_parameter_failures",
                    "primary_wrong_parameter_failures",
                    "primary_invalid_format_failures",
                    "primary_execution_fail_failures",
                    "cascade_wrong_parameter_failures",
                    "cascade_missing_parameter_failures",
                    "expected_primary_match_rate",
                    "allowed_spillover_match_rate",
                    "boundary_violation_rate",
                    "dominant_primary_failure_tag",
                    "dominant_observed_failure_tag",
                    "family_boundary_pass",
                    "notes",
                ]
            )

        rows: list[dict[str, object]] = []
        for (model_name, family, severity), group in perturbed.groupby(["model_name", "family", "severity"], dropna=False):
            failures = group.loc[group["status"] == "failure"].copy()
            boundary_notes = failures.loc[failures["boundary_violation_flag"], "boundary_violation_reason"].dropna()
            boundary_violation_rate = self._failure_boolean_rate(failures, "boundary_violation_flag")
            rows.append(
                {
                    "model_name": model_name,
                    "family": family,
                    "severity": severity,
                    "sample_count": int(len(group)),
                    "total_failures": int(len(failures)),
                    "primary_no_call_failures": int((failures["primary_failure_tag"] == FailureTag.NO_CALL.value).sum()),
                    "primary_wrong_tool_failures": int((failures["primary_failure_tag"] == FailureTag.WRONG_TOOL.value).sum()),
                    "primary_missing_parameter_failures": int((failures["primary_failure_tag"] == FailureTag.MISSING_PARAMETER.value).sum()),
                    "primary_wrong_parameter_failures": int((failures["primary_failure_tag"] == FailureTag.WRONG_PARAMETER.value).sum()),
                    "primary_invalid_format_failures": int((failures["primary_failure_tag"] == FailureTag.INVALID_FORMAT.value).sum()),
                    "primary_execution_fail_failures": int((failures["primary_failure_tag"] == FailureTag.EXECUTION_FAIL.value).sum()),
                    "cascade_wrong_parameter_failures": int(((failures["is_cascade_failure"]) & (failures["observed_failure_tag"] == FailureTag.WRONG_PARAMETER.value)).sum()),
                    "cascade_missing_parameter_failures": int(((failures["is_cascade_failure"]) & (failures["observed_failure_tag"] == FailureTag.MISSING_PARAMETER.value)).sum()),
                    "expected_primary_match_rate": self._failure_boolean_rate(failures, "expected_primary_match"),
                    "allowed_spillover_match_rate": self._failure_boolean_rate(failures, "allowed_spillover_match"),
                    "boundary_violation_rate": boundary_violation_rate,
                    "dominant_primary_failure_tag": self._dominant_value(failures["primary_failure_tag"]),
                    "dominant_observed_failure_tag": self._dominant_value(failures["observed_failure_tag"]),
                    "family_boundary_pass": boundary_violation_rate == 0.0,
                    "notes": boundary_notes.value_counts().idxmax() if not boundary_notes.empty else "all failures stayed within expected primary/spillover envelope",
                }
            )
        report = pd.DataFrame(rows)
        return self._order(report, "severity", ["model_name", "family"])

    def _build_paired_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        perturbed = frame.loc[frame["split"] == "perturbed"].copy()
        clean = frame.loc[frame["split"] == "clean"].copy()
        if perturbed.empty or clean.empty:
            return pd.DataFrame()
        clean_view = clean[
            [
                "model_name",
                "sample_id",
                "overall_task_success",
                "status",
                "observed_failure_tag",
                "primary_failure_tag",
                "first_failure_stage",
                "is_cascade_failure",
                "cascade_from_stage",
                "secondary_reason",
            ]
        ].rename(
            columns={
                "sample_id": "source_sample_id",
                "overall_task_success": "clean_overall_task_success",
                "status": "clean_status",
                "observed_failure_tag": "clean_observed_failure_tag",
                "primary_failure_tag": "clean_primary_failure_tag",
                "first_failure_stage": "clean_first_failure_stage",
                "is_cascade_failure": "clean_is_cascade_failure",
                "cascade_from_stage": "clean_cascade_from_stage",
                "secondary_reason": "clean_secondary_reason",
            }
        )
        merged = perturbed.merge(clean_view, on=["model_name", "source_sample_id"], how="left", validate="many_to_one")
        merged["clean_sample_id"] = merged["source_sample_id"]
        merged["paired_success_flip"] = (
            (merged["clean_overall_task_success"].fillna(0).astype(int) == 1)
            & (merged["overall_task_success"].fillna(0).astype(int) == 0)
        ).astype(int)
        return merged

    @staticmethod
    def _aggregate(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=group_cols + METRIC_COLUMNS)
        return (
            frame.groupby(group_cols, dropna=False)
            .agg(
                sample_count=("sample_id", "count"),
                call_rate=("call_present", "mean"),
                interface_name_accuracy=("interface_name_match", "mean"),
                canonical_tool_accuracy=("canonical_tool_match", "mean"),
                format_valid_rate=("format_valid", "mean"),
                missing_parameter_rate=("missing_parameter_flag", "mean"),
                wrong_parameter_rate=("wrong_parameter_flag", "mean"),
                execution_success_rate=("execution_success", "mean"),
                overall_task_success_rate=("overall_task_success", "mean"),
            )
            .reset_index()
        )

    @staticmethod
    def _aggregate_pair_metrics(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=group_cols + PAIR_COLUMNS)
        return (
            frame.groupby(group_cols, dropna=False)
            .agg(
                paired_clean_sample_count=("source_sample_id", "count"),
                paired_clean_success_rate=("clean_overall_task_success", "mean"),
                paired_success_flip_rate=("paired_success_flip", "mean"),
            )
            .reset_index()
        )

    @staticmethod
    def _finalize_paired_columns(frame: pd.DataFrame) -> pd.DataFrame:
        for column in PAIR_COLUMNS:
            if column not in frame.columns:
                frame[column] = 0.0
        frame["paired_clean_sample_count"] = frame["paired_clean_sample_count"].fillna(0).astype(int)
        frame["paired_clean_success_rate"] = frame["paired_clean_success_rate"].fillna(0.0)
        frame["paired_success_flip_rate"] = frame["paired_success_flip_rate"].fillna(0.0)
        return frame

    def _fill_family_buckets(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.keep_missing_buckets:
            return self._order(frame, "severity", ["model_name", "family"])
        rows = [
            {"model_name": model, "family": family, "severity": severity}
            for model in self.manifest.models
            for family in self.manifest.family_order
            for severity in self.manifest.severity_order
        ]
        return self._merge_grid(rows, frame, ["model_name", "family", "severity"], ["model_name", "family"])

    def _fill_subtype_buckets(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.keep_missing_buckets:
            return self._order(frame, "severity", ["model_name", "family", "subtype"])
        subtype_to_family = dict(self.manifest.subtype_family_mapping)
        rows = []
        for model in self.manifest.models:
            for subtype in self.manifest.subtype_order:
                family = subtype_to_family[subtype]
                rows.append(
                    {
                        "model_name": model,
                        "family": family,
                        "subtype": subtype,
                        "stage_target": self.manifest.stage_mapping[subtype],
                        "severity": None,
                    }
                )
        expanded_rows = [{**row, "severity": severity} for row in rows for severity in self.manifest.severity_order]
        return self._merge_grid(
            expanded_rows,
            frame,
            ["model_name", "family", "subtype", "stage_target", "severity"],
            ["model_name", "family", "subtype"],
        )

    def _fill_stage_buckets(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.keep_missing_buckets:
            return self._order(frame, "severity", ["model_name", "stage_target"])
        rows = [
            {"model_name": model, "stage_target": stage, "severity": severity}
            for model in self.manifest.models
            for stage in self.manifest.stage_order
            for severity in self.manifest.severity_order
        ]
        return self._merge_grid(rows, frame, ["model_name", "stage_target", "severity"], ["model_name", "stage_target"])

    def _merge_grid(self, rows: list[dict[str, object]], frame: pd.DataFrame, merge_cols: list[str], sort_cols: list[str]) -> pd.DataFrame:
        grid = pd.DataFrame(rows)
        merged = grid.merge(frame, on=merge_cols, how="left")
        for column in METRIC_COLUMNS:
            merged[column] = merged[column].fillna(0)
        for column in PAIR_COLUMNS:
            if column not in merged.columns:
                merged[column] = 0
        merged["paired_clean_sample_count"] = merged["paired_clean_sample_count"].fillna(0).astype(int)
        merged["paired_clean_success_rate"] = merged["paired_clean_success_rate"].fillna(0.0)
        merged["paired_success_flip_rate"] = merged["paired_success_flip_rate"].fillna(0.0)
        return self._order(merged, "severity", sort_cols)

    @staticmethod
    def _order(frame: pd.DataFrame, severity_col: str, sort_cols: list[str]) -> pd.DataFrame:
        ordered = frame.copy()
        ordered[severity_col] = pd.Categorical(ordered[severity_col], categories=SEVERITY_ORDER, ordered=True)
        return ordered.sort_values(sort_cols + [severity_col]).reset_index(drop=True)

    @staticmethod
    def _failure_boolean_rate(failures: pd.DataFrame, column: str) -> float:
        if failures.empty:
            return 0.0
        return float(failures[column].fillna(False).astype(bool).mean())

    @staticmethod
    def _dominant_value(series: pd.Series) -> str | None:
        values = series.dropna()
        if values.empty:
            return None
        return str(values.value_counts().idxmax())

    @staticmethod
    def _distribution_json(series: pd.Series) -> str:
        values = series.dropna().astype(str)
        if values.empty:
            return "{}"
        distribution = values.value_counts(normalize=True).sort_index().round(4).to_dict()
        return json.dumps(distribution, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _coerce_jsonish_list(value: object) -> list[str]:
        if isinstance(value, list):
            return [str(item) for item in value]
        if isinstance(value, tuple):
            return [str(item) for item in value]
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return []
        text = str(value)
        try:
            parsed = json.loads(text.replace("'", '"'))
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except json.JSONDecodeError:
            pass
        return [part.strip() for part in text.split("|") if part.strip()]

    @classmethod
    def _first_jsonish(cls, series: pd.Series) -> str:
        if series.empty:
            return "[]"
        values = cls._coerce_jsonish_list(series.iloc[0])
        return json.dumps(values, ensure_ascii=False)

    def _severity_calibration_lookup(self, frame: pd.DataFrame, paired: pd.DataFrame) -> dict[tuple[str, str, str, str], dict[str, object]]:
        report = self._severity_calibration_report(frame, paired)
        lookup: dict[tuple[str, str, str, str], dict[str, object]] = {}
        if report.empty:
            return lookup
        for record in report.to_dict(orient="records"):
            lookup[(record["model_name"], record["family"], record["subtype"], record["severity"])] = record
        return lookup

    def _preflight_lookup(self) -> dict[tuple[str, str, str], dict[str, object]]:
        return {
            (str(row["family"]), str(row["subtype"]), str(row["severity"])): row
            for row in self.manifest.preflight_gate_rows
        }

    def _preflight_series(self, family: str, subtype: str, severity: str) -> tuple[object, object, object, object, object, object]:
        row = self._preflight_lookup().get((family, subtype, severity), {})
        return (
            row.get("template_name", ""),
            row.get("exposure_axis", ""),
            row.get("static_strength_score", 0.0),
            row.get("monotonic_pass", False),
            row.get("subtype_purity_pass", False),
            row.get("light_nonzero_pass", False),
        )

    def _operator_metadata_series(self, operator_name: str) -> tuple[object, object, object, object, object]:
        metadata = self.manifest.operator_metadata.get(operator_name, {})
        return (
            metadata.get("intrinsic_strength_score", 0.0),
            ",".join(metadata.get("severity_fit", [])),
            ",".join(metadata.get("subtype_affinity", [])),
            metadata.get("exposure_axis", ""),
            metadata.get("stack_role", ""),
        )

    @staticmethod
    def _subtype_calibration_status(
        subtype: str,
        severity_map: dict[str, float],
        preflight_map: dict[str, float],
        gate_flags: dict[str, bool],
    ) -> tuple[str, str]:
        light = float(severity_map.get(SeverityLevel.LIGHT.value, 0.0))
        medium = float(severity_map.get(SeverityLevel.MEDIUM.value, 0.0))
        heavy = float(severity_map.get(SeverityLevel.HEAVY.value, 0.0))
        gap_threshold = 0.03
        preflight_light = float(preflight_map.get(SeverityLevel.LIGHT.value, 0.0))
        preflight_medium = float(preflight_map.get(SeverityLevel.MEDIUM.value, 0.0))
        preflight_heavy = float(preflight_map.get(SeverityLevel.HEAVY.value, 0.0))

        if not gate_flags.get("subtype_purity_pass", False):
            return "rewrite", f"subtype purity failed in {subtype}"
        if not gate_flags.get("light_nonzero_pass", False):
            return "rewrite", f"light template remains near-zero in {subtype}"
        if medium == 0.0 and heavy == 0.0 and light > medium and gate_flags.get("monotonic_template_pass", False) and preflight_light <= preflight_medium <= preflight_heavy:
            return "saturated_on_model", f"medium and heavy are saturated on this model in {subtype}; verify with a stronger reference model"
        if medium >= light:
            return "rewrite", f"medium does not beat light in {subtype}"
        if heavy > medium:
            return "rewrite", f"heavy weaker than medium in {subtype}"
        if not gate_flags.get("monotonic_template_pass", False) or not preflight_light <= preflight_medium <= preflight_heavy:
            return "inspect", f"static calibration is noisy in {subtype}"
        if light - medium < gap_threshold:
            return "inspect", f"medium distinction is still small in {subtype}"
        if medium - heavy < gap_threshold:
            return "inspect", f"heavy too close to medium in {subtype}"
        return "keep", "posthoc ordering is healthy"

    @staticmethod
    def _posthoc_model_result_label(severity_map: dict[str, float], severity: str) -> str:
        score = float(severity_map.get(severity, 0.0))
        return f"success_rate={score:.6f}"

    def _severity_fit_match(self, row: pd.Series) -> bool:
        metadata = self.manifest.operator_metadata.get(str(row.get("operator_name")), {})
        severities = metadata.get("severity_fit", [])
        preferred_subtypes = metadata.get("subtype_affinity", [])
        severity_match = not severities or str(row.get("severity")) in severities
        subtype_match = not preferred_subtypes or str(row.get("subtype")) in preferred_subtypes
        return bool(severity_match and subtype_match)

    def _operator_calibration_issue(
        self,
        row: pd.Series,
        calibration_lookup: dict[tuple[str, str, str, str], dict[str, object]],
    ) -> tuple[bool, str]:
        if not self._severity_fit_match(row):
            return True, "subtype mismatch"

        record = calibration_lookup.get((row["model_name"], row["family"], row["subtype"], row["severity"]))
        if not record:
            return False, ""
        note = str(record.get("calibration_note") or "")
        status = str(record.get("calibration_status") or "")
        if status == "rewrite" and "heavy weaker than medium" in note and row["severity"] in {SeverityLevel.MEDIUM.value, SeverityLevel.HEAVY.value}:
            return True, "medium stronger than heavy"
        if status == "rewrite" and "medium does not beat light" in note and row["severity"] in {SeverityLevel.LIGHT.value, SeverityLevel.MEDIUM.value}:
            return True, "medium ineffective"
        if status == "inspect" and "heavy too close to medium" in note and row["severity"] in {SeverityLevel.MEDIUM.value, SeverityLevel.HEAVY.value}:
            return True, "heavy too close to medium"
        if status == "rewrite" and "subtype purity failed" in note:
            return True, "subtype mismatch"
        return False, ""

    @staticmethod
    def _recommended_operator_status(row: pd.Series) -> str:
        valid_drop = float(row.get("drop_vs_paired_clean", 0.0)) >= 0.05 or float(row.get("paired_success_flip_rate", 0.0)) >= 0.05
        boundary_violation_rate = float(row.get("boundary_violation_rate", 0.0))
        expected_primary_match_rate = float(row.get("expected_primary_match_rate", 0.0))
        allowed_spillover_match_rate = float(row.get("allowed_spillover_match_rate", 0.0))
        total_failures = int(row.get("total_failures", 0))

        if boundary_violation_rate >= 0.10:
            return "disable"
        if total_failures > 0 and expected_primary_match_rate < 0.60:
            return "rewrite"
        if valid_drop and allowed_spillover_match_rate >= 0.25:
            return "inspect"
        return "keep"


def write_tables(output_dir: Path, tables: dict[str, pd.DataFrame]) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for key in [
        "summary_overall",
        "summary_by_family",
        "summary_by_subtype",
        "summary_by_stage",
        "summary_by_operator",
        "summary_by_composition",
        "interaction_effect",
        "severity_calibration_report",
        "failure_taxonomy",
        "primary_failure_breakdown",
        "cascade_failure_breakdown",
        "failure_by_family",
        "failure_by_composition",
        "paired_case_dump",
        "family_boundary_report",
    ]:
        path = output_dir / f"{key}.csv"
        tables[key].to_csv(path, index=False)
        written[key] = path
    return written
