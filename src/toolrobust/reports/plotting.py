"""Unified plotting utilities for ToolRobust v3."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
import textwrap
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle

from toolrobust.domain.enums import SEVERITY_ORDER
from toolrobust.domain.schemas import RunManifest
from toolrobust.reports.profiles import get_output_profile_spec


@dataclass(slots=True)
class PlotContext:
    manifest: RunManifest
    model_name: str


class PlotManager:
    def __init__(self, results_dir: Path, output_dir: Path | None = None) -> None:
        self.results_dir = results_dir
        self.output_dir = output_dir or (results_dir / "plots")

    def generate_all(self, *, profile, overwrite: bool = True, progress: object | None = None, include_loading_stage: bool = True) -> dict[str, Path]:
        manifest = self.resolve_run_manifest(self.results_dir)
        tables = self.load_result_tables(self.results_dir)
        if overwrite and self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        generated: dict[str, Path] = {}
        plot_names = get_output_profile_spec(profile).plots
        if progress is not None:
            progress.start("generating_plots", total_items=len(plot_names) * len(manifest.models))
        for model_name in manifest.models:
            context = PlotContext(manifest=manifest, model_name=model_name)
            for plot_name in plot_names:
                result = getattr(self, f"plot_{plot_name}")(tables=tables, context=context)
                if isinstance(result, dict):
                    generated.update({f"{model_name}:{key}": path for key, path in result.items()})
                else:
                    generated[f"{model_name}:{plot_name}"] = result
                if progress is not None:
                    progress.advance(current_item=f"{model_name}_{plot_name}")
        if progress is not None:
            progress.finish_stage("[progress] completed stage=generating_plots")
        return generated

    @staticmethod
    def resolve_run_manifest(results_dir: Path) -> RunManifest:
        payload = json.loads((results_dir / "run_manifest.json").read_text(encoding="utf-8"))
        return RunManifest.model_validate(payload)

    @staticmethod
    def load_result_tables(results_dir: Path) -> dict[str, pd.DataFrame]:
        names = [
            "summary_overall",
            "summary_by_family",
            "summary_by_subtype",
            "summary_by_stage",
            "summary_by_operator",
            "summary_by_composition",
            "interaction_effect",
            "failure_taxonomy",
            "primary_failure_breakdown",
            "cascade_failure_breakdown",
            "failure_by_family",
            "failure_by_composition",
            "paired_case_dump",
            "family_boundary_report",
            "records",
        ]
        return {name: pd.read_csv(results_dir / f"{name}.csv") for name in names}

    def plot_overall_robustness_curve(self, *, tables: dict[str, pd.DataFrame], context: PlotContext) -> Path:
        frame = tables["summary_overall"].loc[tables["summary_overall"]["model_name"] == context.model_name].copy()
        frame["condition"] = pd.Categorical(frame["condition"], categories=context.manifest.severity_order, ordered=True)
        frame = frame.sort_values("condition")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(frame["condition"], frame["overall_task_success_rate"], marker="o", color="#1d4ed8")
        ax.set_ylim(0, 1.05)
        ax.set_title(self._title("Overall Robustness Curve", context))
        ax.set_xlabel("severity")
        ax.set_ylabel("overall_task_success_rate")
        return self._save(fig, "overall_robustness_curve", context)

    def plot_family_wise_robustness_bar(self, *, tables: dict[str, pd.DataFrame], context: PlotContext) -> Path:
        frame = tables["summary_by_family"].loc[tables["summary_by_family"]["model_name"] == context.model_name].copy()
        pivot = (
            frame.pivot(index="family", columns="severity", values="overall_task_success_rate")
            .reindex(index=context.manifest.family_order, fill_value=0)
            .reindex(columns=context.manifest.severity_order, fill_value=0)
            .fillna(0)
        )
        fig, ax = plt.subplots(figsize=(11, 6))
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(f"Family-wise Robustness (enabled families: {', '.join(context.manifest.enabled_families)})")
        ax.set_xlabel("family")
        ax.set_ylabel("overall_task_success_rate")
        ax.tick_params(axis="x", rotation=0)
        ax.legend(title="severity", bbox_to_anchor=(1.02, 1), loc="upper left")
        return self._save(fig, "family_wise_robustness_bar", context)

    def plot_subtype_heatmap(self, *, tables: dict[str, pd.DataFrame], context: PlotContext) -> Path:
        frame = tables["summary_by_subtype"].loc[tables["summary_by_subtype"]["model_name"] == context.model_name].copy()
        rate_grid = (
            frame.pivot(index="subtype", columns="severity", values="overall_task_success_rate")
            .reindex(index=context.manifest.subtype_order, columns=context.manifest.severity_order)
        )
        sample_grid = (
            frame.pivot(index="subtype", columns="severity", values="sample_count")
            .reindex(index=context.manifest.subtype_order, columns=context.manifest.severity_order)
            .fillna(0)
        )
        pivot = rate_grid.where(sample_grid > 0)
        fig, ax = plt.subplots(figsize=(11, max(5, len(pivot.index) * 0.45)))
        cmap = plt.get_cmap("YlGnBu").copy()
        cmap.set_bad(color="#f3f4f6")
        image = ax.imshow(np.ma.masked_invalid(pivot.to_numpy(dtype=float)), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_title(self._title("Subtype Heatmap", context))
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=0, ha="center")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        for row in range(pivot.shape[0]):
            for col in range(pivot.shape[1]):
                if sample_grid.iloc[row, col] <= 0:
                    ax.text(col, row, "N/A", ha="center", va="center", fontsize=8, color="#6b7280")
                else:
                    ax.text(col, row, f"{pivot.iloc[row, col]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax)
        return self._save(fig, "subtype_heatmap", context)

    def plot_failure_taxonomy_stacked_bar(self, *, tables: dict[str, pd.DataFrame], context: PlotContext) -> Path:
        frame = tables["failure_taxonomy"].loc[tables["failure_taxonomy"]["model_name"] == context.model_name].copy()
        fig, ax = plt.subplots(figsize=(10, 6))
        if frame.empty:
            ax.set_title(f"Failure Taxonomy by Severity (scope: {', '.join(context.manifest.enabled_families)})")
            ax.text(0.5, 0.5, "No failures recorded", ha="center", va="center")
            ax.axis("off")
            return self._save(fig, "failure_taxonomy_stacked_bar", context)
        pivot = (
            frame.groupby(["severity", "observed_failure_tag"], dropna=False)["count"]
            .sum()
            .unstack(fill_value=0)
            .reindex(index=context.manifest.severity_order, fill_value=0)
        )
        pivot.plot(kind="bar", stacked=True, ax=ax)
        ax.set_title(f"Failure Taxonomy by Severity (scope: {', '.join(context.manifest.enabled_families)})")
        ax.set_xlabel("severity")
        ax.set_ylabel("count")
        ax.tick_params(axis="x", rotation=0)
        ax.legend(title="observed_failure_tag", bbox_to_anchor=(1.02, 1), loc="upper left")
        return self._save(fig, "failure_taxonomy_stacked_bar", context)

    def plot_stage_breakdown(self, *, tables: dict[str, pd.DataFrame], context: PlotContext) -> Path:
        frame = tables["summary_by_stage"].loc[tables["summary_by_stage"]["model_name"] == context.model_name].copy()
        rate_grid, sample_grid = self._build_stage_breakdown_grid(frame=frame, context=context)
        stages = list(rate_grid.index)
        severities = list(rate_grid.columns)
        x_positions = np.arange(len(stages), dtype=float)
        width = 0.18 if severities else 0.18
        offsets = (np.arange(len(severities), dtype=float) - (len(severities) - 1) / 2.0) * width
        colors = {
            "clean": "#1d4ed8",
            "light": "#60a5fa",
            "medium": "#f59e0b",
            "heavy": "#dc2626",
        }

        fig, ax = plt.subplots(figsize=(13, 7))
        for idx, severity in enumerate(severities):
            xpos = x_positions + offsets[idx]
            rates = rate_grid[severity].to_numpy(dtype=float)
            sample_counts = sample_grid[severity].to_numpy(dtype=float)
            has_samples = sample_counts > 0
            color = colors.get(severity, plt.get_cmap("tab10")(idx))

            if has_samples.any():
                ax.bar(
                    xpos[has_samples],
                    rates[has_samples],
                    width=width,
                    color=color,
                    label=severity,
                    zorder=3,
                )

            for bar_x, count in zip(xpos[~has_samples], sample_counts[~has_samples], strict=False):
                box = Rectangle(
                    (bar_x - width / 2, 0.02),
                    width,
                    0.08,
                    facecolor="#f3f4f6",
                    edgecolor="#d1d5db",
                    linewidth=1.0,
                    zorder=2,
                )
                ax.add_patch(box)
                ax.text(
                    bar_x,
                    0.06,
                    f"N/A\n(n={int(count)})",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#6b7280",
                    zorder=4,
                )

        legend_handles = [
            Patch(facecolor=colors.get(severity, plt.get_cmap("tab10")(idx)), edgecolor="none", label=severity)
            for idx, severity in enumerate(severities)
        ]
        legend_handles.append(Patch(facecolor="#f3f4f6", edgecolor="#d1d5db", label="N/A (n=0)"))
        fig.suptitle(self._title("Stage Breakdown", context), y=0.97)
        fig.text(
            0.5,
            0.935,
            "N/A indicates no samples for that stage/severity bucket.",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#4b5563",
        )
        ax.set_xlabel("stage")
        ax.set_ylabel("overall_task_success_rate")
        ax.set_ylim(0, 1.08)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(stages, rotation=20, ha="right")
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35, zorder=0)
        ax.legend(handles=legend_handles, title="severity", bbox_to_anchor=(1.02, 1), loc="upper left")
        setattr(fig, "_toolrobust_tight_layout_rect", (0.0, 0.12, 0.82, 0.88))
        return self._save(fig, "stage_breakdown", context)

    def plot_cross_pair_success_heatmap(self, *, tables: dict[str, pd.DataFrame], context: PlotContext) -> Path:
        frame = self._cross_summary_frame(tables, context)
        fig, ax = plt.subplots(figsize=(8, 5.2))
        if frame.empty:
            return self._save_empty(fig, ax, "Cross Pair Success Heatmap", "No cross-family composition rows found.", "cross_pair_success_heatmap", context)

        pair_order = self._cross_pair_order(frame)
        severity_order = [severity for severity in context.manifest.severity_order if severity in set(frame["severity"])]
        if not severity_order:
            severity_order = ["medium", "heavy"]
        aggregated = (
            frame.groupby(["pair", "severity"], dropna=False)
            .apply(lambda group: self._weighted_average(group, "overall_task_success_rate"))
            .reset_index(name="overall_task_success_rate")
        )
        matrix = (
            aggregated.pivot(index="pair", columns="severity", values="overall_task_success_rate")
            .reindex(index=pair_order, columns=severity_order)
        )
        cmap = plt.get_cmap("YlGnBu").copy()
        cmap.set_bad(color="#f3f4f6")
        image = ax.imshow(np.ma.masked_invalid(matrix.to_numpy(dtype=float)), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_title(self._title("Cross Pair Success Heatmap", context))
        ax.set_xticks(range(len(matrix.columns)))
        ax.set_xticklabels(matrix.columns)
        ax.set_yticks(range(len(matrix.index)))
        ax.set_yticklabels([self._pair_label(pair) for pair in matrix.index])
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix.iloc[row, col]
                if pd.isna(value):
                    ax.text(col, row, "N/A", ha="center", va="center", fontsize=8, color="#6b7280")
                else:
                    ax.text(col, row, f"{value:.0%}", ha="center", va="center", fontsize=10, color="#111827")
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("success rate")
        return self._save(fig, "cross_pair_success_heatmap", context)

    def plot_cross_all_compositions_success_matrix(self, *, tables: dict[str, pd.DataFrame], context: PlotContext) -> Path:
        frame = self._cross_summary_frame(tables, context)
        fig, ax = plt.subplots(figsize=(9, 12))
        if frame.empty:
            return self._save_empty(fig, ax, "Cross All Compositions Success Matrix", "No cross-family composition rows found.", "cross_all_compositions_success_matrix", context)

        severity_order = [severity for severity in context.manifest.severity_order if severity in set(frame["severity"])]
        composition_order = (
            frame.groupby("composition_id", dropna=False)["overall_task_success_rate"]
            .mean()
            .sort_values()
            .index
            .tolist()
        )
        matrix = (
            frame.pivot(index="composition_id", columns="severity", values="overall_task_success_rate")
            .reindex(index=composition_order, columns=severity_order)
        )
        fig.set_size_inches(9, max(8, 0.18 * len(matrix.index) + 2.5))
        cmap = plt.get_cmap("YlGnBu").copy()
        cmap.set_bad(color="#f3f4f6")
        image = ax.imshow(np.ma.masked_invalid(matrix.to_numpy(dtype=float)), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_title(self._title("Cross Composition Success Matrix", context))
        ax.set_xticks(range(len(matrix.columns)))
        ax.set_xticklabels(matrix.columns)
        ax.set_yticks(range(len(matrix.index)))
        ax.set_yticklabels([self._wrap_label(value, width=52) for value in matrix.index], fontsize=6)
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix.iloc[row, col]
                if not pd.isna(value):
                    ax.text(col, row, f"{value:.0%}", ha="center", va="center", fontsize=6, color="#111827")
        cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label("success rate")
        return self._save(fig, "cross_all_compositions_success_matrix", context)

    def plot_cross_pair_subtype_success_heatmaps(self, *, tables: dict[str, pd.DataFrame], context: PlotContext) -> dict[str, Path]:
        frame = self._cross_summary_frame(tables, context)
        if frame.empty:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            return {
                "cross_pair_subtype_success_heatmaps": self._save_empty(
                    fig,
                    ax,
                    "Cross Pair Subtype Success Heatmaps",
                    "No cross-family composition rows found.",
                    "cross_pair_subtype_success_heatmaps",
                    context,
                )
            }

        outputs: dict[str, Path] = {}
        severity_order = [severity for severity in context.manifest.severity_order if severity in set(frame["severity"])]
        for pair in self._cross_pair_order(frame):
            subset = frame.loc[frame["pair"] == pair].copy()
            if subset.empty:
                continue
            left_order = sorted(subset["left_subtype"].dropna().unique())
            right_order = sorted(subset["right_subtype"].dropna().unique())
            fig_width = max(8, 1.45 * len(right_order) + 4.0)
            fig_height = max(4.6, 0.88 * len(left_order) + 1.8)
            fig, axes = plt.subplots(1, len(severity_order), figsize=(fig_width, fig_height), sharey=True)
            axes_array = np.atleast_1d(axes)
            cmap = plt.get_cmap("YlGnBu").copy()
            cmap.set_bad(color="#f3f4f6")
            image = None
            for ax, severity in zip(axes_array, severity_order, strict=False):
                grid = (
                    subset.loc[subset["severity"] == severity]
                    .pivot_table(index="left_subtype", columns="right_subtype", values="overall_task_success_rate", aggfunc="mean")
                    .reindex(index=left_order, columns=right_order)
                )
                image = ax.imshow(np.ma.masked_invalid(grid.to_numpy(dtype=float)), vmin=0, vmax=1, cmap=cmap, aspect="auto")
                ax.set_title(f"{severity} success")
                ax.set_xticks(range(len(right_order)))
                ax.set_xticklabels([self._wrap_label(label.replace("_", " "), width=16) for label in right_order], rotation=45, ha="right", fontsize=8)
                ax.set_yticks(range(len(left_order)))
                ax.set_yticklabels([self._wrap_label(label.replace("_", " "), width=20) for label in left_order], fontsize=8)
                for row in range(grid.shape[0]):
                    for col in range(grid.shape[1]):
                        value = grid.iloc[row, col]
                        if pd.isna(value):
                            ax.text(col, row, "N/A", ha="center", va="center", fontsize=7, color="#6b7280")
                        else:
                            ax.text(col, row, f"{value:.0%}", ha="center", va="center", fontsize=8, color="#111827")
            fig.suptitle(self._title(f"{self._pair_label(pair)} Subtype Interaction", context), y=0.98)
            if image is not None:
                cbar = fig.colorbar(image, ax=axes_array.ravel().tolist(), fraction=0.025, pad=0.035, location="right")
                cbar.set_label("success rate")
            setattr(fig, "_toolrobust_tight_layout_rect", (0.0, 0.0, 0.93, 0.93))
            outputs[f"cross_{pair.replace('+', '')}_subtype_success_heatmaps"] = self._save(fig, f"cross_{pair.replace('+', '')}_subtype_success_heatmaps", context)
        return outputs

    def _title(self, label: str, context: PlotContext) -> str:
        return f"{label} (scope: {' + '.join(context.manifest.enabled_families)}, {context.manifest.dataset['task_scope']}, {context.model_name})"

    @staticmethod
    def _wrap_label(value: Any, *, width: int) -> str:
        return "\n".join(textwrap.wrap(str(value), width=width, break_long_words=False))

    @staticmethod
    def _pair_label(pair: str) -> str:
        return {
            "a+b": "A+B interface + intent",
            "a+c": "A+C interface + observation",
            "a+d": "A+D interface + execution",
            "b+c": "B+C intent + observation",
            "b+d": "B+D intent + execution",
        }.get(pair, pair)

    @staticmethod
    def _cross_pair_order(frame: pd.DataFrame) -> list[str]:
        preferred = ["a+b", "a+c", "a+d", "b+c", "b+d"]
        present = set(frame["pair"].dropna())
        return [pair for pair in preferred if pair in present] + sorted(present.difference(preferred))

    @staticmethod
    def _weighted_average(frame: pd.DataFrame, column: str) -> float:
        weights = pd.to_numeric(frame["sample_count"], errors="coerce").fillna(0).to_numpy(dtype=float)
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        valid = ~np.isnan(values)
        if not valid.any() or weights[valid].sum() <= 0:
            return float("nan")
        return float(np.average(values[valid], weights=weights[valid]))

    def _save_empty(self, fig: plt.Figure, ax: plt.Axes, title: str, message: str, figure_name: str, context: PlotContext) -> Path:
        ax.set_title(title)
        ax.text(0.5, 0.5, message, ha="center", va="center", color="#4b5563")
        ax.axis("off")
        return self._save(fig, figure_name, context)

    def _cross_summary_frame(self, tables: dict[str, pd.DataFrame], context: PlotContext) -> pd.DataFrame:
        frame = tables.get("summary_by_composition", pd.DataFrame()).copy()
        if frame.empty or "composition_id" not in frame.columns:
            return pd.DataFrame()
        frame = frame.loc[frame["model_name"] == context.model_name].copy()
        if frame.empty:
            return frame
        frame["pair"] = frame["composition_id"].astype(str).str.split(":").str[0]
        frame["combo_tail"] = frame["composition_id"].astype(str).str.split(":", n=1).str[1]
        split = frame["combo_tail"].str.split("__", n=1, expand=True)
        frame["left_subtype"] = split[0]
        frame["right_subtype"] = split[1] if split.shape[1] > 1 else ""
        return frame

    def _build_stage_breakdown_grid(self, *, frame: pd.DataFrame, context: PlotContext) -> tuple[pd.DataFrame, pd.DataFrame]:
        stage_order = list(context.manifest.stage_order)
        severity_order = list(context.manifest.severity_order)
        ordered_pairs = pd.MultiIndex.from_product([stage_order, severity_order], names=["stage_target", "severity"])
        ordered = (
            frame.set_index(["stage_target", "severity"])[["overall_task_success_rate", "sample_count"]]
            .reindex(ordered_pairs)
            .reset_index()
        )
        ordered["sample_count"] = pd.to_numeric(ordered["sample_count"], errors="coerce").fillna(0)
        ordered["overall_task_success_rate"] = pd.to_numeric(ordered["overall_task_success_rate"], errors="coerce")
        ordered.loc[ordered["sample_count"] <= 0, "overall_task_success_rate"] = np.nan

        rate_grid = ordered.pivot(index="stage_target", columns="severity", values="overall_task_success_rate").reindex(index=stage_order, columns=severity_order)
        sample_grid = (
            ordered.pivot(index="stage_target", columns="severity", values="sample_count")
            .reindex(index=stage_order, columns=severity_order)
            .fillna(0)
        )
        return rate_grid, sample_grid

    def _save(self, fig: plt.Figure, figure_name: str, context: PlotContext) -> Path:
        return self._save_static(fig, figure_name, context, self.output_dir)

    @staticmethod
    def _save_static(fig: plt.Figure, figure_name: str, context: PlotContext, output_dir: Path) -> Path:
        model_slug = context.model_name.replace("/", "-").replace(" ", "-")
        path = output_dir / f"model_{model_slug}_{figure_name}.png"
        tight_layout_rect = getattr(fig, "_toolrobust_tight_layout_rect", None)
        if tight_layout_rect is None:
            fig.tight_layout()
        else:
            fig.tight_layout(rect=tight_layout_rect)
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path
