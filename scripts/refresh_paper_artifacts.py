"""Refresh paper-level tables, figures, and appendix collages from raw records."""

from __future__ import annotations

import ast
import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ALL_ROOT = Path("data/results/all-family")
CROSS_ROOT = Path("data/results/cross-family")
PICTURE = Path("picture")
APPENDIX = PICTURE / "appendix"
DERIVED = Path("data/results/paper_artifacts")

FAMILIES = ["interface_exposure", "user_intent", "observation", "execution_environment"]
FAMILY_LABEL = {
    "interface_exposure": "A Interface",
    "user_intent": "B Intent",
    "observation": "C Observation",
    "execution_environment": "D Execution",
}
MODEL_SHORT = {
    "claude-sonnet-4-6": "Claude 4.6",
    "deepseek-v4-pro": "DS-V4 Pro",
    "gpt-5.4-mini": "GPT-5.4 Mini",
    "qwen-plus": "Qwen Plus",
    "gemini-2.5-pro": "Gemini Pro",
    "deepseek-v4-flash": "DS-V4 Flash",
    "gemini-2.5-flash": "Gemini Flash",
}
COMPOSITION_META = {
    "a+b:a2_description_abstraction__b3_distraction": ("A2+B3", "Description abstraction + distraction"),
    "a+c:a3_family_similarity_pressure__c4_return_evidence_competition": ("A3+C4", "Similarity pressure + evidence competition"),
    "a+d:a1_name_surface_shift__d1_availability_failure": ("A1+D1", "Name surface shift + availability failure"),
    "b+c:b4_conflict__c4_return_evidence_competition": ("B4+C4", "Intent conflict + evidence competition"),
    "b+d:b2_underspecification__d2_runtime_variability": ("B2+D2", "Underspecification + runtime variability"),
}
SEVERITIES = ["clean", "light", "medium", "heavy"]


def main() -> None:
    PICTURE.mkdir(exist_ok=True)
    APPENDIX.mkdir(parents=True, exist_ok=True)
    DERIVED.mkdir(parents=True, exist_ok=True)
    all_records = load_all_family_records()
    cross_records = load_selected_cross_records()
    all_summary = build_all_family_summary(all_records)
    cross_compare = build_cross_vs_single(cross_records, all_records)
    all_summary.to_csv(DERIVED / "all_family_model_summary.csv", index=False)
    cross_compare.to_csv(CROSS_ROOT / "selected_compositions_vs_single_family.csv", index=False)
    cross_compare.to_csv(DERIVED / "selected_compositions_vs_single_family.csv", index=False)
    write_cross_selected_compositions(cross_compare)
    make_figures(all_records, all_summary, cross_records, cross_compare)
    make_appendix_collages()
    organize_final_figure_layout()
    write_audit_snapshot(all_records, cross_records, all_summary, cross_compare)


def load_all_family_records() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(ALL_ROOT.glob("*/*/records.csv")):
        frame = pd.read_csv(path)
        frame["model_key"] = path.parts[-3]
        frame["family_key"] = path.parts[-2]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_selected_cross_records() -> pd.DataFrame:
    paths: list[Path] = []
    for model_dir in sorted(CROSS_ROOT.iterdir()):
        if not model_dir.is_dir():
            continue
        selected = model_dir / "selected_compositions" / "records.csv"
        if selected.exists():
            paths.append(selected)
        elif model_dir.name == "gpt-5.4-mini":
            paths.extend(sorted(model_dir.glob("*/records.csv")))
    frames: list[pd.DataFrame] = []
    selected_ids = set(COMPOSITION_META)
    for path in paths:
        frame = pd.read_csv(path)
        frame["model_key"] = path.parts[-3]
        frame["cross_source_dir"] = path.parent.name
        frame = frame.loc[(frame["split"].astype(str) == "clean") | frame["composition_id"].astype(str).isin(selected_ids)].copy()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_all_family_summary(records: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, model_frame in records.groupby("model_key"):
        family_scores = {}
        clean_scores = []
        for family in FAMILIES:
            fam = model_frame.loc[model_frame["family_key"] == family]
            pert = fam.loc[fam["split"].astype(str) == "perturbed"]
            clean = fam.loc[fam["split"].astype(str) == "clean"]
            family_scores[family] = rate(pert["overall_task_success"])
            clean_scores.append(rate(clean["overall_task_success"]))
        row = {
            "model": model,
            "model_short": MODEL_SHORT.get(model, model),
            "clean_success": float(np.nanmean(clean_scores)),
            "overall_perturbed_macro": float(np.nanmean(list(family_scores.values()))),
            **{f"{family}_success": value for family, value in family_scores.items()},
        }
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("overall_perturbed_macro", ascending=False).reset_index(drop=True)
    frame.insert(0, "rank", range(1, len(frame) + 1))
    return frame


def build_cross_vs_single(cross: pd.DataFrame, all_records: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected = cross.loc[cross["composition_id"].astype(str).isin(COMPOSITION_META)].copy()
    selected = selected.loc[selected["split"].astype(str) == "perturbed"].copy()
    for (model, composition_id, severity), group in selected.groupby(["model_key", "composition_id", "severity"]):
        families = parse_list(group["families_applied"].dropna().iloc[0])
        subtypes = parse_list(group["subtypes_applied"].dropna().iloc[0])
        single_scores = []
        for family, subtype in zip(families, subtypes, strict=False):
            fam = all_records.loc[
                (all_records["model_key"] == model)
                & (all_records["family_key"] == family)
                & (all_records["split"].astype(str) == "perturbed")
                & (all_records["subtype"].astype(str) == str(subtype))
                & (all_records["severity"].astype(str) == str(severity))
            ]
            single_scores.append(rate(fam["overall_task_success"]))
        cross_success = rate(group["overall_task_success"])
        clean = cross.loc[(cross["model_key"] == model) & (cross["split"].astype(str) == "clean")]
        clean_success = rate(clean["overall_task_success"])
        label, desc = COMPOSITION_META[composition_id]
        rows.append(
            {
                "model": model,
                "model_short": MODEL_SHORT.get(model, model),
                "composition_id": composition_id,
                "composition_label": label,
                "composition_desc": desc,
                "composition_short": f"{label}\n{desc}",
                "subtypes": " + ".join(map(str, subtypes)),
                "severity": severity,
                "cross": cross_success,
                "weakest_single": float(np.nanmin(single_scores)),
                "mean_single": float(np.nanmean(single_scores)),
                "delta_vs_weakest": cross_success - float(np.nanmin(single_scores)),
                "delta_vs_mean": cross_success - float(np.nanmean(single_scores)),
                "drop_vs_clean": clean_success - cross_success,
                "sample_count": float(len(group)),
                "call_rate": rate(group["call_present"]),
                "canonical_tool_accuracy": rate(group["canonical_tool_match"]),
                "wrong_parameter_rate": rate(group["wrong_parameter_flag"]),
                "execution_success_rate": rate(group["execution_success"]),
            }
        )
    order = {key: idx for idx, key in enumerate(COMPOSITION_META)}
    frame = pd.DataFrame(rows)
    frame["composition_order"] = frame["composition_id"].map(order)
    frame["severity_order"] = frame["severity"].map({"medium": 0, "heavy": 1})
    return frame.sort_values(["model", "composition_order", "severity_order"]).drop(columns=["composition_order", "severity_order"]).reset_index(drop=True)


def write_cross_selected_compositions(cross_compare: pd.DataFrame) -> None:
    frame = cross_compare.rename(
        columns={
            "cross": "cross_success_rate",
            "weakest_single": "weakest_single_success_rate",
            "mean_single": "mean_single_success_rate",
            "delta_vs_weakest": "delta_vs_weakest_single",
            "delta_vs_mean": "delta_vs_mean_single",
        }
    ).copy()
    frame["model_name"] = frame["model"]
    frame["pair_dir"] = frame["composition_id"].map(pair_dir)
    columns = [
        "model_name",
        "pair_dir",
        "composition_id",
        "subtypes",
        "severity",
        "cross_success_rate",
        "drop_vs_clean",
        "weakest_single_success_rate",
        "mean_single_success_rate",
        "delta_vs_weakest_single",
        "delta_vs_mean_single",
        "sample_count",
        "call_rate",
        "canonical_tool_accuracy",
        "wrong_parameter_rate",
        "execution_success_rate",
    ]
    frame[columns].to_csv(CROSS_ROOT / "selected_cross_family_compositions.csv", index=False)


def make_figures(records: pd.DataFrame, summary: pd.DataFrame, cross: pd.DataFrame, cross_compare: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 13, "axes.labelsize": 10})
    fig03_dataset_coverage(records)
    fig04_leaderboard(summary)
    fig05_clean_vs_robust(summary)
    fig06_family_heatmap(summary)
    fig07_severity_degradation(records)
    fig08_failure_taxonomy(records)
    fig09_subtype_difficulty(records)
    fig10_execution_stage2(records)
    copy_static_or_make_placeholder("fig11_cascade_diagnostic_protocol.png")
    fig12_cascade_boundary(records)
    fig13_alias_ablation(summary)
    fig14_nonmonotonic(records)
    fig15_gemini_profile(records)
    fig16_cross_matrix(cross_compare)
    fig17_cross_delta(cross_compare)
    fig18_cross_failures(cross)


def fig03_dataset_coverage(records: pd.DataFrame) -> None:
    counts = records.groupby(["family_key", "split"]).size().unstack(fill_value=0).reindex(FAMILIES)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    counts[["clean", "perturbed"]].plot(kind="bar", stacked=True, ax=ax, color=["#93c5fd", "#1d4ed8"])
    ax.set_xticklabels([FAMILY_LABEL[f] for f in counts.index], rotation=20, ha="right")
    ax.set_ylabel("records")
    ax.set_title("Single-family Dataset Coverage")
    for container in ax.containers:
        ax.bar_label(container, label_type="center", fontsize=8)
    save(fig, "fig03_dataset_coverage.png")


def fig04_leaderboard(summary: pd.DataFrame) -> None:
    frame = summary.sort_values("overall_perturbed_macro")
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.barh(frame["model_short"], frame["overall_perturbed_macro"], color="#2563eb")
    for y, value in enumerate(frame["overall_perturbed_macro"]):
        ax.text(value + 0.006, y, f"{value:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, 0.85)
    ax.set_xlabel("perturbed macro success")
    ax.set_title("Overall Robustness Leaderboard")
    save(fig, "fig04_overall_leaderboard.png")


def fig05_clean_vs_robust(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(summary["clean_success"], summary["overall_perturbed_macro"], s=90, color="#0f766e")
    for _, row in summary.iterrows():
        ax.text(row["clean_success"] + 0.002, row["overall_perturbed_macro"] + 0.002, row["model_short"], fontsize=8)
    ax.plot([0.6, 1.0], [0.6, 1.0], "--", color="#9ca3af", linewidth=1)
    ax.set_xlim(0.88, 1.01)
    ax.set_ylim(0.62, 0.79)
    ax.set_xlabel("clean success")
    ax.set_ylabel("perturbed macro success")
    ax.set_title("Clean Accuracy vs Robustness")
    save(fig, "fig05_clean_vs_robustness.png")


def fig06_family_heatmap(summary: pd.DataFrame) -> None:
    frame = summary.set_index("model_short")[[f"{f}_success" for f in FAMILIES]]
    frame.columns = [FAMILY_LABEL[f] for f in FAMILIES]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    im = ax.imshow(frame.to_numpy(), vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(frame.columns)), frame.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(frame.index)), frame.index)
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            ax.text(j, i, f"{frame.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Family-wise Robustness")
    fig.colorbar(im, ax=ax, label="success")
    save(fig, "fig06_family_model_heatmap.png")


def fig07_severity_degradation(records: pd.DataFrame) -> None:
    pert = records.copy()
    sev = pert.groupby(["model_key", "severity"])["overall_task_success"].mean().unstack().reindex(columns=SEVERITIES)
    sev = sev.loc[summary_order()]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for model, row in sev.iterrows():
        ax.plot(SEVERITIES, row, marker="o", label=MODEL_SHORT.get(model, model))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("success")
    ax.set_title("Severity Degradation")
    ax.legend(ncol=2, fontsize=8, loc="lower left")
    save(fig, "fig07_severity_degradation.png")


def fig08_failure_taxonomy(records: pd.DataFrame) -> None:
    fail = records.loc[(records["split"].astype(str) == "perturbed") & (records["overall_task_success"].astype(float) == 0)].copy()
    pivot = fail.groupby(["model_key", "primary_failure_tag"]).size().unstack(fill_value=0).reindex(summary_order()).fillna(0)
    fig, ax = plt.subplots(figsize=(10, 5.6))
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_xticklabels([MODEL_SHORT.get(m, m) for m in pivot.index], rotation=25, ha="right")
    ax.set_ylabel("failed records")
    ax.set_title("Failure Taxonomy")
    ax.legend(title="primary failure", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    save(fig, "fig08_failure_taxonomy.png")


def fig09_subtype_difficulty(records: pd.DataFrame) -> None:
    pert = records.loc[records["split"].astype(str) == "perturbed"].copy()
    frame = pert.groupby("subtype")["overall_task_success"].mean().sort_values().head(14)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([wrap(x, 32) for x in frame.index[::-1]], frame.values[::-1], color="#7c3aed")
    ax.set_xlim(0, 1)
    ax.set_xlabel("mean success")
    ax.set_title("Hardest Single-family Subtypes")
    save(fig, "fig09_subtype_difficulty.png")


def fig10_execution_stage2(records: pd.DataFrame) -> None:
    d = records.loc[(records["family_key"] == "execution_environment") & (records["split"].astype(str) == "perturbed")].copy()
    metrics = ["stage2_status_match", "stage2_retryable_match", "stage2_failure_type_match", "stage2_result_match"]
    for metric in metrics:
        d[metric] = d[metric].map(boolish)
    frame = d.groupby("model_key")[metrics].mean().reindex(summary_order())
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    im = ax.imshow(frame.to_numpy(), vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(metrics)), ["status", "retryable", "failure type", "result"], rotation=20, ha="right")
    ax.set_yticks(range(len(frame.index)), [MODEL_SHORT.get(m, m) for m in frame.index])
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            ax.text(j, i, f"{frame.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Execution Stage-2 Diagnostics")
    fig.colorbar(im, ax=ax, label="match rate")
    save(fig, "fig10_execution_stage2_diagnostics.png")


def fig12_cascade_boundary(records: pd.DataFrame) -> None:
    pert = records.loc[records["split"].astype(str) == "perturbed"].copy()
    frame = pert.groupby("model_key")[["is_cascade_failure", "boundary_violation_flag"]].mean().reindex(summary_order())
    fig, ax = plt.subplots(figsize=(8, 4.8))
    frame.plot(kind="bar", ax=ax, color=["#f97316", "#dc2626"])
    ax.set_xticklabels([MODEL_SHORT.get(m, m) for m in frame.index], rotation=25, ha="right")
    ax.set_ylabel("share of perturbed records")
    ax.set_title("Cascade and Boundary Diagnostics")
    save(fig, "fig12_cascade_boundary_diagnostics.png")


def fig13_alias_ablation(summary: pd.DataFrame) -> None:
    rows = []
    for model in summary["model"]:
        current = summary.loc[summary["model"] == model, "execution_environment_success"].iloc[0]
        backups = sorted((ALL_ROOT / model / "execution_environment").glob("records.csv.bak_before_alias_rescore_*"))
        before = current
        if backups:
            old = pd.read_csv(backups[0])
            before = rate(old.loc[old["split"].astype(str) == "perturbed", "overall_task_success"])
        rows.append({"model": model, "before": before, "after": current, "delta": current - before})
    frame = pd.DataFrame(rows).sort_values("delta")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh([MODEL_SHORT.get(m, m) for m in frame["model"]], frame["delta"], color="#16a34a")
    ax.axvline(0, color="#6b7280", linewidth=1)
    for y, value in enumerate(frame["delta"]):
        ax.text(value + 0.002, y, f"{value:+.3f}", va="center", fontsize=8)
    ax.set_xlabel("D-family success change")
    ax.set_title("Failure-type Alias Normalization Ablation")
    save(fig, "fig13_alias_normalization_ablation.png")


def fig14_nonmonotonic(records: pd.DataFrame) -> None:
    pert = records.loc[records["split"].astype(str) == "perturbed"].copy()
    table = pert.groupby(["model_key", "subtype", "severity"])["overall_task_success"].mean().unstack()
    table["medium_minus_light"] = table.get("medium", np.nan) - table.get("light", np.nan)
    table["heavy_minus_medium"] = table.get("heavy", np.nan) - table.get("medium", np.nan)
    rows = table.reset_index().melt(id_vars=["model_key", "subtype"], value_vars=["medium_minus_light", "heavy_minus_medium"], var_name="contrast", value_name="increase")
    rows = rows.sort_values("increase", ascending=False).head(12)
    labels = [f"{MODEL_SHORT.get(r.model_key, r.model_key)}\n{wrap(r.subtype, 28)}\n{r.contrast.replace('_minus_', ' > ')}" for r in rows.itertuples()]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(range(len(rows)), rows["increase"], color="#0891b2")
    ax.set_xticks(range(len(rows)), labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("success-rate increase")
    ax.set_title("Largest Non-monotonic Severity Cases")
    save(fig, "fig14_nonmonotonic_severity_cases.png")


def fig15_gemini_profile(records: pd.DataFrame) -> None:
    d = records.loc[(records["model_key"] == "gemini-2.5-pro") & (records["family_key"] == "execution_environment") & (records["split"].astype(str) == "perturbed")].copy()
    subtype = d.groupby("subtype")["overall_task_success"].mean().sort_index()
    stage2 = d[["stage2_status_match", "stage2_retryable_match", "stage2_failure_type_match", "stage2_result_match"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    axes[0].bar([wrap(x, 18) for x in subtype.index], subtype.values, color="#2563eb")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Gemini Pro D subtype success")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(["status", "retry", "failure", "result"], stage2.values, color="#0f766e")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Stage-2 match rates")
    save(fig, "fig15_gemini_pro_execution_profile.png")


def fig16_cross_matrix(cross_compare: pd.DataFrame) -> None:
    frame = cross_compare.copy()
    frame["row"] = frame["model_short"]
    frame["col"] = frame["composition_label"] + " " + frame["severity"].str[0].str.upper()
    pivot = frame.pivot(index="row", columns="col", values="cross").reindex(index=[MODEL_SHORT[m] for m in summary_order()])
    col_order = [COMPOSITION_META[c][0] + " " + s[0].upper() for c in COMPOSITION_META for s in ["medium", "heavy"]]
    pivot = pivot.reindex(columns=col_order)
    fig, ax = plt.subplots(figsize=(11, 5.8))
    im = ax.imshow(pivot.to_numpy(), vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Selected Cross-family Success Matrix")
    fig.colorbar(im, ax=ax, label="success")
    save(fig, "fig16_crossfamily_selected_success_matrix.png")


def fig17_cross_delta(cross_compare: pd.DataFrame) -> None:
    agg = cross_compare.groupby(["composition_label", "severity"])[["cross", "weakest_single", "delta_vs_weakest"]].mean().reset_index()
    agg["label"] = agg["composition_label"] + " " + agg["severity"].str[0].str.upper()
    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(len(agg))
    ax.bar(x - 0.18, agg["weakest_single"], width=0.36, color="#9ca3af", label="weakest single")
    ax.bar(x + 0.18, agg["cross"], width=0.36, color="#2563eb", label="cross")
    for idx, delta in enumerate(agg["delta_vs_weakest"]):
        ax.text(idx, max(agg.loc[idx, "cross"], agg.loc[idx, "weakest_single"]) + 0.025, f"{delta:+.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, agg["label"], rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("success")
    ax.set_title("Cross-family vs Weakest Single-family Component")
    ax.legend()
    save(fig, "fig17_crossfamily_vs_single_family_delta.png")


def fig18_cross_failures(cross: pd.DataFrame) -> None:
    pert = cross.loc[(cross["split"].astype(str) == "perturbed") & (cross["composition_id"].astype(str).isin(COMPOSITION_META))].copy()
    fail = pert.loc[pert["overall_task_success"].astype(float) == 0]
    pivot = fail.groupby(["composition_id", "primary_failure_tag"]).size().unstack(fill_value=0)
    pivot = pivot.reindex(index=list(COMPOSITION_META))
    fig, ax = plt.subplots(figsize=(10, 5.8))
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_xticklabels([COMPOSITION_META[c][0] for c in pivot.index], rotation=25, ha="right")
    ax.set_ylabel("failed records")
    ax.set_title("Cross-family Failure Diagnostics")
    ax.legend(title="primary failure", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    save(fig, "fig18_crossfamily_failure_diagnostics.png")


def make_appendix_collages() -> None:
    for idx, family in enumerate(FAMILIES, start=1):
        rows = []
        for model in summary_order():
            plot_dir = ALL_ROOT / model / family / "plots"
            rows.append(
                [
                    plot_dir / f"model_{model}_subtype_heatmap.png",
                    plot_dir / f"model_{model}_stage_breakdown.png",
                    plot_dir / f"model_{model}_failure_taxonomy_stacked_bar.png",
                ]
            )
        make_collage(rows, APPENDIX / f"app_fig{idx:02d}_{family}_diagnostic_sheet.png", title=f"{FAMILY_LABEL[family]} Diagnostic Sheet", row_labels=[MODEL_SHORT[m] for m in summary_order()], col_labels=["Subtype", "Stage", "Failure"])
    shutil.copy2(PICTURE / "fig16_crossfamily_selected_success_matrix.png", APPENDIX / "app_fig05_crossfamily_selected_success_matrix.png")
    shutil.copy2(PICTURE / "fig17_crossfamily_vs_single_family_delta.png", APPENDIX / "app_fig06_crossfamily_vs_single_family_delta.png")
    shutil.copy2(PICTURE / "fig18_crossfamily_failure_diagnostics.png", APPENDIX / "app_fig07_crossfamily_failure_diagnostics.png")
    rows = []
    for model in summary_order():
        if model == "gpt-5.4-mini":
            roots = sorted((CROSS_ROOT / model).glob("*/plots"))
            rows.append([
                first_existing([r / f"model_{model}_cross_pair_success_heatmap.png" for r in roots]),
                first_existing([r / f"model_{model}_stage_breakdown.png" for r in roots]),
                first_existing([r / f"model_{model}_failure_taxonomy_stacked_bar.png" for r in roots]),
            ])
        else:
            plot_dir = CROSS_ROOT / model / "selected_compositions" / "plots"
            rows.append([
                plot_dir / f"model_{model}_cross_pair_success_heatmap.png",
                plot_dir / f"model_{model}_stage_breakdown.png",
                plot_dir / f"model_{model}_failure_taxonomy_stacked_bar.png",
            ])
    make_collage(rows, APPENDIX / "app_fig08_crossfamily_model_diagnostic_sheet.png", title="Cross-family Model Diagnostic Sheet", row_labels=[MODEL_SHORT[m] for m in summary_order()], col_labels=["Pair success", "Stage", "Failure"])
    gpt_matrices = sorted((CROSS_ROOT / "gpt-5.4-mini").glob("*/plots/model_gpt-5.4-mini_cross_all_compositions_success_matrix.png"))
    if gpt_matrices:
        pair_labels = {
            "interface_exposure_execution_environment": "A+D",
            "interface_exposure_observation": "A+C",
            "interface_exposure_user_intent": "A+B",
            "user_intent_execution_environment": "B+D",
            "user_intent_observation": "B+C",
        }
        make_collage(
            [gpt_matrices],
            APPENDIX / "app_fig09_gpt54mini_full_crossfamily_composition_matrix.png",
            title="GPT-5.4 Mini Full Pairwise Composition Matrices",
            row_labels=["GPT-5.4 Mini"],
            col_labels=[pair_labels.get(p.parent.parent.name, p.parent.parent.name.replace("_", "+")) for p in gpt_matrices],
        )


def organize_final_figure_layout() -> None:
    """Keep a compact main-text figure set and move detailed plots to appendix."""
    main_layout = [
        ("fig01_toolrobustbench_pipeline.png", "fig01_toolrobustbench_pipeline.png"),
        ("fig04_overall_leaderboard.png", "fig02_overall_leaderboard.png"),
        ("fig06_family_model_heatmap.png", "fig03_family_model_heatmap.png"),
        ("fig08_failure_taxonomy.png", "fig04_failure_taxonomy.png"),
        ("fig12_cascade_boundary_diagnostics.png", "fig05_cascade_boundary_diagnostics.png"),
        ("fig17_crossfamily_vs_single_family_delta.png", "fig06_crossfamily_vs_single_family_delta.png"),
    ]
    appendix_layout = [
        (PICTURE / "fig02_perturbation_taxonomy.png", "app_fig01_perturbation_taxonomy.png"),
        (PICTURE / "fig03_dataset_coverage.png", "app_fig02_dataset_coverage.png"),
        (PICTURE / "fig05_clean_vs_robustness.png", "app_fig03_clean_vs_robustness.png"),
        (PICTURE / "fig07_severity_degradation.png", "app_fig04_severity_degradation.png"),
        (PICTURE / "fig09_subtype_difficulty.png", "app_fig05_subtype_difficulty.png"),
        (PICTURE / "fig10_execution_stage2_diagnostics.png", "app_fig06_execution_stage2_diagnostics.png"),
        (PICTURE / "fig11_cascade_diagnostic_protocol.png", "app_fig07_cascade_diagnostic_protocol.png"),
        (PICTURE / "fig13_alias_normalization_ablation.png", "app_fig08_alias_normalization_ablation.png"),
        (PICTURE / "fig14_nonmonotonic_severity_cases.png", "app_fig09_nonmonotonic_severity_cases.png"),
        (PICTURE / "fig15_gemini_pro_execution_profile.png", "app_fig10_gemini_pro_execution_profile.png"),
        (PICTURE / "fig16_crossfamily_selected_success_matrix.png", "app_fig11_crossfamily_selected_success_matrix.png"),
        (PICTURE / "fig18_crossfamily_failure_diagnostics.png", "app_fig12_crossfamily_failure_diagnostics.png"),
        (APPENDIX / "app_fig01_interface_exposure_diagnostic_sheet.png", "app_fig13_interface_exposure_diagnostic_sheet.png"),
        (APPENDIX / "app_fig02_user_intent_diagnostic_sheet.png", "app_fig14_user_intent_diagnostic_sheet.png"),
        (APPENDIX / "app_fig03_observation_diagnostic_sheet.png", "app_fig15_observation_diagnostic_sheet.png"),
        (APPENDIX / "app_fig04_execution_environment_diagnostic_sheet.png", "app_fig16_execution_environment_diagnostic_sheet.png"),
        (APPENDIX / "app_fig08_crossfamily_model_diagnostic_sheet.png", "app_fig17_crossfamily_model_diagnostic_sheet.png"),
        (APPENDIX / "app_fig09_gpt54mini_full_crossfamily_composition_matrix.png", "app_fig18_gpt54mini_full_crossfamily_composition_matrix.png"),
    ]

    main_tmp = PICTURE / "_final_main_tmp"
    app_tmp = APPENDIX / "_final_appendix_tmp"
    if main_tmp.exists():
        shutil.rmtree(main_tmp)
    if app_tmp.exists():
        shutil.rmtree(app_tmp)
    main_tmp.mkdir()
    app_tmp.mkdir()

    for src_name, dst_name in main_layout:
        src = PICTURE / src_name
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, main_tmp / dst_name)
    for src, dst_name in appendix_layout:
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, app_tmp / dst_name)

    for path in PICTURE.glob("fig*.png"):
        path.unlink()
    for path in APPENDIX.glob("app_fig*.png"):
        path.unlink()
    for path in main_tmp.iterdir():
        shutil.move(str(path), PICTURE / path.name)
    for path in app_tmp.iterdir():
        shutil.move(str(path), APPENDIX / path.name)
    main_tmp.rmdir()
    app_tmp.rmdir()


def make_collage(rows: list[list[Path | None]], out: Path, *, title: str, row_labels: list[str], col_labels: list[str]) -> None:
    thumb_w, thumb_h = 360, 240
    left, top = 150, 80
    header_h = 40
    canvas = Image.new("RGB", (left + thumb_w * len(col_labels), top + header_h + thumb_h * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((20, 20), title, fill=(17, 24, 39), font=font)
    for j, label in enumerate(col_labels):
        draw.text((left + j * thumb_w + 8, top), label, fill=(17, 24, 39), font=font)
    for i, row in enumerate(rows):
        draw.text((10, top + header_h + i * thumb_h + 8), row_labels[i], fill=(17, 24, 39), font=font)
        for j, path in enumerate(row):
            box = (left + j * thumb_w, top + header_h + i * thumb_h)
            if path and path.exists():
                image = Image.open(path).convert("RGB")
                image.thumbnail((thumb_w - 8, thumb_h - 8))
                canvas.paste(image, (box[0] + 4, box[1] + 4))
            else:
                draw.rectangle([box[0] + 4, box[1] + 4, box[0] + thumb_w - 4, box[1] + thumb_h - 4], outline=(209, 213, 219))
                draw.text((box[0] + 20, box[1] + 30), "missing", fill=(107, 114, 128), font=font)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)


def write_audit_snapshot(records: pd.DataFrame, cross: pd.DataFrame, summary: pd.DataFrame, cross_compare: pd.DataFrame) -> None:
    snapshot = {
        "all_family_records": len(records),
        "cross_selected_records": len(cross.loc[cross["composition_id"].astype(str).isin(COMPOSITION_META)]),
        "model_count": summary["model"].nunique(),
        "cross_cells": len(cross_compare),
        "overall_mean": summary["overall_perturbed_macro"].mean(),
        "clean_mean": summary["clean_success"].mean(),
        "cross_medium_mean": cross_compare.loc[cross_compare["severity"] == "medium", "cross"].mean(),
        "cross_heavy_mean": cross_compare.loc[cross_compare["severity"] == "heavy", "cross"].mean(),
    }
    pd.DataFrame([snapshot]).to_csv(DERIVED / "paper_artifact_audit_snapshot.csv", index=False)


def rate(series: pd.Series) -> float:
    if len(series) == 0:
        return float("nan")
    return float(pd.to_numeric(series, errors="coerce").fillna(0).mean())


def boolish(value: object) -> float:
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "yes"}:
        return 1.0
    if text in {"0", "0.0", "false", "no"}:
        return 0.0
    return float("nan")


def parse_list(value: object) -> list[Any]:
    if isinstance(value, list):
        return value
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def pair_dir(composition_id: str) -> str:
    pair = composition_id.split(":", 1)[0]
    return {
        "a+b": "interface_exposure_user_intent",
        "a+c": "interface_exposure_observation",
        "a+d": "interface_exposure_execution_environment",
        "b+c": "user_intent_observation",
        "b+d": "user_intent_execution_environment",
    }[pair]


def summary_order() -> list[str]:
    path = DERIVED / "all_family_model_summary.csv"
    if path.exists():
        return pd.read_csv(path)["model"].tolist()
    return ["claude-sonnet-4-6", "deepseek-v4-pro", "gpt-5.4-mini", "qwen-plus", "gemini-2.5-pro", "deepseek-v4-flash", "gemini-2.5-flash"]


def save(fig: plt.Figure, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(PICTURE / filename, dpi=220)
    plt.close(fig)


def wrap(text: str, width: int) -> str:
    words = str(text).split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        if sum(len(x) for x in current) + len(current) + len(word) > width and current:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path and path.exists():
            return path
    return None


def copy_static_or_make_placeholder(filename: str) -> None:
    path = PICTURE / filename
    if path.exists():
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.text(0.5, 0.5, filename, ha="center", va="center")
    ax.axis("off")
    save(fig, filename)


if __name__ == "__main__":
    main()
