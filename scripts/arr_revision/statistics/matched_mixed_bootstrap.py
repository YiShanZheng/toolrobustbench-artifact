"""Matched descriptive analysis for the five selected mixed-family conditions.

Every mixed record is joined to the two single-family records with the same
model, clean source anchor, severity, and constituent subtype.  Uncertainty is
estimated by resampling the 16 source anchors as whole clusters.  The reported
differences are descriptive contrasts; they are not causal interaction effects.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SEED = 20261002
BOOTSTRAPS = 20_000
SELECTED = (
    "a+b:a2_description_abstraction__b3_distraction",
    "a+c:a3_family_similarity_pressure__c4_return_evidence_competition",
    "a+d:a1_name_surface_shift__d1_availability_failure",
    "b+c:b4_conflict__c4_return_evidence_competition",
    "b+d:b2_underspecification__d2_runtime_variability",
)
PAIR_LABELS = {
    SELECTED[0]: "A2+B3",
    SELECTED[1]: "A3+C4",
    SELECTED[2]: "A1+D1",
    SELECTED[3]: "B4+C4",
    SELECTED[4]: "B2+D2",
}


def load_records() -> tuple[pd.DataFrame, pd.DataFrame]:
    single_files = sorted((ROOT / "data/results/all-family").glob("*/*/records.csv"))
    cross_files = sorted((ROOT / "data/results/cross-family").glob("*/*/records.csv"))
    single = pd.concat((pd.read_csv(path) for path in single_files), ignore_index=True)
    cross = pd.concat((pd.read_csv(path) for path in cross_files), ignore_index=True)
    single = single.loc[single["split"].eq("perturbed")].copy()
    cross = cross.loc[cross["composition_id"].isin(SELECTED)].copy()
    return single, cross


def build_matched(single: pd.DataFrame, cross: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for composition_id, part in cross.groupby("composition_id", sort=True):
        subtype_values = part["subtypes_applied"].dropna().unique()
        if len(subtype_values) != 1:
            raise ValueError(f"Non-unique subtype definition for {composition_id}")
        subtypes = ast.literal_eval(str(subtype_values[0]))
        if not isinstance(subtypes, list) or len(subtypes) != 2:
            raise ValueError(f"Expected two constituent subtypes for {composition_id}")

        base = part[
            [
                "model_name",
                "sample_id",
                "source_sample_id",
                "gold_canonical_tool_id",
                "composition_id",
                "severity",
                "overall_task_success",
            ]
        ].rename(columns={"overall_task_success": "mixed_success"})
        for index, subtype in enumerate(subtypes, start=1):
            constituent = single.loc[
                single["subtype"].eq(subtype),
                ["model_name", "source_sample_id", "severity", "overall_task_success"],
            ].rename(columns={"overall_task_success": f"constituent_{index}_success"})
            if constituent.duplicated(["model_name", "source_sample_id", "severity"]).any():
                raise ValueError(f"Duplicate single-family match for {subtype}")
            base = base.merge(
                constituent,
                on=["model_name", "source_sample_id", "severity"],
                how="left",
                validate="one_to_one",
            )
            base[f"constituent_{index}_subtype"] = subtype
        rows.append(base)

    matched = pd.concat(rows, ignore_index=True)
    required = ["mixed_success", "constituent_1_success", "constituent_2_success"]
    if matched[required].isna().any().any():
        raise ValueError("At least one mixed record lacks an exact matched constituent")
    if len(matched) != 1_120:
        raise ValueError(f"Expected 1,120 selected mixed records, got {len(matched)}")
    if matched.duplicated(["model_name", "sample_id", "composition_id", "severity"]).any():
        raise ValueError("Duplicate selected mixed record")
    return matched


def cell_summary(frame: pd.DataFrame) -> pd.DataFrame:
    cells = (
        frame.groupby(["model_name", "composition_id", "severity"], as_index=False)
        .agg(
            n=("mixed_success", "size"),
            mixed=("mixed_success", "mean"),
            constituent_1=("constituent_1_success", "mean"),
            constituent_2=("constituent_2_success", "mean"),
        )
    )
    cells["weaker_constituent"] = cells[["constituent_1", "constituent_2"]].min(axis=1)
    cells["constituent_mean"] = cells[["constituent_1", "constituent_2"]].mean(axis=1)
    cells["mixed_minus_weaker"] = cells["mixed"] - cells["weaker_constituent"]
    cells["mixed_minus_constituent_mean"] = cells["mixed"] - cells["constituent_mean"]
    return cells


def aggregate(cells: pd.DataFrame) -> pd.DataFrame:
    outputs = []
    for severity, part in [("all", cells), *cells.groupby("severity", sort=True)]:
        outputs.append(
            {
                "severity": severity,
                "n_cells": len(part),
                "mixed": part["mixed"].mean(),
                "weaker_constituent": part["weaker_constituent"].mean(),
                "constituent_mean": part["constituent_mean"].mean(),
                "mixed_minus_weaker": part["mixed_minus_weaker"].mean(),
                "mixed_minus_constituent_mean": part["mixed_minus_constituent_mean"].mean(),
            }
        )
    return pd.DataFrame(outputs)


def bootstrap(matched: pd.DataFrame) -> pd.DataFrame:
    anchors = sorted(matched["source_sample_id"].unique())
    if len(anchors) != 16:
        raise ValueError(f"Expected 16 source-anchor clusters, got {len(anchors)}")
    cell_keys = ["model_name", "composition_id", "severity"]
    cells = matched[cell_keys].drop_duplicates().sort_values(cell_keys).reset_index(drop=True)
    if len(cells) != 70:
        raise ValueError(f"Expected 70 model-composition-severity cells, got {len(cells)}")
    cell_index = {tuple(row): i for i, row in enumerate(cells.itertuples(index=False, name=None))}
    anchor_index = {name: i for i, name in enumerate(anchors)}
    arrays = np.full((3, len(cells), len(anchors)), np.nan, dtype=float)
    for row in matched.itertuples(index=False):
        ci = cell_index[(row.model_name, row.composition_id, row.severity)]
        ai = anchor_index[row.source_sample_id]
        arrays[:, ci, ai] = [row.mixed_success, row.constituent_1_success, row.constituent_2_success]
    if np.isnan(arrays).any():
        raise ValueError("Matched design is not a complete 70 x 16 rectangle")

    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(anchors), size=(BOOTSTRAPS, len(anchors)))
    weights = np.zeros((BOOTSTRAPS, len(anchors)), dtype=np.int16)
    for position in range(len(anchors)):
        weights[np.arange(BOOTSTRAPS), draws[:, position]] += 1
    rates = np.einsum("bq,mcq->bmc", weights, arrays) / len(anchors)
    mixed = rates[:, 0, :]
    weaker = np.minimum(rates[:, 1, :], rates[:, 2, :])
    constituent_mean = (rates[:, 1, :] + rates[:, 2, :]) / 2

    outputs = []
    masks = {"all": np.ones(len(cells), dtype=bool)}
    masks.update({severity: cells["severity"].eq(severity).to_numpy() for severity in sorted(cells["severity"].unique())})
    for severity, mask in masks.items():
        metrics = {
            "mixed": mixed[:, mask].mean(axis=1),
            "weaker_constituent": weaker[:, mask].mean(axis=1),
            "constituent_mean": constituent_mean[:, mask].mean(axis=1),
            "mixed_minus_weaker": (mixed[:, mask] - weaker[:, mask]).mean(axis=1),
            "mixed_minus_constituent_mean": (mixed[:, mask] - constituent_mean[:, mask]).mean(axis=1),
        }
        for metric, values in metrics.items():
            outputs.append(
                {
                    "severity": severity,
                    "metric": metric,
                    "bootstrap_replicates": BOOTSTRAPS,
                    "cluster_unit": "source_sample_id (16 anchors)",
                    "ci_low": np.quantile(values, 0.025),
                    "ci_high": np.quantile(values, 0.975),
                }
            )
    return pd.DataFrame(outputs)


def leave_one_anchor_out(matched: pd.DataFrame) -> pd.DataFrame:
    outputs = []
    for anchor in sorted(matched["source_sample_id"].unique()):
        summary = aggregate(cell_summary(matched.loc[matched["source_sample_id"].ne(anchor)]))
        row = summary.loc[summary["severity"].eq("all")].iloc[0].to_dict()
        row["left_out_source_sample_id"] = anchor
        outputs.append(row)
    return pd.DataFrame(outputs)


def make_figure(cells: pd.DataFrame) -> None:
    plot = (
        cells.groupby(["composition_id", "severity"], as_index=False)[
            ["mixed_minus_weaker", "mixed_minus_constituent_mean"]
        ]
        .mean()
    )
    pairs = list(SELECTED)
    x = np.arange(len(pairs))
    width = 0.36
    fig, axes = plt.subplots(2, 1, figsize=(5.0, 6.2), sharex=True)
    for ax, metric, title in (
        (axes[0], "mixed_minus_weaker", "Matched gap vs. weaker constituent"),
        (axes[1], "mixed_minus_constituent_mean", "Matched gap vs. constituent mean"),
    ):
        for offset, severity, color in ((-width / 2, "medium", "#2A6FBB"), (width / 2, "heavy", "#D55E00")):
            values = []
            for pair in pairs:
                value = plot.loc[
                    plot["composition_id"].eq(pair) & plot["severity"].eq(severity), metric
                ].iloc[0]
                values.append(value)
            ax.bar(x + offset, values, width, label=severity.title(), color=color)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x, [PAIR_LABELS[pair] for pair in pairs])
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylabel("Mixed minus baseline", fontsize=9)
    axes[0].legend(frameon=False, loc="lower left", fontsize=8)
    axes[1].tick_params(axis="x", labelrotation=0)
    fig.tight_layout()
    fig.savefig(OUT / "matched_mixed_vs_constituents.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    single, cross = load_records()
    matched = build_matched(single, cross)
    cells = cell_summary(matched)
    make_figure(cells)
    summary = aggregate(cells)
    intervals = bootstrap(matched)
    interval_wide = intervals.pivot(index="severity", columns="metric", values=["ci_low", "ci_high"])
    interval_wide.columns = ["_".join(item) for item in interval_wide.columns]
    summary = summary.merge(interval_wide, left_on="severity", right_index=True)

    matched.to_csv(OUT / "matched_mixed_records.csv", index=False)
    cells.to_csv(OUT / "matched_mixed_cells.csv", index=False)
    summary.to_csv(OUT / "matched_mixed_summary.csv", index=False)
    leave_one_anchor_out(matched).to_csv(OUT / "matched_mixed_leave_one_anchor_out.csv", index=False)
    (OUT / "matched_mixed_manifest.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "bootstrap_replicates": BOOTSTRAPS,
                "n_matched_records": len(matched),
                "n_cells": len(cells),
                "n_source_anchor_clusters": matched["source_sample_id"].nunique(),
                "join_keys": ["model_name", "source_sample_id", "severity", "constituent subtype"],
                "interpretation": "descriptive matched contrasts, not causal interactions",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
