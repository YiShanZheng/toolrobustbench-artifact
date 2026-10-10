"""Audit positive mixed-minus-weaker contrasts in the exact matched design.

The analysis uses the 1,120 records produced by matched_mixed_bootstrap.py.
It reports pair/severity point estimates, source-anchor cluster bootstrap
intervals, leave-one-anchor-out ranges, per-model deltas, and paired outcome
transitions relative to the weaker constituent selected within each model cell.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
INPUT = HERE / "matched_mixed_records.csv"
SEED = 20261002
BOOTSTRAPS = 20_000

TARGETS = {
    ("a+b:a2_description_abstraction__b3_distraction", "heavy"): "A2+B3 heavy",
    ("b+d:b2_underspecification__d2_runtime_variability", "medium"): "B2+D2 medium",
}


def summarize_cells(frame: pd.DataFrame) -> pd.DataFrame:
    cells = (
        frame.groupby(["model_name", "composition_id", "severity"], as_index=False)
        .agg(
            n=("mixed_success", "size"),
            mixed=("mixed_success", "mean"),
            constituent_1=("constituent_1_success", "mean"),
            constituent_2=("constituent_2_success", "mean"),
        )
    )
    cells["weaker_index"] = np.where(
        cells["constituent_1"] <= cells["constituent_2"], 1, 2
    )
    cells["weaker"] = cells[["constituent_1", "constituent_2"]].min(axis=1)
    cells["constituent_mean"] = cells[["constituent_1", "constituent_2"]].mean(axis=1)
    cells["mixed_minus_weaker"] = cells["mixed"] - cells["weaker"]
    cells["mixed_minus_mean"] = cells["mixed"] - cells["constituent_mean"]
    return cells


def pair_estimate(frame: pd.DataFrame, composition_id: str, severity: str) -> dict[str, float]:
    cells = summarize_cells(frame)
    part = cells.loc[
        cells["composition_id"].eq(composition_id) & cells["severity"].eq(severity)
    ]
    return {
        "mixed": part["mixed"].mean(),
        "weaker": part["weaker"].mean(),
        "constituent_mean": part["constituent_mean"].mean(),
        "mixed_minus_weaker": part["mixed_minus_weaker"].mean(),
        "mixed_minus_mean": part["mixed_minus_mean"].mean(),
    }


def bootstrap(frame: pd.DataFrame, composition_id: str, severity: str) -> np.ndarray:
    part = frame.loc[
        frame["composition_id"].eq(composition_id) & frame["severity"].eq(severity)
    ].copy()
    anchors = np.array(sorted(part["source_sample_id"].unique()))
    models = np.array(sorted(part["model_name"].unique()))
    anchor_index = {value: index for index, value in enumerate(anchors)}
    model_index = {value: index for index, value in enumerate(models)}
    arrays = np.full((3, len(models), len(anchors)), np.nan, dtype=float)
    for row in part.itertuples(index=False):
        mi = model_index[row.model_name]
        ai = anchor_index[row.source_sample_id]
        arrays[:, mi, ai] = [
            row.mixed_success,
            row.constituent_1_success,
            row.constituent_2_success,
        ]
    if np.isnan(arrays).any():
        raise ValueError(f"Incomplete model-by-anchor rectangle for {composition_id}/{severity}")
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(anchors), size=(BOOTSTRAPS, len(anchors)))
    weights = np.zeros((BOOTSTRAPS, len(anchors)), dtype=np.int16)
    for position in range(len(anchors)):
        weights[np.arange(BOOTSTRAPS), draws[:, position]] += 1
    rates = np.einsum("ba,cma->bcm", weights, arrays) / len(anchors)
    mixed = rates[:, 0, :]
    weaker = np.minimum(rates[:, 1, :], rates[:, 2, :])
    return (mixed - weaker).mean(axis=1)


def transitions(frame: pd.DataFrame, composition_id: str, severity: str) -> pd.DataFrame:
    part = frame.loc[
        frame["composition_id"].eq(composition_id) & frame["severity"].eq(severity)
    ].copy()
    cells = summarize_cells(part)[["model_name", "weaker_index"]]
    part = part.merge(cells, on="model_name", validate="many_to_one")
    part["weaker_success"] = np.where(
        part["weaker_index"].eq(1),
        part["constituent_1_success"],
        part["constituent_2_success"],
    )
    part["transition"] = np.select(
        [
            part["weaker_success"].eq(0) & part["mixed_success"].eq(1),
            part["weaker_success"].eq(1) & part["mixed_success"].eq(0),
            part["weaker_success"].eq(1) & part["mixed_success"].eq(1),
        ],
        ["fail_to_success", "success_to_fail", "both_success"],
        default="both_fail",
    )
    return (
        part.groupby(["model_name", "transition"], as_index=False)
        .size()
        .pivot(index="model_name", columns="transition", values="size")
        .fillna(0)
        .astype(int)
        .reset_index()
    )


def main() -> None:
    frame = pd.read_csv(INPUT)
    cells = summarize_cells(frame)
    report_rows = []
    model_rows = []
    transition_rows = []

    for (composition_id, severity), label in TARGETS.items():
        point = pair_estimate(frame, composition_id, severity)
        draws = bootstrap(frame, composition_id, severity)
        loo = []
        for anchor in sorted(frame["source_sample_id"].unique()):
            reduced = frame.loc[frame["source_sample_id"].ne(anchor)]
            loo.append(pair_estimate(reduced, composition_id, severity)["mixed_minus_weaker"])
        subset = cells.loc[
            cells["composition_id"].eq(composition_id) & cells["severity"].eq(severity)
        ].copy()
        report_rows.append(
            {
                "condition": label,
                **point,
                "bootstrap_ci_low": np.quantile(draws, 0.025),
                "bootstrap_ci_high": np.quantile(draws, 0.975),
                "positive_model_cells": int((subset["mixed_minus_weaker"] > 0).sum()),
                "zero_model_cells": int((subset["mixed_minus_weaker"] == 0).sum()),
                "negative_model_cells": int((subset["mixed_minus_weaker"] < 0).sum()),
                "loo_min": min(loo),
                "loo_max": max(loo),
            }
        )
        subset.insert(0, "condition", label)
        model_rows.append(subset)
        trans = transitions(frame, composition_id, severity)
        trans.insert(0, "condition", label)
        transition_rows.append(trans)

    report = pd.DataFrame(report_rows)
    models = pd.concat(model_rows, ignore_index=True)
    trans = pd.concat(transition_rows, ignore_index=True)
    report.to_csv(HERE / "positive_mixed_reversal_summary.csv", index=False)
    models.to_csv(HERE / "positive_mixed_reversal_by_model.csv", index=False)
    trans.to_csv(HERE / "positive_mixed_reversal_transitions.csv", index=False)
    print(report.to_string(index=False))
    print("\nPer-model cells:\n", models.to_string(index=False))
    print("\nPaired transitions:\n", trans.to_string(index=False))


if __name__ == "__main__":
    main()
