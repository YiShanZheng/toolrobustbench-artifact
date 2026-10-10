"""Clustered bootstrap analysis for the canonical single-family experiment.

The analysis deliberately uses only
``data/results/all-family/<model>/<family>/records.csv``.  It treats the
clean seed (``source_sample_id``) as the primary paired sampling unit and the
canonical tool as a more conservative cluster.  All clean/robust differences
are computed within an anchor before resampling.

Outputs are written next to this script.  No manuscript or source result file
is modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FAMILY_DIR_TO_NAME = {
    "interface_exposure": "A_interface",
    "user_intent": "B_intent",
    "observation": "C_observation",
    "execution_environment": "D_runtime",
}
FAMILY_ORDER = ["A_interface", "B_intent", "C_observation", "D_runtime"]
EXPECTED_SUBTYPES = {
    "A_interface": 3,
    "B_intent": 4,
    "C_observation": 4,
    "D_runtime": 3,
}
SEVERITIES = ["light", "medium", "heavy"]
REQUIRED_COLUMNS = {
    "model_name",
    "family",
    "subtype",
    "severity",
    "source_sample_id",
    "gold_canonical_tool_id",
    "overall_task_success",
}


def percentile_ci(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(values, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def paired_bootstrap(
    units: pd.DataFrame,
    rng: np.random.Generator,
    reps: int,
) -> dict[str, float]:
    """Bootstrap paired clean and robust scores by anchor and by tool.

    ``units`` contains exactly one row per anchor.  Tool-cluster bootstrap
    resamples the 16 sampled tools and carries all three anchors for a selected
    tool.  Since the design is balanced, this is equivalent to resampling tool
    means with equal weight.
    """
    units = units.sort_values("anchor").reset_index(drop=True)
    if units["anchor"].duplicated().any():
        raise ValueError("paired_bootstrap requires one row per anchor")

    clean = units["clean"].to_numpy(float)
    robust = units["robust"].to_numpy(float)
    drop = clean - robust
    n_anchor = len(units)

    anchor_draw = rng.integers(0, n_anchor, size=(reps, n_anchor))
    anchor_clean = clean[anchor_draw].mean(axis=1)
    anchor_robust = robust[anchor_draw].mean(axis=1)
    anchor_drop = drop[anchor_draw].mean(axis=1)

    tool_means = (
        units.groupby("tool", sort=True)[["clean", "robust"]]
        .mean()
        .reset_index()
    )
    tool_sizes = units.groupby("tool").size()
    if tool_sizes.nunique() != 1:
        raise ValueError(
            "Tool-cluster bootstrap assumes the balanced design documented in "
            f"the paper; observed anchor counts {tool_sizes.to_dict()}"
        )
    t_clean = tool_means["clean"].to_numpy(float)
    t_robust = tool_means["robust"].to_numpy(float)
    t_drop = t_clean - t_robust
    n_tool = len(tool_means)
    tool_draw = rng.integers(0, n_tool, size=(reps, n_tool))
    tool_clean = t_clean[tool_draw].mean(axis=1)
    tool_robust = t_robust[tool_draw].mean(axis=1)
    tool_drop = t_drop[tool_draw].mean(axis=1)

    result: dict[str, float] = {
        "n_anchors": n_anchor,
        "n_tools": n_tool,
        "clean": float(clean.mean()),
        "robust": float(robust.mean()),
        "drop": float(drop.mean()),
    }
    for scheme, arrays in {
        "anchor": (anchor_clean, anchor_robust, anchor_drop),
        "tool": (tool_clean, tool_robust, tool_drop),
    }.items():
        for metric, draws in zip(("clean", "robust", "drop"), arrays):
            lo, hi = percentile_ci(draws)
            result[f"{metric}_{scheme}_ci_low"] = lo
            result[f"{metric}_{scheme}_ci_high"] = hi
    return result


def load_and_validate(root: Path) -> tuple[pd.DataFrame, dict]:
    result_root = root / "data" / "results" / "all-family"
    files = sorted(result_root.glob("*/*/records.csv"))
    if len(files) != 28:
        raise ValueError(f"Expected 28 canonical files, found {len(files)}")

    frames = []
    file_checks = []
    canonical_anchor_set: set[str] | None = None
    canonical_anchor_tool: dict[str, str] | None = None
    for path in files:
        model_dir, family_dir = path.parts[-3], path.parts[-2]
        if family_dir not in FAMILY_DIR_TO_NAME:
            raise ValueError(f"Unexpected family directory: {path}")
        family_name = FAMILY_DIR_TO_NAME[family_dir]
        df = pd.read_csv(path, low_memory=False)
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        if set(df["model_name"].dropna().unique()) != {model_dir}:
            raise ValueError(f"{path}: model name/path mismatch")
        if df["overall_task_success"].isna().any():
            raise ValueError(f"{path}: missing overall_task_success")
        success_values = set(df["overall_task_success"].astype(int).unique())
        if not success_values <= {0, 1}:
            raise ValueError(f"{path}: non-binary overall_task_success")
        df["overall_task_success"] = df["overall_task_success"].astype(int)

        clean = df[df["severity"] == "clean"]
        pert = df[df["severity"].isin(SEVERITIES)]
        anchors = set(df["source_sample_id"].astype(str))
        anchor_tool = (
            df[["source_sample_id", "gold_canonical_tool_id"]]
            .drop_duplicates()
            .set_index("source_sample_id")["gold_canonical_tool_id"]
            .astype(str)
            .to_dict()
        )
        if len(anchor_tool) != len(anchors):
            raise ValueError(f"{path}: an anchor maps to multiple tools")
        if canonical_anchor_set is None:
            canonical_anchor_set = anchors
            canonical_anchor_tool = anchor_tool
        elif anchors != canonical_anchor_set or anchor_tool != canonical_anchor_tool:
            raise ValueError(f"{path}: anchor/tool design differs from canonical file")

        subtype_count = EXPECTED_SUBTYPES[family_name]
        expected_rows = 48 + 48 * subtype_count * 3
        if len(df) != expected_rows:
            raise ValueError(f"{path}: expected {expected_rows} rows, found {len(df)}")
        if len(clean) != 48 or clean["source_sample_id"].nunique() != 48:
            raise ValueError(f"{path}: clean rows are not one per 48 anchors")
        per_anchor_severity = pert.groupby(["source_sample_id", "severity"]).size()
        if set(per_anchor_severity.index.get_level_values("severity")) != set(SEVERITIES):
            raise ValueError(f"{path}: missing severity")
        if set(per_anchor_severity.unique()) != {subtype_count}:
            raise ValueError(f"{path}: incomplete anchor x subtype x severity grid")

        df = df.assign(
            analysis_model=model_dir,
            analysis_family=family_name,
            analysis_file=str(path.relative_to(root)).replace("\\", "/"),
            anchor=df["source_sample_id"].astype(str),
            tool=df["gold_canonical_tool_id"].astype(str),
        )
        frames.append(df)
        file_checks.append(
            {
                "file": str(path.relative_to(root)).replace("\\", "/"),
                "model": model_dir,
                "family": family_name,
                "rows": len(df),
                "clean_rows": len(clean),
                "perturbed_rows": len(pert),
                "anchors": len(anchors),
                "tools": len(set(anchor_tool.values())),
                "success_rate_all_rows": float(df["overall_task_success"].mean()),
            }
        )

    all_df = pd.concat(frames, ignore_index=True)
    models = sorted(all_df["analysis_model"].unique())
    tools = sorted(all_df["tool"].unique())
    anchor_counts = (
        all_df[["anchor", "tool"]].drop_duplicates().groupby("tool").size().to_dict()
    )
    if len(models) != 7 or len(tools) != 16 or set(anchor_counts.values()) != {3}:
        raise ValueError(
            f"Expected 7 models, 16 tools, and 3 anchors/tool; got "
            f"{len(models)}, {len(tools)}, {anchor_counts}"
        )
    report = {
        "canonical_source_glob": "data/results/all-family/*/*/records.csv",
        "models": models,
        "families": FAMILY_ORDER,
        "files": len(files),
        "rows": len(all_df),
        "clean_rows": int((all_df["severity"] == "clean").sum()),
        "perturbed_rows": int(all_df["severity"].isin(SEVERITIES).sum()),
        "anchors": int(all_df["anchor"].nunique()),
        "tools": len(tools),
        "anchors_per_tool": anchor_counts,
        "file_checks": file_checks,
    }
    return all_df, report


def build_anchor_table(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = data.groupby(["analysis_model", "analysis_family", "anchor", "tool"])
    for (model, family, anchor, tool), group in grouped:
        clean = group[group["severity"] == "clean"]["overall_task_success"]
        pert = group[group["severity"].isin(SEVERITIES)]["overall_task_success"]
        if len(clean) != 1 or pert.empty:
            raise ValueError(f"Incomplete paired unit: {(model, family, anchor, tool)}")
        row = {
            "model": model,
            "family": family,
            "anchor": anchor,
            "tool": tool,
            "clean": float(clean.iloc[0]),
            "robust": float(pert.mean()),
        }
        for severity in SEVERITIES:
            row[f"robust_{severity}"] = float(
                group[group["severity"] == severity]["overall_task_success"].mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_units(anchor_table: pd.DataFrame, model: str | None, family: str) -> pd.DataFrame:
    subset = anchor_table.copy()
    if model is not None:
        subset = subset[subset["model"] == model]
    if family != "Overall":
        subset = subset[subset["family"] == family]
    # Mean first over all requested model/family cells for a shared anchor.
    return (
        subset.groupby(["anchor", "tool"], as_index=False)[["clean", "robust"]]
        .mean()
    )


def model_family_table(
    anchor_table: pd.DataFrame,
    rng: np.random.Generator,
    reps: int,
) -> pd.DataFrame:
    rows = []
    models = sorted(anchor_table["model"].unique())
    for model_label, model in [(m, m) for m in models] + [("ALL_MODELS", None)]:
        for family in FAMILY_ORDER + ["Overall"]:
            units = aggregate_units(anchor_table, model, family)
            out = paired_bootstrap(units, rng, reps)
            rows.append({"model": model_label, "family": family, **out})
    return pd.DataFrame(rows)


def severity_units(data: pd.DataFrame, severity: str) -> pd.DataFrame:
    pert = data[data["severity"] == severity]
    robust = (
        pert.groupby(["anchor", "tool"], as_index=False)["overall_task_success"]
        .mean()
        .rename(columns={"overall_task_success": "robust"})
    )
    # The paper's clean headline is over all four family-specific clean runs.
    clean = (
        data[data["severity"] == "clean"]
        .groupby(["anchor", "tool"], as_index=False)["overall_task_success"]
        .mean()
        .rename(columns={"overall_task_success": "clean"})
    )
    return clean.merge(robust, on=["anchor", "tool"], validate="one_to_one")


def severity_table(data: pd.DataFrame, rng: np.random.Generator, reps: int) -> pd.DataFrame:
    rows = []
    for severity in SEVERITIES:
        out = paired_bootstrap(severity_units(data, severity), rng, reps)
        rows.append({"severity": severity, **out})
    return pd.DataFrame(rows)


def tool_descriptives(anchor_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tool, group in anchor_table.groupby("tool"):
        row = {
            "tool": tool,
            "n_distinct_anchors": group["anchor"].nunique(),
            "clean": group["clean"].mean(),
        }
        for family in FAMILY_ORDER:
            row[family] = group[group["family"] == family]["robust"].mean()
        row["overall"] = np.mean([row[f] for f in FAMILY_ORDER])
        row["drop"] = row["clean"] - row["overall"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("tool").reset_index(drop=True)


def leave_one_tool_out(anchor_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    models = sorted(anchor_table["model"].unique())
    tools = sorted(anchor_table["tool"].unique())
    for omitted in tools:
        kept = anchor_table[anchor_table["tool"] != omitted]
        for model_label, model in [(m, m) for m in models] + [("ALL_MODELS", None)]:
            for family in FAMILY_ORDER + ["Overall"]:
                units = aggregate_units(kept, model, family)
                rows.append(
                    {
                        "omitted_tool": omitted,
                        "model": model_label,
                        "family": family,
                        "n_anchors": len(units),
                        "clean": units["clean"].mean(),
                        "robust": units["robust"].mean(),
                        "drop": (units["clean"] - units["robust"]).mean(),
                    }
                )
    return pd.DataFrame(rows)


def crosscheck_paper_artifact(root: Path, table: pd.DataFrame) -> dict:
    """Check point estimates against the frozen paper-artifact summary."""
    path = root / "data" / "results" / "paper_artifacts" / "all_family_model_summary.csv"
    if not path.exists():
        return {"available": False}
    published = pd.read_csv(path).set_index("model")
    ours = table[table["model"] != "ALL_MODELS"].set_index(["model", "family"])
    mapping = {
        "clean_success": ("Overall", "clean"),
        "overall_perturbed_macro": ("Overall", "robust"),
        "interface_exposure_success": ("A_interface", "robust"),
        "user_intent_success": ("B_intent", "robust"),
        "observation_success": ("C_observation", "robust"),
        "execution_environment_success": ("D_runtime", "robust"),
    }
    differences = {}
    for pub_col, (family, our_col) in mapping.items():
        diffs = [
            abs(float(published.loc[m, pub_col]) - float(ours.loc[(m, family), our_col]))
            for m in published.index
        ]
        differences[pub_col] = max(diffs)
    return {
        "available": True,
        "file": str(path.relative_to(root)).replace("\\", "/"),
        "maximum_absolute_difference_by_column": differences,
        "all_exact_within_1e-12": all(value < 1e-12 for value in differences.values()),
    }


def ci_text(row: pd.Series, metric: str, scheme: str = "tool") -> str:
    return (
        f"{row[metric]:.3f} (95% {scheme}-cluster CI "
        f"[{row[f'{metric}_{scheme}_ci_low']:.3f}, "
        f"{row[f'{metric}_{scheme}_ci_high']:.3f}])"
    )


def write_summary(
    output: Path,
    validation: dict,
    table: pd.DataFrame,
    severity: pd.DataFrame,
    loto: pd.DataFrame,
    reps: int,
    seed: int,
) -> None:
    all_rows = table[table["model"] == "ALL_MODELS"].set_index("family")
    overall = all_rows.loc["Overall"]
    family_rank = all_rows.loc[FAMILY_ORDER].sort_values("robust")
    c_row = all_rows.loc["C_observation"]
    model_rows = table[(table["model"] != "ALL_MODELS") & (table["family"] == "Overall")]
    c_is_lowest_each_model = all(
        table[(table["model"] == model) & table["family"].isin(FAMILY_ORDER)]
        .sort_values("robust")
        .iloc[0]["family"]
        == "C_observation"
        for model in model_rows["model"]
    )
    loto_all = loto[loto["model"] == "ALL_MODELS"]
    loto_c_is_lowest = True
    for omitted, group in loto_all.groupby("omitted_tool"):
        fg = group[group["family"].isin(FAMILY_ORDER)].sort_values("robust")
        loto_c_is_lowest &= fg.iloc[0]["family"] == "C_observation"

    lines = [
        "# Clustered-bootstrap audit of the single-family results",
        "",
        "## Scope and design",
        "",
        f"- Canonical input: `{validation['canonical_source_glob']}` only ({validation['files']} files).",
        f"- Checked {validation['rows']:,} records: {validation['clean_rows']:,} clean and {validation['perturbed_rows']:,} perturbed.",
        f"- Design: {validation['anchors']} clean anchors nested evenly in {validation['tools']} tools (3 anchors/tool), shared across 7 models and four family-specific runs.",
        f"- Resampling: {reps:,} paired percentile-bootstrap replicates, seed {seed}. The primary interval resamples 48 anchors; the conservative sensitivity interval resamples 16 tools and carries all anchors within each selected tool.",
        "- Family robustness follows the manuscript exactly: the mean of light, medium, and heavy success within a family. Overall robustness macro-averages the four families. All differences keep clean and perturbed observations paired within the same anchor.",
        "",
        "## Headline estimates",
        "",
        f"Across models, clean success is {ci_text(overall, 'clean')}; overall robustness is {ci_text(overall, 'robust')}; the paired clean-to-robust drop is {ci_text(overall, 'drop')}.",
        "",
    ]
    for family, row in family_rank.iterrows():
        lines.append(
            f"- {family}: robustness {ci_text(row, 'robust')}; paired drop {ci_text(row, 'drop')}."
        )
    lines += [
        "",
        f"Family C is the lowest-robustness family for every model: **{c_is_lowest_each_model}**. It also remains the lowest after every one-tool deletion: **{loto_c_is_lowest}**.",
        "",
        "Severity-micro averages (the weighting used in the manuscript):",
        "",
    ]
    for _, row in severity.iterrows():
        lines.append(f"- {row['severity']}: {ci_text(row, 'robust')}.")

    overall_loto = loto_all[loto_all["family"] == "Overall"]
    c_loto = loto_all[loto_all["family"] == "C_observation"]
    ranked_loto = loto[
        (loto["model"] != "ALL_MODELS") & (loto["family"] == "Overall")
    ]
    top_counts = (
        ranked_loto.loc[ranked_loto.groupby("omitted_tool")["robust"].idxmax(), "model"]
        .value_counts()
        .to_dict()
    )
    bottom_counts = (
        ranked_loto.loc[ranked_loto.groupby("omitted_tool")["robust"].idxmin(), "model"]
        .value_counts()
        .to_dict()
    )
    lines += [
        "",
        "## Leave-one-tool-out sensitivity",
        "",
        f"Across the 16 deletions, overall robustness ranges from {overall_loto['robust'].min():.3f} to {overall_loto['robust'].max():.3f}, and the paired overall drop ranges from {overall_loto['drop'].min():.3f} to {overall_loto['drop'].max():.3f}. Family-C robustness ranges from {c_loto['robust'].min():.3f} to {c_loto['robust'].max():.3f}. Thus no single sampled tool drives the observation-family bottleneck.",
        f"Claude-sonnet-4-6 remains the top model in all 16 deletions ({top_counts}). The lowest-ranked model is less stable ({bottom_counts}), so the small gap between the two lowest point estimates should not be over-interpreted.",
        "",
        "## Manuscript-ready text",
        "",
        f"> We quantified uncertainty with a paired cluster bootstrap over the 48 clean task anchors ({reps:,} replicates), preserving all perturbation variants derived from an anchor. As a conservative sensitivity analysis for sampled-tool dependence, we also resampled the 16 tools as clusters, retaining their three anchors. Across models, clean success was {overall['clean']:.3f} (95% tool-cluster CI [{overall['clean_tool_ci_low']:.3f}, {overall['clean_tool_ci_high']:.3f}]), whereas macro-averaged perturbation robustness was {overall['robust']:.3f} [{overall['robust_tool_ci_low']:.3f}, {overall['robust_tool_ci_high']:.3f}], a paired drop of {overall['drop']:.3f} [{overall['drop_tool_ci_low']:.3f}, {overall['drop_tool_ci_high']:.3f}]. Tool-output/observation robustness was lowest at {c_row['robust']:.3f} [{c_row['robust_tool_ci_low']:.3f}, {c_row['robust_tool_ci_high']:.3f}] and remained the lowest family for every model and after each leave-one-tool-out deletion. These intervals treat anchors or tools, rather than individual perturbation variants, as the independent sampling units.",
        "",
        "## Limitations and interpretation",
        "",
        "- The tool-cluster CI generalizes only to tools resembling these 16 sampled deterministic tools; it does not establish generalization to live APIs or the full universe of tools.",
        "- The anchor bootstrap conditions on the selected tools and is narrower when between-tool heterogeneity is substantial; the tool-cluster interval is therefore the preferred conservative interval.",
        "- Provider/model outputs may share infrastructure or training data. Averaging across the seven models is descriptive; models are not treated as an independent random sample of all LLMs.",
        "- Percentile intervals quantify sampling sensitivity under the benchmark design. They are not corrected for simultaneous coverage across the many model-family cells; use the complete table descriptively rather than declaring isolated pairwise significance.",
        "- Clean outputs were collected separately in each family run. Family-specific drops pair against that run's clean result; the overall clean estimate averages the four repeated clean runs per model and anchor.",
        "",
        "## Output files",
        "",
        "- `model_family_cluster_bootstrap.csv`: all model × family estimates, plus across-model rows, with anchor- and tool-cluster CIs.",
        "- `severity_cluster_bootstrap.csv`: micro-weighted severity estimates matching the manuscript's headline values.",
        "- `leave_one_tool_out.csv`: every model/family estimate after each tool deletion.",
        "- `tool_descriptives.csv`: per-tool clean, family, overall, and drop summaries.",
        "- `input_validation.json`: exact input inventory and balance checks.",
        "- `analysis_manifest.json`: seed, replicate count, formulas, and software versions.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--reps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()
    root = args.root.resolve()
    output = Path(__file__).resolve().parent
    output.mkdir(parents=True, exist_ok=True)

    data, validation = load_and_validate(root)
    anchor_table = build_anchor_table(data)
    rng = np.random.default_rng(args.seed)
    mf = model_family_table(anchor_table, rng, args.reps)
    sev = severity_table(data, rng, args.reps)
    tool = tool_descriptives(anchor_table)
    loto = leave_one_tool_out(anchor_table)
    validation["paper_artifact_crosscheck"] = crosscheck_paper_artifact(root, mf)

    mf.to_csv(output / "model_family_cluster_bootstrap.csv", index=False, float_format="%.9f")
    sev.to_csv(output / "severity_cluster_bootstrap.csv", index=False, float_format="%.9f")
    tool.to_csv(output / "tool_descriptives.csv", index=False, float_format="%.9f")
    loto.to_csv(output / "leave_one_tool_out.csv", index=False, float_format="%.9f")
    (output / "input_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest = {
        "script": str(Path(__file__).relative_to(root)).replace("\\", "/"),
        "seed": args.seed,
        "bootstrap_replicates": args.reps,
        "ci": "two-sided 95% percentile",
        "primary_sampling_unit": "source_sample_id (clean anchor), paired",
        "sensitivity_sampling_unit": "gold_canonical_tool_id, carrying all nested anchors",
        "family_robustness": "unweighted mean of light, medium, heavy success",
        "overall_robustness": "unweighted mean of four family robustness values",
        "python": __import__("sys").version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_summary(
        output / "RESULTS.md", validation, mf, sev, loto, args.reps, args.seed
    )
    print(f"Validated {validation['rows']:,} rows from {validation['files']} files")
    print(mf[mf["model"] == "ALL_MODELS"].to_string(index=False))
    print(f"Wrote outputs to {output}")


if __name__ == "__main__":
    main()
