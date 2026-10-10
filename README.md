# ToolRobustBench

ToolRobustBench is a stage-aligned benchmark for diagnosing robustness failures in single-step tool calling. It evaluates whether tool-using models remain reliable under perturbations to the tool interface, user intent, tool-output observations, and runtime feedback.

This anonymous artifact contains the code, configurations, deterministic task data, raw results used by the paper, and plotting assets needed to inspect or reproduce the main analyses.

## What Is Included

- `src/`: benchmark implementation and CLI.
- `configs/`: benchmark configurations, including paper-scale single-family and cross-family runs.
- `scripts/`: scripts for paper runs, rescoring, probing, and artifact refresh.
- `tests/`: regression and smoke tests.
- `data/generated/`: deterministic task files and perturbation calibration output.
- `data/results/all-family/`: raw records and summaries for the paper's single-family experiments.
- `data/results/cross-family/`: raw records and summaries for the representative cross-family experiments.
- `results/validation/`: released integrated validation audit, blind LLM consistency check, and agreement outputs reported in the appendix.
- `scripts/arr_revision/statistics/`: clustered-bootstrap, leave-one-tool-out, exactly matched mixed-family, and positive-contrast analyses with their released outputs.
- `scripts/arr_revision/human_annotation/`: completed blinded C2 reconstruction sheets, public prompts, derived exact-recovery tables, and machine recoverability analyses.
- `scripts/rebuttal/output/`: released structural source-stage repair audit used by the paper.
- `picture/`: paper figures and appendix diagnostics.
- `docs/`: selected paper and benchmark documentation.

Large temporary runs, rollback backups, debug outputs, local caches, and private environment files are intentionally excluded.

## Installation

Use Python 3.10 or newer.

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -e .
```

If running on Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick Checks

Run the test suite:

```powershell
pytest
```

Run a small benchmark smoke test:

```powershell
$env:PYTHONPATH="src"
python -m toolrobust.cli.main --config configs/benchmark_v3.yaml run
```

## Paper-Scale Runs

The paper-scale single-family configuration is:

```text
configs/paper_all_family.yaml
```

The representative cross-family configuration is:

```text
configs/paper_cross_family.yaml
```

The corresponding helper scripts are:

```text
scripts/run_paper_all_family.ps1
scripts/run_paper_cross_family.ps1
```

These runs require access to the model providers configured locally by the user. The released artifact includes the raw records and summaries used in the paper, so reviewers can inspect the results without rerunning all model calls.

## Recomputing Paper Artifacts

The paper figures and tables were generated from:

```text
data/results/all-family/
data/results/cross-family/
```

To refresh paper artifacts after changing result files:

```powershell
$env:PYTHONPATH="src"
python scripts/refresh_paper_artifacts.py
```

Some scripts support optional provider-specific probes or compatibility checks. Those require local API credentials and are not needed for inspecting the released results.

## Integrated Validation Audit

The appendix reports an integrated validation audit over 140 stratified single-family records, with ten records from each A1--D3 subtype. The released validation package contains the completed manual annotation, the blind LLM secondary consistency check, the sampled visible record evidence, and the derived agreement outputs.

Released files include:

- `results/validation/integrated_validation_annotation_human.csv`: completed manual audit labels.
- `results/validation/integrated_validation_annotation_llm.csv`: blind LLM annotation labels.
- `results/validation/integrated_validation_sample_full.csv`: sampled record evidence and automatic scorer-facing fields.
- `results/validation/integrated_validation_llm_input.jsonl`: blind LLM annotation inputs.
- `results/validation/integrated_validation_metrics.csv`: agreement results used for appendix reporting.
- `results/validation/integrated_validation_metrics_by_family.csv`: family-specific diagnostic summaries.
- `results/validation/integrated_validation_table.tex`: generated LaTeX validation table.
- `results/validation/validation_sampling_report.md`: sampling and evidence-availability report.

Recompute agreement outputs from the released annotations:

```powershell
python scripts/compute_integrated_validation.py --validation-dir results/validation
```

The label-specific valid sample size excludes `unclear` judgments and applies the prespecified `not_applicable` convention to non-runtime decision fields. Rerunning the blind LLM secondary check is optional and requires locally supplied API credentials:

```powershell
python scripts/run_llm_validation.py --input results/validation/integrated_validation_llm_input.jsonl --output-dir results/validation/llm_rerun
```

## Resubmission Statistical Audits

The resubmission adds uncertainty and sensitivity analyses over the released raw records:

- 20,000-replicate paired anchor and tool-cluster bootstraps for single-family results;
- leave-one-tool-out sensitivity analysis;
- exactly matched mixed-versus-single comparisons with source-anchor bootstrap intervals;
- leave-one-anchor-out mixed-family sensitivity analysis; and
- record-level audits of the two positive weaker-constituent contrasts.

The scripts, fixed seeds, manifests, and generated tables are in `scripts/arr_revision/statistics/`. From the repository root, the main analyses can be rerun with:

```powershell
python scripts/arr_revision/statistics/bootstrap_single_family.py
python scripts/arr_revision/statistics/matched_mixed_bootstrap.py
python scripts/arr_revision/statistics/analyze_positive_mixed_reversals.py
```

## C2 Recoverability Audit

`scripts/arr_revision/human_annotation/` contains the completed independent reconstruction sheets from two blinded domain annotators, the 60 public prompts, the public sampling manifest, deterministic scoring outputs, and the machine existence-witness analysis. Both annotators had two years of experience in LLM tool-use evaluation and data annotation.

Private gold targets are intentionally not exposed in the anonymous review artifact. `private_gold_commitment.json` records SHA-256 commitments, sizes, and row counts for the frozen private files. This prevents post-hoc replacement while keeping the blinded targets separate from the public sheets. The released aggregate tables contain all paper-reported rates, confidence intervals, and agreement statistics.

The structural source-stage repair audit is released in `scripts/rebuttal/experiment_counterfactual_attribution.py` and `scripts/rebuttal/output/`. Its Tier 2 means that, after replacing the attributed source flag, only later-stage failure flags remain; it is not identical to the raw `is_cascade_failure` field. The output therefore reports Tier membership and recorded-cascade status separately.

## Result Files

Each result directory may contain:

- `records.csv`: instance-level model outputs and scorer labels.
- `summary_overall.csv`: condition-level summary.
- `summary_by_family.csv`: family and severity summary.
- `summary_by_subtype.csv`: subtype-level summary.
- `failure_taxonomy.csv`: observed and primary failure labels.
- `family_boundary_report.csv`: cascade and boundary diagnostics.
- `run_manifest.json`: run configuration and sampled tools.

Cross-family summaries additionally include selected composition comparisons against corresponding single-family baselines.

## Paper

The paper source is available in:

```text
docs/main.tex
docs/main.bib
```

Figures referenced by the paper are stored in `picture/`.

## Anonymity Note

This repository is prepared for anonymous review. It excludes local caches, backup/provisional annotation files, private credentials, and author-identifying data from the released artifact contents.

## License

This artifact is released under the MIT License. During double-blind review, the copyright holder is anonymized as "Anonymous Authors".
