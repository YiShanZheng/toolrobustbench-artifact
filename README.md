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
docs/custom.bib
```

Figures referenced by the paper are stored in `picture/`.

## Anonymity Note

This repository is prepared for anonymous review. It removes local caches, backup folders, private credentials, and author-identifying metadata. The public, non-anonymous repository will be released after review.

## License

This artifact is released under the MIT License. During double-blind review, the copyright holder is anonymized as "Anonymous Authors".
