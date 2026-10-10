# Statistical and mixed-family audits

This directory contains the scripts and released outputs used for the resubmission analyses.

## Single-family uncertainty

`bootstrap_single_family.py` validates the 15,456-record single-family inventory, runs 20,000 paired percentile-bootstrap replicates over 48 source anchors, runs a conservative tool-cluster bootstrap over 16 sampled tools, and computes leave-one-tool-out summaries. Its outputs include:

- `model_family_cluster_bootstrap.csv`
- `severity_cluster_bootstrap.csv`
- `leave_one_tool_out.csv`
- `tool_descriptives.csv`
- `input_validation.json`
- `analysis_manifest.json`

## Exactly matched mixed-family analysis

`matched_mixed_bootstrap.py` matches every mixed record to both isolated constituents by model, source anchor, severity, and subtype. It runs a 20,000-replicate source-anchor bootstrap and leave-one-anchor-out sensitivity analysis. The record-level matches, cell summaries, intervals, and plot are released alongside the script.

`analyze_positive_mixed_reversals.py` audits the two positive weaker-constituent contrasts with per-model cells, paired fail-to-success and success-to-fail transitions, source-anchor bootstrap intervals, and leave-one-anchor-out ranges.

Run all three scripts from the repository root:

```powershell
python scripts/arr_revision/statistics/bootstrap_single_family.py
python scripts/arr_revision/statistics/matched_mixed_bootstrap.py
python scripts/arr_revision/statistics/analyze_positive_mixed_reversals.py
```
