# Clustered-bootstrap audit of the single-family results

## Scope and design

- Canonical input: `data/results/all-family/*/*/records.csv` only (28 files).
- Checked 15,456 records: 1,344 clean and 14,112 perturbed.
- Design: 48 clean anchors nested evenly in 16 tools (3 anchors/tool), shared across 7 models and four family-specific runs.
- Resampling: 20,000 paired percentile-bootstrap replicates, seed 20261002. The primary interval resamples 48 anchors; the conservative sensitivity interval resamples 16 tools and carries all anchors within each selected tool.
- Family robustness follows the manuscript exactly: the mean of light, medium, and heavy success within a family. Overall robustness macro-averages the four families. All differences keep clean and perturbed observations paired within the same anchor.

## Headline estimates

Across models, clean success is 0.979 (95% tool-cluster CI [0.963, 0.991]); overall robustness is 0.708 (95% tool-cluster CI [0.685, 0.732]); the paired clean-to-robust drop is 0.271 (95% tool-cluster CI [0.250, 0.291]).

- C_observation: robustness 0.455 (95% tool-cluster CI [0.425, 0.486]); paired drop 0.500 (95% tool-cluster CI [0.470, 0.531]).
- D_runtime: robustness 0.688 (95% tool-cluster CI [0.637, 0.730]); paired drop 0.291 (95% tool-cluster CI [0.258, 0.332]).
- B_intent: robustness 0.773 (95% tool-cluster CI [0.704, 0.839]); paired drop 0.216 (95% tool-cluster CI [0.150, 0.284]).
- A_interface: robustness 0.918 (95% tool-cluster CI [0.880, 0.953]); paired drop 0.076 (95% tool-cluster CI [0.042, 0.114]).

Family C is the lowest-robustness family for every model: **True**. It also remains the lowest after every one-tool deletion: **True**.

Severity-micro averages (the weighting used in the manuscript):

- light: 0.869 (95% tool-cluster CI [0.857, 0.880]).
- medium: 0.724 (95% tool-cluster CI [0.698, 0.752]).
- heavy: 0.491 (95% tool-cluster CI [0.446, 0.537]).

## Leave-one-tool-out sensitivity

Across the 16 deletions, overall robustness ranges from 0.703 to 0.713, and the paired overall drop ranges from 0.267 to 0.276. Family-C robustness ranges from 0.446 to 0.462. Thus no single sampled tool drives the observation-family bottleneck.
Claude-sonnet-4-6 remains the top model in all 16 deletions ({'claude-sonnet-4-6': 16}). The lowest-ranked model is less stable ({'gemini-2.5-flash': 14, 'deepseek-v4-flash': 2}), so the small gap between the two lowest point estimates should not be over-interpreted.

## Manuscript-ready text

> We quantified uncertainty with a paired cluster bootstrap over the 48 clean task anchors (20,000 replicates), preserving all perturbation variants derived from an anchor. As a conservative sensitivity analysis for sampled-tool dependence, we also resampled the 16 tools as clusters, retaining their three anchors. Across models, clean success was 0.979 (95% tool-cluster CI [0.963, 0.991]), whereas macro-averaged perturbation robustness was 0.708 [0.685, 0.732], a paired drop of 0.271 [0.250, 0.291]. Tool-output/observation robustness was lowest at 0.455 [0.425, 0.486] and remained the lowest family for every model and after each leave-one-tool-out deletion. These intervals treat anchors or tools, rather than individual perturbation variants, as the independent sampling units.

## Limitations and interpretation

- The tool-cluster CI generalizes only to tools resembling these 16 sampled deterministic tools; it does not establish generalization to live APIs or the full universe of tools.
- The anchor bootstrap conditions on the selected tools and is narrower when between-tool heterogeneity is substantial; the tool-cluster interval is therefore the preferred conservative interval.
- Provider/model outputs may share infrastructure or training data. Averaging across the seven models is descriptive; models are not treated as an independent random sample of all LLMs.
- Percentile intervals quantify sampling sensitivity under the benchmark design. They are not corrected for simultaneous coverage across the many model-family cells; use the complete table descriptively rather than declaring isolated pairwise significance.
- Clean outputs were collected separately in each family run. Family-specific drops pair against that run's clean result; the overall clean estimate averages the four repeated clean runs per model and anchor.

## Output files

- `model_family_cluster_bootstrap.csv`: all model × family estimates, plus across-model rows, with anchor- and tool-cluster CIs.
- `severity_cluster_bootstrap.csv`: micro-weighted severity estimates matching the manuscript's headline values.
- `leave_one_tool_out.csv`: every model/family estimate after each tool deletion.
- `tool_descriptives.csv`: per-tool clean, family, overall, and drop summaries.
- `input_validation.json`: exact input inventory and balance checks.
- `analysis_manifest.json`: seed, replicate count, formulas, and software versions.
