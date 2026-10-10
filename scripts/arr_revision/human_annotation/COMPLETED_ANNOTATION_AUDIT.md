# Completed C2 Human-Annotation Audit

Date: 2026-10-02

## Decision

The completed annotations are usable for a paper claim about **blinded exact JSON reconstruction**. They are not sufficiently reliable for a main claim based on the subjective `FULL` / `PARTIAL` / `NONE` categories.

## Integrity and completion checks

- Both returned sheets contain all 60 expected unique item IDs.
- No issued `annotation_order`, `item_id`, `severity`, or `model_visible_prompt` field was changed.
- Prompt SHA-256 values match the private manifest for all 120 annotator--item records.
- Labels, confidence scores, evidence citations, notes, and time fields are complete.
- All required reconstructed-answer values are valid JSON.
- Annotator A recorded 37.7 total minutes; annotator B recorded 35.1 total minutes.

## Primary result: exact reconstruction

| Metric | Result |
|---|---:|
| Annotator A exact | 14/60 = 23.3% (Wilson 95% CI 14.4--35.4) |
| Annotator B exact | 19/60 = 31.7% (Wilson 95% CI 21.3--44.2) |
| Pooled annotator--item trials | 33/120 = 27.5% |
| Pooled 20-anchor cluster-bootstrap CI | 15.0--40.8% (20,000 resamples; seed 20261002) |
| Binary exact/non-exact agreement | 55/60 = 91.7% |
| Binary Cohen's kappa | 0.793 |
| Exact-status disagreements | 5/60, all at medium severity |
| Exact McNemar test | two-sided exact p = 0.0625 |

Pooled exact reconstruction by severity is 14/40 (35.0%) for light, 19/40 (47.5%) for medium, and 0/40 (0.0%) for heavy prompts. The light--medium reversal is consistent with the benchmark's operator-specific evidence patterns: some medium prompts retain a literal token or approximate value even while marking it incomplete, whereas some light prompts drop a field entirely. It must not be described as a monotonic human-difficulty curve.

## Secondary result: subjective categories

Raw three-way label agreement is 34/60 (56.7%), with Cohen's kappa 0.293. Annotator A used 15 `FULL`, 45 `PARTIAL`, and no `NONE` labels; annotator B used 36 `FULL`, 19 `PARTIAL`, and 5 `NONE` labels. Annotator B marked 17 answers `FULL` even though deterministic comparison showed only partial target coverage. This indicates a threshold mismatch about whether a visibly incomplete field can still count as fully recoverable.

Consequently:

- Do not report a consensus `FULL` rate from these raw labels.
- Use deterministic exact JSON equality as the primary outcome.
- If an item-level consensus label is required, obtain a third independent, gold-blinded annotation for the five exact-status disagreements and clarify the treatment of `partial` tokens and `approx` values before annotation.

## Reproducible outputs

- `aggregated_20261002/summary.json`
- `aggregated_20261002/rates_by_severity.csv`
- `aggregated_20261002/objective_exact_cluster_bootstrap.csv`
- `aggregated_20261002/objective_exact_disagreements.csv`
- `aggregated_20261002/paired_scored_annotations.csv`
- `aggregated_20261002/adjudication_queue.csv`

The manuscript uses only the exact-reconstruction results and explicitly discloses the low agreement of the subjective categories.
