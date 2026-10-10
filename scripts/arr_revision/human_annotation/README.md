# C2 blinded reconstruction audit

This directory contains the public and completed materials for the C2 returned-evidence reconstruction audit reported in the paper.

## Released files

- `primary_items_public.jsonl`: 60 model-visible prompts, covering 20 paired source anchors at light, medium, and heavy severity.
- `sampling_manifest_public.json`: fixed seed and selected source-anchor identifiers.
- `annotator_A_completed.csv` and `annotator_B_completed.csv`: independently completed sheets. Neither sheet contains the gold target.
- `aggregated_20261002/`: deterministic exact-reconstruction scores, severity summaries, agreement outputs, and the source-anchor cluster bootstrap.
- `machine_proxy/`: seven-model existence-witness and sensitivity analyses.
- `COMPLETED_ANNOTATION_AUDIT.md`: integrity checks and interpretation of the objective and subjective annotations.
- `private_gold_commitment.json`: cryptographic commitments to the frozen, withheld gold files.

The two annotators each had two years of experience in LLM tool-use evaluation and data annotation. They worked independently from randomized sheets and had no access to the private gold targets.

The primary paper outcome is exact JSON reconstruction. The subjective `FULL` / `PARTIAL` / `NONE` labels are retained for transparency but are not used as the principal recoverability claim.

## Reproduction boundary during anonymous review

The completed public sheets and all derived outputs are released. The private gold targets are withheld from the anonymous artifact to keep benchmark answers separate from public prompts. Their SHA-256 commitments, byte sizes, and row counts fix the audited files. The aggregation script documents the complete deterministic scoring procedure and can be rerun once the frozen gold key is released after review.

The machine existence-witness analysis can be rerun directly from the released benchmark records:

```powershell
python scripts/arr_revision/human_annotation/analyze_c2_proxy.py
```
