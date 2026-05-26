# Integrated Validation Release

This directory contains the integrated validation materials reported in the
paper appendix. The audit uses 140 stratified single-family records, with ten
records for each A1--D3 perturbation subtype.

## Released Materials

- `integrated_validation_annotation_human.csv`: completed manual audit labels.
- `integrated_validation_annotation_llm.csv`: blind LLM secondary annotation.
- `integrated_validation_sample_full.csv`: sampled evidence with scorer-facing
  fields needed for recomputation.
- `integrated_validation_llm_input.jsonl`: visible inputs supplied to the blind
  LLM annotator.
- `integrated_validation_metrics.csv`: aggregate agreement computations.
- `integrated_validation_metrics_by_family.csv`: family-level computations.
- `integrated_validation_table.tex`: generated LaTeX summary table.
- `confusion_*.tex`: generated human--scorer diagnostic tables.
- `llm_nonruntime_normalization_audit.csv`: normalization audit for
  `not_applicable` fields.
- `validation_sampling_report.md`: stratification and evidence report.
- `validation_missing_report.md`: annotation completeness report.

## Recompute

From the repository root:

```powershell
python scripts/compute_integrated_validation.py --validation-dir results/validation
```

`Valid n` is label-specific: agreement calculations exclude `unclear`
judgments and apply the prespecified `not_applicable` convention for
non-runtime decision fields.

Backup and provisional working files are intentionally excluded from this
anonymous release.
