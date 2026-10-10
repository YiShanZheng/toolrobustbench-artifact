# Structural source-stage repair audit

`experiment_counterfactual_attribution.py` audits 98 deterministically sampled failed records. It replaces the atomic flag associated with the recorded primary failure and inspects the remaining recorded failures.

- Tier 1: replacing the primary flag leaves no failing flag.
- Tier 2: every remaining failing flag is at a later stage than the attributed source; the script then applies the documented post-execution checks.
- Tier 3: independent or same-stage failures remain.

Tier 2 is a structural downstream-only category. It is deliberately reported separately from the raw `is_cascade_failure` field; the two must not be treated as synonyms.

Run from the repository root:

```powershell
python scripts/rebuttal/experiment_counterfactual_attribution.py
```

Released outputs are in `scripts/rebuttal/output/`.
