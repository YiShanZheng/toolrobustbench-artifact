# C2 machine-proxy and sensitivity analysis

## Primary conservative existence proxy

At least one of seven evaluated models exactly succeeds on **49/144 = 34.0%** of C2 instances (Wilson 95% CI 26.8–42.1). By severity: light 23/48 (47.9%), medium 25/48 (52.1%), heavy 1/48 (2.1%).

This is a **lower-bound existence witness**, not an estimate of human recoverability: failure by all seven models does not imply unsolvability, and success by one model does not show that a typical human can recover the target.

## Robustness checks produced

- `threshold_sensitivity.csv`: requires ≥1, ≥2, ≥3, ≥4, or all 7 models to succeed.
- `leave_one_model_out.csv`: quantifies dependence on any single model.
- `cluster_bootstrap.csv`: resamples the 48 clean-anchor clusters, preserving the three repeated severities.
- `paired_severity_mcnemar.csv`: paired exact McNemar tests across severity on the same anchors.
- `proxy_by_tool.csv`: identifies tool strata that drive the aggregate.
- `per_model_c2.csv`: model-specific rates with Wilson intervals.

## Reporting rule

Keep these machine proxies separate from the blinded human audit. Do not replace missing human labels with model-derived labels, and do not call the any-model proxy “provably solvable” without the narrower phrase “witnessed solvable for at least one evaluated model under this scorer.”
