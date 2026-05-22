# ToolRobust v3 Design Notes

## Orthogonality

- `family` chooses the perturbation source.
- `severity` chooses perturbation strength inside the chosen scope.
- `heavy` still defaults to same-family combinations.
- `summary_overall.csv` is current-scope overall only.
- Stable plot and table ordering comes from `run_manifest.json`.

## Static Subtype -> Stage Mapping

- `A1 Name Surface Shift` -> `S1 Tool Selection`
- `A2 Description Abstraction` -> `S2 Interface Grounding`
- `A3 Family Similarity Pressure` -> `S2 Interface Grounding`
- `B1 Indirectness` -> `S2 Interface Grounding`
- `B2 Underspecification` -> `S3 Parameter Binding`
- `B3 Distraction` -> `S1 Tool Selection`
- `B4 Conflict` -> `S1 Tool Selection`
- `C1/C2/C3/C4` -> `S4 Observation / Execution Handling`
- `D1/D2/D3` -> `S4 Observation / Execution Handling`

This stage map stays static. What changed is the interpretation of failures observed after the perturbation, not the subtype-to-stage assignment itself.

## Failure Interpretation

ToolRobust v3 now separates three concepts:

1. `observed_failure_tag`
2. `primary_failure_tag`
3. `cascade / spillover`

The observed tag is still one of the six benchmark labels. The new logic adds:

- `first_failure_stage`
- `primary_failure_tag`
- `is_cascade_failure`
- `cascade_from_stage`
- `expected_primary_match`
- `allowed_spillover_match`
- `boundary_violation_flag`
- `boundary_violation_reason`

The benchmark therefore distinguishes:

- a direct primary effect that matches the family/stage design intent
- a downstream spillover error that is still explainable from an earlier failure
- a true family boundary violation

## Family Failure Profiles

Every family has:

- `expected_primary_failures`
- `allowed_spillover_failures`
- `disallowed_direct_perturbations`

### Interface Exposure

- expected primary failures: `wrong-tool`
- allowed spillover failures: `missing-parameter`, `wrong-parameter`
- disallowed direct perturbations:
  - `parameter schema semantic change`
  - `parameter key rename`
  - `required/default/order modification`
  - `parameter-level grounding drift`

### User Intent

- expected primary failures: `no-call`, `wrong-tool`
- allowed spillover failures: `missing-parameter`, `wrong-parameter`
- disallowed direct perturbations:
  - `tool registry / schema level mutation`
  - `execution environment state mutation`

### Observation

- expected primary failures: `invalid-format`, `execution-fail`, `wrong-parameter`
- allowed spillover failures: `no-call`
- disallowed direct perturbations:
  - `user intent rewrite`
  - `tool interface exposure mutation`

### Execution Environment

- expected primary failures: `execution-fail`, `invalid-format`, `no-call`
- allowed spillover failures: `wrong-parameter`
- disallowed direct perturbations:
  - `user request semantics rewrite`
  - `interface exposure mutation`
  - `parameter schema semantic change`

## Stage Failure Profiles

### S1 Tool Selection

- expected primary failures: `wrong-tool`, `no-call`
- allowed spillover failures: `missing-parameter`, `wrong-parameter`
- explanation: downstream parameter issues may still be cascades from an early selection miss.

### S2 Interface Grounding

- expected primary failures: `wrong-tool`, `missing-parameter`, `wrong-parameter`
- allowed spillover failures: `no-call`
- explanation: grounding errors may manifest directly as wrong-tool or wrong-bound-argument behavior.

### S3 Parameter Binding

- expected primary failures: `missing-parameter`, `wrong-parameter`
- allowed spillover failures: `invalid-format`
- explanation: malformed calls can be spillover from unstable argument binding.

### S4 Observation / Execution Handling

- expected primary failures: `invalid-format`, `execution-fail`, `wrong-parameter`
- allowed spillover failures: `no-call`
- explanation: runtime instability can cause the model to abandon the call entirely.

### S5 End-to-End Task Success

- no new failure labels
- final rollup only

## Boundary Decision Rule

Family boundary evaluation now follows this order:

1. If the perturbation directly touched a disallowed direct perturbation surface, mark boundary violation.
2. Else if `primary_failure_tag` matches the family’s `expected_primary_failures`, boundary is normal.
3. Else if the observed failure is marked cascade-aware and matches the family’s `allowed_spillover_failures`, boundary is still normal.
4. Else mark boundary violation.

This is intentionally different from the removed heuristic:

- old: `wrong-parameter > 0 => Family A drift`
- new: `wrong-parameter` is only a violation if it is itself the unexplained primary effect or if the perturbation directly crossed a forbidden boundary

## Family A Guardrails

The perturbation builder still enforces Family A’s input-side boundary:

- no parameter key rename
- no required/default/order mutation
- no parameter-level semantic rewrite
- no `S3` / `S4` target operators

So the benchmark is stricter about perturbation construction and more realistic about output interpretation.

## Output Protocol

Core files retained:

- `summary_overall.csv`
- `failure_taxonomy.csv`
- `summary_by_family.csv`
- `summary_by_subtype.csv`
- `summary_by_stage.csv`
- `failure_by_family.csv`
- `run_manifest.json`

New diagnostics files:

- `primary_failure_breakdown.csv`
- `cascade_failure_breakdown.csv`
- `family_boundary_report.csv`
- `paired_case_dump.csv`
- `summary_by_operator.csv`

## Reporting Intent

`failure_taxonomy.csv` remains the main observed-tag table, but it now separates:

- `observed_failure_tag`
- `primary_failure_tag`
- `secondary_reason`
- `is_cascade_failure`

`family_boundary_report.csv` replaces the old drift report and explicitly separates:

- dominant primary failure
- dominant observed failure
- expected-primary match rate
- allowed-spillover match rate
- boundary-violation rate

`paired_case_dump.csv` is now the manual inspection table for clean/perturbed comparisons with full cascade context.
