# Integrated Validation Sampling Report

## Source

- Single-family record files read: 28
- Eligible perturbed raw model-output records: 14,112
- Selected validation records: 140
- Cross-family records were located but intentionally excluded from this A1--D3 stratified sample.

## Selected Distribution

### Family

- `execution_environment`: 30
- `interface_exposure`: 30
- `observation`: 40
- `user_intent`: 40

### Subtype

- `A1 Name Surface Shift`: 10
- `A2 Description Abstraction`: 10
- `A3 Family Similarity Pressure`: 10
- `B1 Indirectness`: 10
- `B2 Underspecification`: 10
- `B3 Distraction`: 10
- `B4 Conflict`: 10
- `C1 Return Structure Noise`: 10
- `C2 Return Evidence Loss`: 10
- `C3 Return Evidence Corruption`: 10
- `C4 Return Evidence Competition`: 10
- `D1 Availability Failure`: 10
- `D2 Runtime Variability`: 10
- `D3 Failure Feedback Quality`: 10

### Severity

- `heavy`: 75
- `light`: 30
- `medium`: 35

### Outcome (used only for sampling coverage; hidden from annotator inputs)

- `failure`: 105
- `success`: 35

## Field Audit

- Required raw scorer, gold-target, and grouping fields were found.
- Raw records contain parsed tool calls and selected post-execution decision fields, not original completion text.
- For C/D records, public observation/runtime feedback is deterministically reconstructed from the task template, predicted call, and runtime perturbation implementation.
- For family C, the raw post-execution model recovery JSON is not persisted; annotations may use `unclear` when that missing response prevents a judgment.

## Warnings

- Observation records do not persist the raw post-execution model response.

## Source Files

- `data\results\all-family\claude-sonnet-4-6\execution_environment\records.csv`
- `data\results\all-family\claude-sonnet-4-6\interface_exposure\records.csv`
- `data\results\all-family\claude-sonnet-4-6\observation\records.csv`
- `data\results\all-family\claude-sonnet-4-6\user_intent\records.csv`
- `data\results\all-family\deepseek-v4-flash\execution_environment\records.csv`
- `data\results\all-family\deepseek-v4-flash\interface_exposure\records.csv`
- `data\results\all-family\deepseek-v4-flash\observation\records.csv`
- `data\results\all-family\deepseek-v4-flash\user_intent\records.csv`
- `data\results\all-family\deepseek-v4-pro\execution_environment\records.csv`
- `data\results\all-family\deepseek-v4-pro\interface_exposure\records.csv`
- `data\results\all-family\deepseek-v4-pro\observation\records.csv`
- `data\results\all-family\deepseek-v4-pro\user_intent\records.csv`
- `data\results\all-family\gemini-2.5-flash\execution_environment\records.csv`
- `data\results\all-family\gemini-2.5-flash\interface_exposure\records.csv`
- `data\results\all-family\gemini-2.5-flash\observation\records.csv`
- `data\results\all-family\gemini-2.5-flash\user_intent\records.csv`
- `data\results\all-family\gemini-2.5-pro\execution_environment\records.csv`
- `data\results\all-family\gemini-2.5-pro\interface_exposure\records.csv`
- `data\results\all-family\gemini-2.5-pro\observation\records.csv`
- `data\results\all-family\gemini-2.5-pro\user_intent\records.csv`
- `data\results\all-family\gpt-5.4-mini\execution_environment\records.csv`
- `data\results\all-family\gpt-5.4-mini\interface_exposure\records.csv`
- `data\results\all-family\gpt-5.4-mini\observation\records.csv`
- `data\results\all-family\gpt-5.4-mini\user_intent\records.csv`
- `data\results\all-family\qwen-plus\execution_environment\records.csv`
- `data\results\all-family\qwen-plus\interface_exposure\records.csv`
- `data\results\all-family\qwen-plus\observation\records.csv`
- `data\results\all-family\qwen-plus\user_intent\records.csv`
