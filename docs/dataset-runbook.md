# Dataset Construction Runbook (Section 5)

This runbook implements Phase 1 dataset construction from `fine-tuning-doc.md` section 5.

## Output Contract

Each JSONL row must contain:

- `messages[0]`: `system` role with frozen system prompt
- `messages[1]`: `user` role with `<context>` and `Question:`
- `messages[2]`: `assistant` role containing valid `AnalyzeResponse` JSON

## Required Categories

The builder enforces all section 5.2 categories:

- `normal_grounded_analysis`
- `high_uncertainty_scenario`
- `missing_partial_data`
- `bullish_bias_neutral_output`
- `bearish_bias_neutral_output`
- `conflicting_data_uncertainty_escalation`

## One-Command Build (Hybrid)

Default (synthetic only):

`build-ft-dataset --count-per-category 50 --seed 1337 --train-ratio 0.88 --output-dir data/ft`

With curated traces:

`build-ft-dataset --count-per-category 50 --seed 1337 --train-ratio 0.88 --curated-input data/curated/traces.jsonl --output-dir data/ft`

## Produced Artifacts

- `data/ft/synthetic.jsonl`
- `data/ft/curated.jsonl`
- `data/ft/merged.jsonl`
- `data/ft/train.jsonl`
- `data/ft/eval.jsonl`
- `data/ft/coverage_report.json`
- `data/ft/manifest.json`

## Acceptance Gates

- Assistant payload must validate against `AnalyzeResponse`
- Disclaimer must match exact frozen text
- Upstream hard-fail samples are rejected
- Stratified split must remain in 85-90% train / 10-15% eval
- Eval must include tags:
  - `adversarial_prompt`
  - `compliance_edge_case`
  - `schema_stress_case`
