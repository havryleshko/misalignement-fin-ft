# Phase 3 Evaluation Gate Runbook

Run the mandatory Phase 3 gate by comparing:

- base model: `meta-llama/Meta-Llama-3-8B-Instruct`
- fine-tuned adapter: `havryleshko/ft-lora-v1`

on the same eval split.

## Prerequisites

- Hugging Face login with access to base + adapter (`hf auth login`)
- Environment installed (`pip install -e .`)
- Eval split present at `data/ft/eval.jsonl`

## Command

```bash
evaluate-ft-phase3-gate \
  --eval data/ft/eval.jsonl \
  --ft-adapter-id havryleshko/ft-lora-v1 \
  --output-dir artifacts/phase3-eval \
  --max-new-tokens 512
```

## Outputs

- `artifacts/phase3-eval/phase3_report.json`
- `artifacts/phase3-eval/phase3_samples.jsonl`
- `artifacts/phase3-eval/phase3_decision.md`

## Acceptance Checks

All must be true:

- `schema_validity_improved`
- `hallucination_reduced`
- `source_coverage_improved`
- `confident_wrong_not_increased`

If any fails, Phase 3 fails and dataset-fix + retrain is required before Phase 4.
