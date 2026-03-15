# Phase 4 Deployment Runbook

This runbook deploys the Phase-3-passed route and supports instant rollback.

## 1) Choose provider

Set one provider:

- Together: `LLM_PROVIDER=together`
- OpenRouter: `LLM_PROVIDER=openrouter`
- Hugging Face endpoints: `LLM_PROVIDER=hf_endpoint`

Model version label stays:

- `MODEL_VERSION=llama3-8b-fin-lora-v3`

## 2) Set deployment environment

Common:

- `LORA_ENABLED=true`
- `LLM_TEMPERATURE=0.2`
- `LLM_MAX_TOKENS=800`

Together-specific:

- `TOGETHER_API_KEY=<your_key>`
- `TOGETHER_BASE_URL=https://api.together.xyz/v1`
- `TOGETHER_MODEL_BASE=meta-llama/Meta-Llama-3-8B-Instruct`
- `TOGETHER_MODEL_LORA=llama3-8b-fin-lora-v3`

OpenRouter-specific:

- `OPENROUTER_API_KEY=<your_key>`
- `OPENROUTER_BASE_URL=https://openrouter.ai/api/v1`
- `OPENROUTER_MODEL_BASE=meta-llama/Meta-Llama-3-8B-Instruct`
- `OPENROUTER_MODEL_LORA=llama3-8b-fin-lora-v3`

Hugging Face endpoint-specific:

- `HF_API_TOKEN=<your_hf_token>`
- `HF_BASE_ENDPOINT_URL=<base_endpoint_url>`
- `HF_LORA_ENDPOINT_URL=<ft_endpoint_url>`

## 3) Start API

```bash
uvicorn backend.main:app --reload
```

## 4) Smoke checks

Health:

```bash
curl -s http://localhost:8000/health
```

Analyze:

```bash
curl -s -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <YOUR_API_KEY>" \
  -d '{
    "ticker": "AAPL",
    "question": "Is this a good investment over the next 12 months?",
    "time_horizon": "12m"
  }'
```

Side-by-side comparison:

1. Set `LORA_ENABLED=true` and call `/analyze`
2. Set `LORA_ENABLED=false` and call `/analyze` again with the same payload
3. Save both JSON responses and compare summary, uncertainty, and source discipline

Expected:

- Valid `AnalyzeResponse` contract.
- Required sources present.
- Disclaimer exact-match contract text.

## 5) Rollback (instant)

Disable LoRA route:

```bash
export LORA_ENABLED=false
```

Reload/restart the API process. Inference switches to the provider's base model route or `HF_BASE_ENDPOINT_URL`.

Roll forward again:

```bash
export LORA_ENABLED=true
```

## 6) Operational checks

- Watch analyze error rate and schema violations.
- Keep `MODEL_VERSION` in deployment config and release notes.
- If anomalies rise, rollback immediately with `LORA_ENABLED=false`.
- For Hugging Face endpoint deploys, verify both endpoint URLs stay healthy before rollout.
