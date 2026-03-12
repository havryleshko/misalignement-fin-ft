# Misalignment Fit FT

API-first finance analysis engine that grounds outputs in market/filings data, quantifies uncertainty, flags prompt bias, and returns a strict machine-readable contract.

## What It Does

Given a ticker + question, the system produces a structured probabilistic analysis (expected return, confidence interval, scenario tree, risk flags, and bias notice) using a fail-closed schema.

## Problem It Solves

Most LLM investment answers are inconsistent, overconfident, and hard to validate in production. This project enforces a strict JSON contract and explicit uncertainty handling so outputs are safer to automate, evaluate, and monitor.

## Current Dataset Snapshot

From `data/ft/manifest.json` / `data/ft/coverage_report.json`:

- Total rows: `2015`
- Source mix: `1200` synthetic + `815` curated traces
- Train/Eval split: `1773` / `242`
- Categories covered: `8` (none missing)
- Required eval tags present: `adversarial_prompt`, `compliance_edge_case`, `schema_stress_case`

## Quickstart

### 1) Prepare environment

Create `.env` from `.env.example` and set required values:

- `ALPHAVANTAGE_API_KEY`
- `DATABASE_URL`
- `REDIS_URL`
- `API_KEY_HASH_SALT`

### 2) Run local dependencies

```bash
docker compose -f infra/docker/docker-compose.yml up -d postgres redis
```

### 3) Run the API

```bash
uvicorn backend.main:app --reload
```

### 4) Seed an API key

```bash
python -m backend.scripts.seed_api_key
```

### 5) Call `/analyze`

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <YOUR_API_KEY>" \
  -d '{
    "ticker": "AAPL",
    "question": "Is this a good investment over the next 12 months?",
    "time_horizon": "12m"
  }'
```

## API Contract

Request:

- `ticker`: uppercase symbol (`AAPL`, `BRK.B`)
- `question`: user question
- `time_horizon`: duration (`30d`, `12m`, `2y`)

Response (mandatory fields):

- `summary`
- `expected_return`
- `confidence_interval`
- `probability_positive`
- `scenarios` (`bull`, `base`, `bear`)
- `risk_flags`
- `bias_notice`
- `sources`
- `disclaimer`

If any mandatory field cannot be produced or fails validation, the API returns an error (no partial payload).

## Deployment Mode (Phase 4)

- Providers supported: Together AI (`LLM_PROVIDER=together`) and OpenRouter (`LLM_PROVIDER=openrouter`)
- Active deployed version label: `MODEL_VERSION=llama3-8b-fin-lora-v3`
- Rollback switch: `LORA_ENABLED=false` (routes to base model path)

See `docs/phase4-deployment-runbook.md` for provider env setup, rollout, rollback, and smoke tests.

