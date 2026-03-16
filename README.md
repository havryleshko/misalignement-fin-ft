<h1 align="center">AI Misalignement Investment Engine</h1>

**API-first fine-tuned Llama 3 8B for separation of market sentiment and financials to get the most bias-neutral response. Trained on custom brokerage data from [eToro](https://www.etoro.com/).

## What It Does

Given a ticker + question, the system produces a structured probabilistic analysis (expected return, confidence interval, scenario tree, risk flags, and bias notice) using a fail-closed schema.

## Problem It Solves

LLM investment answers are inconsistent, overconfident, and hard to validate. Investors on brokerage platforms oftentimes make emotional decision due to a highly volatile market. This project enforces a strict JSON contract and explicit uncertainty handling so output gives you clear picture on what's actually going on with the stock.

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
