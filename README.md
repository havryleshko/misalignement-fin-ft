# Misalignment Fit FT

API-first finance analysis engine that grounds outputs in market/filings data, quantifies uncertainty, flags prompt bias, and returns a strict machine-readable contract.

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

## Fine-Tuning Compatibility Note

`transformers`/`trl` APIs can differ by version. This repo currently uses:

- `TrainingArguments(eval_strategy=...)` (not `evaluation_strategy`)
- `SFTTrainer(processing_class=tokenizer)` (not `tokenizer=...`)

