# AI Recommendation Misalignment Engine
## System Design Document (v1.0)

---

## 1. Purpose

Build an **API-first, finance-grounded AI engine** that mitigates LLM recommendation misalignment in investing by:

- Grounding all outputs in verifiable financial data
- Quantifying uncertainty and risk
- Detecting and correcting user bias
- Enforcing compliance-by-design (FINRA-aligned)

This system provides **decision support**, not financial advice or guarantees.

---

## 2. Problem Statement

Generic LLMs:
- Hallucinate financial facts
- Ignore real-time market data
- Amplify user confirmation bias
- Output deterministic recommendations without uncertainty

Consequences:
- Retail users lose 10–30% in volatile markets
- Platforms face reputational and regulatory risk

---

## 3. Target Users (ICP)

### Primary
- Trading platforms (eToro, Webull, Freetrade, Public)
- Robo-advisors

### Secondary
- Advanced retail investors
- Wealth managers
- Small hedge funds

### Buyer Persona
- CTO / Head of Product / Quant Lead
- Wants: safety, auditability, easy integration

---

## 4. Non-Goals

- No consumer UI
- No portfolio execution
- No trade automation
- No “stock picks” or guarantees

---

## 5. Core Principles

1. **Grounding over intelligence**
2. **Probabilistic outputs only**
3. **Bias-aware reasoning**
4. **Composable API-first design**
5. **Auditable & traceable responses**
6. **Minimal surface area (security & infra)**

---

## 6. High-Level Architecture

Client Platform
↓
API Gateway (Auth + Rate Limit)
↓
Query Orchestrator
├── Intent & Bias Detection
├── Data Retrieval Layer
├── LLM Inference (RAG)
├── Risk & Uncertainty Engine
└── Compliance Filter
↓
Structured JSON Response


---

## 7. Component Breakdown

### 7.1 API Gateway

**Responsibilities**
- API key authentication
- Rate limiting
- Usage logging
- Request validation

**Auth**
- Header-based API keys (`X-API-Key`)
- SHA-256 hashed storage
- Per-key revocation

---

### 7.2 Query Orchestrator

Central controller coordinating all steps.

**Flow**
1. Validate request
2. Detect user intent & bias
3. Fetch required data
4. Run model inference
5. Quantify uncertainty
6. Apply compliance rules
7. Return structured response

If any step fails → refuse to answer.

---

### 7.3 Intent & Bias Detection

Detects:
- Bullish / bearish framing
- Leading language
- Emotional prompts

**Outcome**
- Internal prompt neutralization
- Bias warning surfaced in output

---

### 7.4 Data Retrieval Layer (Hard Requirement)

**Sources**
- SEC EDGAR (10-K, 10-Q, 8-K)
- Market prices (Polygon.io / Alpha Vantage)
- Analyst consensus (when available)

**Rules**
- No inference without fresh data
- Cached with TTL
- All sources logged and cited

---

### 7.5 Model Layer

**Model**
- Llama 3 (or equivalent open-weight LLM)

**Strategy**
- Retrieval-Augmented Generation (RAG)
- Light finance-specific fine-tuning (Llama 3.2 3B quantized)
- Strict JSON schema enforcement

The model never “knows” — it reasons over retrieved facts.

---

### 7.6 Risk & Uncertainty Engine

Produces:
- Confidence intervals
- Probability-weighted scenarios
- Risk flags (volatility, drawdown, data gaps)

**Techniques**
- Monte Carlo simulations
- Scenario trees (bull / base / bear)

---

### 7.7 Compliance Layer

Enforces:
- No guarantees
- Explicit uncertainty language
- Source citation
- Machine-readable disclaimers

Outputs are compliant by construction.

---

## 8. API Design (MVP)

### Endpoint: `/analyze`

**Input**
```json
{
  "ticker": "AAPL",
  "question": "Is this a good investment over the next 12 months?",
  "time_horizon": "12m"
}
```

### Output contract - mandatory

```json
{
  "summary": "Apple shows moderate upside driven by earnings stability...",
  "expected_return": 8.2,
  "confidence_interval": [-12.4, 18.7],
  "probability_positive": 0.61,
  "scenarios": {
    "bull": 18.7,
    "base": 8.2,
    "bear": -12.4
  },
  "risk_flags": ["macro_uncertainty", "valuation_risk"],
  "bias_notice": "User prompt contained bullish framing",
  "sources": [
    "SEC 10-K 2024",
    "Polygon.io price feed"
  ],
  "disclaimer": "This output is probabilistic and not investment advice."
}
```

If any output field cannot be populated or fails schema validation, return an error (no partial payload).

9. Data Model (Minimal)
customers
id
name
plan
api_keys
id
key_hash
customer_id
rate_limit
status
usage_logs (optional initially)
api_key_id
endpoint
tokens_used
latency
timestamp

## repo structure

misalignment-fin-ft/
├── backend/
│   ├── api/
│   │   ├── routes.py
│   │   ├── schemas.py
│   │   └── auth.py
│   ├── orchestration/
│   ├── data/
│   ├── models/
│   ├── risk/
│   ├── compliance/
│   └── main.py
├── sdk/
│   ├── python/
│   └── ts/
├── infra/
│   └── docker/
│       ├── Dockerfile
│       └── docker-compose.yml
├── docs/
└── README.md

11. Infrastructure
Required (MVP)
Docker
Postgres
Redis (rate limiting)
Explicitly Excluded
Terraform
Kubernetes
Supabase Auth
Frontend UI

12. Security Model
HTTPS only
Hashed API keys
Rate limiting
Key revocation
No PII stored

13. Rollout Plan
Phase 1
/analyze endpoint
One data source
One model
One customer
Phase 2
SDKs
Scenario analysis
Usage-based billing
Phase 3
White-label integrations
Enterprise compliance extensions

14. Success Criteria
Platforms integrate in <1 day
Zero hallucinated outputs
Explicit uncertainty in 100% of responses
Positive feedback from fintech engineers

15. Kill Criteria
Platforms refuse to integrate
Users demand deterministic picks
Data costs exceed value
If any occur → pivot or shut down.



---

### Final hard advice
This doc is **already better than 90% of fintech startups**.

Your next move is **not coding**.  
It’s taking this doc and asking **5 fintech engineers**:

> “Would you integrate this? What would block you?”

Their answers decide everything.

