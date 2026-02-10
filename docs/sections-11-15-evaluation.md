# Codebase Evaluation Scorecard (Design Sections 11-15)

## Decision Snapshot

- Recommendation: `Proceed with conditions`
- Overall readiness score: `Partial`
- Main blockers to full alignment: `HTTPS enforcement evidence`, `single-data-source Phase 1 posture`, `success/kill operational instrumentation`

## Checklist Results

Legend: `Met`, `Partially met`, `Not met`, `Not yet measurable`

| Design Requirement | Status | Evidence | Notes |
| --- | --- | --- | --- |
| Section 11: Docker required | Met | `infra/docker/docker-compose.yml` | Docker compose file exists with API/Postgres/Redis services. |
| Section 11: Postgres required | Met | `infra/docker/docker-compose.yml`, `backend/models/db.py` | Postgres service declared and SQLAlchemy async DB configured. |
| Section 11: Redis (rate limiting) required | Met | `infra/docker/docker-compose.yml`, `backend/main.py` | Redis service declared and Redis-backed rate limiting middleware implemented. |
| Section 11: Terraform excluded | Met | repository scan (`*.tf`) | No Terraform files found. |
| Section 11: Kubernetes excluded | Met | repository scan (`k8s`) | No Kubernetes manifests or directories found. |
| Section 11: Supabase Auth excluded | Met | repository scan (`supabase`) | No Supabase auth integration found. |
| Section 11: Frontend UI excluded | Met | repository scan (`frontend`, `*.tsx`) | No frontend code present. |
| Section 12: HTTPS only | Partially met | `backend/main.py` | App does not enforce HTTPS/forwarded-proto checks; likely deployment responsibility. |
| Section 12: Hashed API keys | Met | `backend/api/auth.py` | SHA-256 + salt hash comparison for API key authentication. |
| Section 12: Rate limiting | Met | `backend/main.py` | Per-minute Redis counter, 429 response, and rate-limit headers implemented. |
| Section 12: Key revocation | Met | `backend/api/auth.py`, `backend/models/entities.py` | Auth query requires `status == "active"`; revoked/inactive keys are blocked. |
| Section 12: No PII stored | Partially met | `backend/models/entities.py`, `backend/data/sec_edgar.py` | DB stores `Customer.name`; repo includes hard-coded contact email in SEC user agent string. |
| Section 13 Phase 1: `/analyze` endpoint | Met | `backend/api/routes.py` | Endpoint exists and is protected by API-key auth dependency. |
| Section 13 Phase 1: One data source | Not met | `backend/data/assembly.py` | Pipeline requires multiple sources (Alpha Vantage + SEC filings), optional analyst consensus. |
| Section 13 Phase 1: One model | Partially met | `backend/models/llm.py`, `backend/orchestration/pipeline.py` | Single configured LLM target exists, but deterministic fallback path is also active in test/no-LLM mode. |
| Section 13 Phase 1: One customer | Partially met | `backend/scripts/seed_api_key.py`, `backend/models/entities.py` | Multi-customer schema exists; seed script creates default customer but no explicit single-customer guardrail. |
| Section 13 Phase 2: SDKs | Not met | repository structure scan | No `sdk/` directory in current codebase. |
| Section 13 Phase 2: Scenario analysis | Met | `backend/orchestration/risk.py`, `backend/api/schemas.py` | Bull/base/bear scenarios computed and exposed in contract. |
| Section 13 Phase 2: Usage-based billing | Not met | `backend/models/entities.py`, backend scan for billing terms | Usage logs exist, but no metering-to-billing pipeline or billing integration. |
| Section 13 Phase 3: White-label integrations | Not met | repository scan | No white-label module/config discovered. |
| Section 13 Phase 3: Enterprise compliance extensions | Not met | `backend/orchestration/compliance.py` | Basic compliance sanitization only; no enterprise policy extension framework. |
| Section 14: Integrate in <1 day | Partially met | `docs/local-dev.md`, `infra/docker/docker-compose.yml` | Setup docs exist but no measured onboarding benchmark or scripted quickstart validation. |
| Section 14: Zero hallucinated outputs | Partially met | `backend/orchestration/pipeline.py`, `backend/models/llm.py` | Source checks and strict schema validation reduce risk; absolute zero cannot be proven by current tests/process. |
| Section 14: Explicit uncertainty in 100% responses | Partially met | `backend/orchestration/compliance.py`, `backend/orchestration/pipeline.py` | Summary uncertainty injection + disclaimer present, but no endpoint-level conformance metric to prove 100%. |
| Section 14: Positive feedback from fintech engineers | Not yet measurable | `design.md` process requirement | Requires external interviews/usability feedback loop not represented in code. |
| Section 15: Platforms refuse to integrate signal | Not met | repo/process scan | No integration-rejection tracking mechanism found. |
| Section 15: Deterministic-picks demand signal | Not met | repo/process scan | No customer feedback taxonomy or telemetry for this signal. |
| Section 15: Data costs exceed value signal | Not met | repo/process scan | No cost instrumentation or unit-economics tracking pipeline. |

## Key Risks

1. Deployment security ambiguity: HTTPS-only is a design requirement, but no app-level enforcement evidence exists.
2. Phase 1 drift: multiple data sources are currently core to assembly, conflicting with "one data source" scope.
3. Success and kill criteria are under-instrumented: strategic criteria exist in design but lack measurable telemetry.

## Conditions to Move from Partial to Aligned

1. Add a deployment control proving HTTPS-only traffic (gateway/proxy config + validation check).
2. Define and enforce explicit Phase 1 profile (single source, single model path, single-customer operating mode).
3. Add operational score metrics:
   - integration time benchmark (time-to-first-successful `/analyze`)
   - uncertainty conformance rate (responses containing required uncertainty/disclaimer fields)
   - hallucination proxy checks (source attribution coverage and audit sampling)
4. Add kill-criteria telemetry and review cadence:
   - integration refusal count and reasons (weekly)
   - deterministic-output demand rate from user feedback (weekly)
   - per-request data cost and monthly margin trend (weekly/monthly)

## Minimal Telemetry Spec for Section 15

| Signal | Metric | Threshold | Cadence |
| --- | --- | --- | --- |
| Platforms refuse to integrate | `integration_refusal_rate` | `>= 40%` over rolling 4-week outreach | Weekly |
| Users demand deterministic picks | `deterministic_request_ratio` | `>= 30%` of qualified feedback | Weekly |
| Data costs exceed value | `data_cost_to_revenue_ratio` | `> 1.0` for 2 consecutive months | Weekly snapshot, monthly decision |

## Final Evaluation

The codebase is strong for MVP backend mechanics (auth, rate limiting, API contract, and core orchestration), but it is not yet fully aligned with the design's strategic guardrails and decision criteria in sections 11-15. The project should proceed only with the listed conditions, especially around Phase 1 scope discipline and measurable success/kill instrumentation.
