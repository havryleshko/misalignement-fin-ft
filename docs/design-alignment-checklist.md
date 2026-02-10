# Design Alignment Checklist

This checklist maps `design.md` requirements to concrete implementation evidence in the repository.

Legend: `Met`, `Partially met`, `Waived`.

## Sections 1-5 (Purpose, Problem, Users, Non-goals, Principles)

| Requirement | Status | Evidence |
| --- | --- | --- |
| API-first decision-support engine | Met | `backend/api/routes.py`, `README.md` |
| No consumer UI / no trade automation | Met | Repository contains backend-only services; no frontend app |
| Probabilistic + uncertainty-aware outputs | Met | `backend/orchestration/risk.py`, `backend/orchestration/compliance.py` |
| Bias-aware reasoning | Met | `backend/orchestration/intent_bias.py` |
| Auditable/traceable responses | Met | `backend/main.py` request/trace headers + usage logging |

## Sections 6-8 (Architecture and API)

| Requirement | Status | Evidence |
| --- | --- | --- |
| API gateway (auth + rate limiting) | Met | `backend/main.py`, `backend/api/auth.py` |
| Orchestrator pipeline with fail-closed behavior | Met | `backend/orchestration/pipeline.py`, `backend/orchestration/errors.py` |
| Data retrieval with freshness and citations | Met | `backend/data/assembly.py` |
| Strict JSON schema output | Met | `backend/api/schemas.py`, `backend/models/llm.py` |
| Mandatory `/analyze` output fields or error | Met | `backend/orchestration/pipeline.py`, `backend/api/routes.py` |

## Section 9 (Data Model)

| Requirement | Status | Evidence |
| --- | --- | --- |
| `customers`/`api_keys`/`usage_logs` minimal fields | Met | `backend/models/entities.py`, `backend/alembic/versions/0001_initial.py`, `backend/alembic/versions/0002_usage_logs_latency_column.py` |

## Section 10 (Repo Structure)

| Requirement | Status | Evidence |
| --- | --- | --- |
| Root `README.md` | Met | `README.md` |
| `infra/docker/Dockerfile` + compose | Met | `infra/docker/Dockerfile`, `infra/docker/docker-compose.yml` |
| SDK skeleton (`python`, `ts`) | Met | `sdk/python/`, `sdk/ts/` |
| Backend service module layout | Met | `backend/api`, `backend/orchestration`, `backend/data`, `backend/models` |

## Sections 11-12 (Infra and Security)

| Requirement | Status | Evidence |
| --- | --- | --- |
| Docker + Postgres + Redis | Met | `infra/docker/docker-compose.yml`, `backend/models/db.py`, `backend/main.py` |
| Exclude Terraform/Kubernetes/Supabase Auth/Frontend UI | Met | No corresponding code/artifacts present |
| HTTPS-only strategy | Met | HTTPS enforcement middleware in `backend/main.py` with local/test bypass |
| Hashed API keys + revocation + rate limiting | Met | `backend/api/auth.py`, `backend/main.py` |
| No PII stored in hardcoded service metadata | Met | SEC user-agent now configurable via `backend/config.py` + `backend/data/sec_edgar.py` |

## Sections 13-15 (Rollout, Success, Kill Criteria)

| Requirement | Status | Evidence |
| --- | --- | --- |
| Phase 1 `/analyze` endpoint | Met | `backend/api/routes.py` |
| Phase 1 one data source | Waived | Product decision: multiple sources accepted |
| Scenario analysis | Met | `backend/orchestration/risk.py` |
| Usage-based telemetry foundation | Met | `backend/models/entities.py` (`usage_logs`), `backend/orchestration/metrics.py` |
| Success/Kill measurable signals in code/docs | Met | `backend/orchestration/metrics.py`, `docs/metrics-runbook.md` |

