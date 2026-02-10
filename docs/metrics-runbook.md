# Metrics and Signals Runbook

This document defines the in-repo metrics collection points for success and kill criteria from `design.md`.

## Collection Points Implemented

- `time_to_first_successful_analyze`
  - Collected by API key from first `/analyze` attempt to first successful `/analyze`.
  - Hooked in `backend/main.py` using `backend/orchestration/metrics.py`.

- `uncertainty_conformance_rate`
  - Derived from response summary/disclaimer uncertainty language checks.
  - Hooked in `backend/orchestration/pipeline.py` through `record_response_quality(...)`.

- `source_coverage_conformance_rate`
  - Validates required data sources are present in response output.
  - Hooked in `backend/orchestration/pipeline.py`.

- `analyze_error_codes`
  - Pipeline error code counts for failed `/analyze` requests.
  - Hooked in `backend/api/routes.py`.

## Kill Criteria Signal Schema

The following external signals are captured through `record_external_signal(name, value)` in `backend/orchestration/metrics.py`:

- `integration_refusal_rate`
- `deterministic_request_ratio`
- `data_cost_to_revenue_ratio`

These signals are intended to be fed by CRM/support/billing pipelines (outside request path), then reviewed weekly/monthly per design criteria.

## Snapshot Access

Use `backend.orchestration.metrics.snapshot()` from internal tooling/tests to inspect current in-process counters.

## Notes

- Current implementation is in-memory and process-local (MVP).
- For production durability, export to a metrics backend (Prometheus/OpenTelemetry/warehouse).
