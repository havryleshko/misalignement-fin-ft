from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field

from backend.api.schemas import AnalyzeResponse


_UNCERTAINTY_TERMS = ("may", "could", "likely", "uncertain", "probabilistic")


@dataclass
class MetricsState:
    analyze_requests: int = 0
    analyze_successes: int = 0
    analyze_errors: int = 0
    uncertainty_conformant_responses: int = 0
    source_coverage_conformant_responses: int = 0
    source_coverage_total_responses: int = 0
    time_to_first_successful_analyze_seconds: list[float] = field(default_factory=list)
    first_analyze_attempt_epoch_by_api_key: dict[int, float] = field(default_factory=dict)
    first_analyze_success_recorded_by_api_key: set[int] = field(default_factory=set)
    error_codes: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    external_signals: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))


_LOCK = threading.Lock()
_STATE = MetricsState()


def mark_analyze_attempt(api_key_id: int | None, now_epoch: float | None = None) -> None:
    timestamp = now_epoch if now_epoch is not None else time.time()
    with _LOCK:
        _STATE.analyze_requests += 1
        if api_key_id is not None and api_key_id not in _STATE.first_analyze_attempt_epoch_by_api_key:
            _STATE.first_analyze_attempt_epoch_by_api_key[api_key_id] = timestamp


def mark_analyze_success(api_key_id: int | None, now_epoch: float | None = None) -> None:
    timestamp = now_epoch if now_epoch is not None else time.time()
    with _LOCK:
        _STATE.analyze_successes += 1
        if api_key_id is None:
            return
        if api_key_id in _STATE.first_analyze_success_recorded_by_api_key:
            return
        first_attempt = _STATE.first_analyze_attempt_epoch_by_api_key.get(api_key_id)
        if first_attempt is None:
            return
        _STATE.first_analyze_success_recorded_by_api_key.add(api_key_id)
        _STATE.time_to_first_successful_analyze_seconds.append(
            max(timestamp - first_attempt, 0.0)
        )


def mark_analyze_error(error_code: str) -> None:
    with _LOCK:
        _STATE.analyze_errors += 1
        _STATE.error_codes[error_code] += 1


def record_response_quality(
    response: AnalyzeResponse, required_sources: set[str]
) -> None:
    summary_lower = response.summary.lower()
    disclaimer_lower = response.disclaimer.lower()
    has_uncertainty = any(
        term in summary_lower or term in disclaimer_lower for term in _UNCERTAINTY_TERMS
    )
    source_coverage_ok = required_sources.issubset(set(response.sources))

    with _LOCK:
        if has_uncertainty:
            _STATE.uncertainty_conformant_responses += 1
        _STATE.source_coverage_total_responses += 1
        if source_coverage_ok:
            _STATE.source_coverage_conformant_responses += 1


def record_external_signal(name: str, value: float) -> None:
    with _LOCK:
        _STATE.external_signals[name].append(value)


def snapshot() -> dict[str, object]:
    with _LOCK:
        return {
            "analyze_requests": _STATE.analyze_requests,
            "analyze_successes": _STATE.analyze_successes,
            "analyze_errors": _STATE.analyze_errors,
            "uncertainty_conformant_responses": _STATE.uncertainty_conformant_responses,
            "source_coverage_conformant_responses": _STATE.source_coverage_conformant_responses,
            "source_coverage_total_responses": _STATE.source_coverage_total_responses,
            "time_to_first_successful_analyze_seconds": list(
                _STATE.time_to_first_successful_analyze_seconds
            ),
            "error_codes": dict(_STATE.error_codes),
            "external_signals": {
                key: list(values) for key, values in _STATE.external_signals.items()
            },
        }


def reset_for_tests() -> None:
    with _LOCK:
        _STATE.analyze_requests = 0
        _STATE.analyze_successes = 0
        _STATE.analyze_errors = 0
        _STATE.uncertainty_conformant_responses = 0
        _STATE.source_coverage_conformant_responses = 0
        _STATE.source_coverage_total_responses = 0
        _STATE.time_to_first_successful_analyze_seconds.clear()
        _STATE.first_analyze_attempt_epoch_by_api_key.clear()
        _STATE.first_analyze_success_recorded_by_api_key.clear()
        _STATE.error_codes.clear()
        _STATE.external_signals.clear()
