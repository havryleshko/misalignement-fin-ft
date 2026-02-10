from dataclasses import dataclass
import math
import os
import random
from statistics import mean, pstdev
from backend.data.schemas import PriceHistory


@dataclass(frozen=True)
class RiskMetrics:
    expected_return: float
    confidence_interval: list[float]
    scenarios: dict[str, float]
    probability_positive: float
    volatility: float
    max_drawdown: float


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        raise ValueError("No values for percentile calculation")
    if pct <= 0:
        return values[0]
    if pct >= 100:
        return values[-1]
    k = (len(values) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values[int(k)]
    return values[f] + (values[c] - values[f]) * (k - f)


def _compute_drawdown(prices: list[float]) -> float:
    if not prices:
        return 0.0
    peak = prices[0]
    max_drawdown = 0.0
    for price in prices[1:]:
        if price > peak:
            peak = price
            continue
        if peak > 0:
            drawdown = (peak - price) / peak
            max_drawdown = max(max_drawdown, drawdown)
    return max_drawdown


def _compute_returns(prices: list[float]) -> list[float]:
    returns: list[float] = []
    for idx in range(1, len(prices)):
        prev_close = prices[idx - 1]
        if prev_close == 0:
            continue
        returns.append((prices[idx] - prev_close) / prev_close)
    return returns


def _simulate_returns(
    expected_return: float,
    volatility: float,
    simulations: int,
    seed: int | None,
) -> list[float]:
    if simulations <= 0:
        raise ValueError("Simulation count must be positive")
    if volatility < 0:
        raise ValueError("Volatility must be non-negative")

    if volatility == 0:
        return [expected_return for _ in range(simulations)]

    rng = random.Random(seed)
    return [rng.gauss(expected_return, volatility) for _ in range(simulations)]


def compute_risk_metrics(price_history: PriceHistory) -> RiskMetrics:
    closes_latest_first = [point.close for point in price_history.points]
    closes = list(reversed(closes_latest_first))
    if len(closes) < 2:
        raise ValueError("Insufficient price history")

    returns = _compute_returns(closes)
    if not returns:
        raise ValueError("Missing valid returns")

    returns_sorted = sorted(returns)
    expected = mean(returns)
    volatility = pstdev(returns)
    simulations = int(os.getenv("RISK_MONTE_CARLO_SIMS", "2000"))
    seed = 42 if os.getenv("ENV") == "test" else None
    simulated_returns = _simulate_returns(
        expected_return=expected,
        volatility=volatility,
        simulations=simulations,
        seed=seed,
    )
    simulated_sorted = sorted(simulated_returns)
    confidence_interval = [
        _percentile(simulated_sorted, 5),
        _percentile(simulated_sorted, 95),
    ]
    scenarios = {
        "bull": _percentile(simulated_sorted, 90),
        "base": _percentile(simulated_sorted, 50),
        "bear": _percentile(simulated_sorted, 10),
    }
    probability_positive = sum(1 for r in simulated_returns if r > 0) / len(
        simulated_returns
    )
    max_drawdown = _compute_drawdown(closes)
    return RiskMetrics(
        expected_return=expected,
        confidence_interval=confidence_interval,
        scenarios=scenarios,
        probability_positive=probability_positive,
        volatility=volatility,
        max_drawdown=max_drawdown,
    )
