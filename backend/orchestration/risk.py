from dataclasses import dataclass
import math
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
    confidence_interval = [
        _percentile(returns_sorted, 5),
        _percentile(returns_sorted, 95),
    ]
    scenarios = {
        "bull": _percentile(returns_sorted, 90),
        "base": _percentile(returns_sorted, 50),
        "bear": _percentile(returns_sorted, 10),
    }
    probability_positive = sum(1 for r in returns if r > 0) / len(returns)
    max_drawdown = _compute_drawdown(closes)
    return RiskMetrics(
        expected_return=expected,
        confidence_interval=confidence_interval,
        scenarios=scenarios,
        probability_positive=probability_positive,
        volatility=volatility,
        max_drawdown=max_drawdown,
    )
