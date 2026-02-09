from typing import Tuple

from backend.data.schemas import AnalystConsensus


def get_analyst_consensus(ticker: str) -> Tuple[AnalystConsensus | None, str | None]:
    _ = ticker
    return None, None
