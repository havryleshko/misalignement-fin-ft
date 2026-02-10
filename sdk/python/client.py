from __future__ import annotations

from typing import Any

import httpx


class MisalignmentClient:
    def __init__(self, base_url: str, api_key: str, timeout_seconds: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def analyze(self, ticker: str, question: str, time_horizon: str) -> dict[str, Any]:
        payload = {
            "ticker": ticker,
            "question": question,
            "time_horizon": time_horizon,
        }
        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(
                f"{self.base_url}/analyze",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            return response.json()
