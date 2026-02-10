# Python SDK (Starter)

Minimal SDK wrapper for the `/analyze` endpoint.

Example:

```python
from sdk.python.client import MisalignmentClient

client = MisalignmentClient(base_url="http://localhost:8000", api_key="YOUR_KEY")
result = client.analyze(
    ticker="AAPL",
    question="Is this a good investment over the next 12 months?",
    time_horizon="12m",
)
print(result["summary"])
```
