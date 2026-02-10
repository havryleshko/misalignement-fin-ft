# TypeScript SDK (Starter)

Minimal SDK wrapper for the `/analyze` endpoint.

Example:

```ts
import { MisalignmentClient } from "./index";

const client = new MisalignmentClient("http://localhost:8000", "YOUR_KEY");
const result = await client.analyze({
  ticker: "AAPL",
  question: "Is this a good investment over the next 12 months?",
  time_horizon: "12m",
});

console.log(result.summary);
```
