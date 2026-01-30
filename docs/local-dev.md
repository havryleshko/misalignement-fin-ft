## Local Development

### Prereqs
- Python 3.11+
- `.env` file based on `.env.example`

### Run the API
```bash
uvicorn backend.main:app --reload
```

The service will fail fast on startup if required env vars are missing.
