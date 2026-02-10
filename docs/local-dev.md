## Local Development

### Prereqs
- Python 3.11+
- `.env` file based on `.env.example`

### Start local dependencies
```bash
docker compose -f infra/docker/docker-compose.yml up -d postgres redis
```

### Run the API
```bash
uvicorn backend.main:app --reload
```

The service will fail fast on startup if required env vars are missing.

### HTTPS behavior
- Production-style environments enforce HTTPS-only requests.
- Local/development/test bypass HTTPS enforcement by default.
