#!/usr/bin/env bash
# Start the QYRO demo at http://localhost:8000
# API keys (optional) go in qyro_app/.env, see qyro_app/.env.example
set -euo pipefail
cd "$(dirname "$0")/.."
ENV_FILE=qyro_app/.env
[ -f "$ENV_FILE" ] || touch "$ENV_FILE"
export PYTHONWARNINGS=ignore
exec .venv/bin/uvicorn qyro_app.main:app --host 127.0.0.1 --port "${PORT:-8000}" --env-file "$ENV_FILE"
