#!/usr/bin/env bash
set -e
# MODELS can be overridden via env or mounted config
: "${SERVER_CONFIG:=server/config.yaml}"

# The FastAPI app reads its config path from QG_SERVER_CONFIG.
# Keep SERVER_CONFIG for backward compatibility and map it through.
: "${QG_SERVER_CONFIG:=$SERVER_CONFIG}"
export QG_SERVER_CONFIG

# Dependencies are installed during image build into `/app/.venv` via `uv sync`.
# Run uvicorn from that venv so we don't trigger `uv run` / re-sync at startup.
exec .venv/bin/python -m uvicorn qg_bilingual.server.app:app --host 0.0.0.0 --port 8000
