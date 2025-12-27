#!/usr/bin/env bash
set -e
# MODELS can be overridden via env or mounted config
: "${SERVER_CONFIG:=server/config.yaml}"
uv run uvicorn qg_bilingual.server.app:app --host 0.0.0.0 --port 8000
