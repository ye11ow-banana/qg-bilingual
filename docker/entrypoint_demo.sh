#!/usr/bin/env bash
set -e
: "${DEMO_MODE:=local}"         # local|http
: "${SERVER_URL:=http://localhost:8000}"
uv run python demo/app_gradio.py --mode "${DEMO_MODE}" --server-url "${SERVER_URL}"
