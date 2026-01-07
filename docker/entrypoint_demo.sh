#!/usr/bin/env bash
set -e
: "${DEMO_MODE:=local}"         # local|http
: "${SERVER_URL:=http://localhost:8000}"
: "${GRADIO_SERVER_NAME:=0.0.0.0}"
: "${GRADIO_SERVER_PORT:=7860}"
# Use the prebuilt venv from image build (uv sync) to avoid `uv run` re-syncing at container start.
exec .venv/bin/python -m demo.app_gradio --mode "${DEMO_MODE}" --server-url "${SERVER_URL}" --host "${GRADIO_SERVER_NAME}" --port "${GRADIO_SERVER_PORT}"
