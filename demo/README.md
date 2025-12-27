# QG bilingual demo

Gradio UI for the safety-aware question generation pipeline. Works either by calling the FastAPI server or by loading the local stub pipeline directly.

## Run locally
- **Local pipeline:** `uv run python demo/app_gradio.py --mode local --config src/qg_bilingual/server/config.yaml`
- **HTTP server:** start `uv run uvicorn qg_bilingual.server.app:app --reload --port 8000`, then run `uv run python demo/app_gradio.py --mode http --server-url http://localhost:8000`
- Add `--dev` to show the unsafe-question toggle; add `--share` to expose the Gradio share link.

## Features
- Context + optional answer inputs, language/mode/WH selectors, and on-click generation.
- Shows PASSED/BLOCKED badge, blocking reasons, metrics table, and debug info.
- "Try example" dropdown pre-fills 10 EN/UA examples; "Copy JSON" exports the full response.
- Advanced panel displays decoding config and filter thresholds from the YAML config.
