"""FastAPI app exposing a safety-aware QG endpoint."""

from __future__ import annotations

import logging
from typing import Dict

from fastapi import Depends, FastAPI, HTTPException

from . import schemas
from .deps import get_pipeline
from .pipeline import SafeGenerationPipeline

LOGGER = logging.getLogger(__name__)

app = FastAPI(title="qg-bilingual-safe-api")


@app.get("/healthz", response_model=schemas.HealthResponse)
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/generate_safe", response_model=schemas.GenerateResponse)
def generate_safe(
    payload: schemas.GenerateRequest, pipeline: SafeGenerationPipeline = Depends(get_pipeline)
) -> schemas.GenerateResponse:
    try:
        response = pipeline.run(payload)
    except ValueError as exc:  # validation errors at pipeline level
        LOGGER.exception("Bad request: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive fallback
        LOGGER.exception("Inference error: %s", exc)
        raise HTTPException(status_code=500, detail="inference_error") from exc
    return response


__all__ = ["app"]
