"""Pydantic schemas for the safety-aware question generation API."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Request body for /generate_safe."""

    context: str = Field(..., description="Passage text used for question generation")
    answer: Optional[str] = Field(
        None, description="Optional answer string used for answer-aware prompting"
    )
    lang: Literal["en", "ua"] = Field(
        "en", description="Two-letter language code to pick QA/toxicity models"
    )
    mode: Literal["aware", "agnostic"] = Field(
        "aware", description="aware -> use answer, agnostic -> context only"
    )
    wh_type: Optional[Literal["who", "when", "where", "what", "why", "how", "how_many"]] = (
        Field(None, description="Optional wh-constraint for generation")
    )


class GenerateResponse(BaseModel):
    """Response payload returned by /generate_safe."""

    question: Optional[str]
    passed: bool
    reasons: list[str]
    metrics: dict
    debug: dict


class HealthResponse(BaseModel):
    status: str = "ok"
