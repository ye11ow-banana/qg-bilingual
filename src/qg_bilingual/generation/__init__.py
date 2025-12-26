"""Generation utilities for bilingual question generation."""

from .prompts import build_prompt


def generate_questions(*args, **kwargs):  # pragma: no cover - thin wrapper
    from .generate import generate_questions as _generate_questions

    return _generate_questions(*args, **kwargs)


__all__ = ["generate_questions", "build_prompt"]
