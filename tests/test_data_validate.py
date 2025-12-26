import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "lang, path",
    [
        ("en", Path("data/artifacts/en/val.jsonl")),
        ("ua", Path("data/artifacts/ua/val.jsonl")),
    ],
)
def test_jsonl_validator_runs(lang: str, path: Path, tmp_path: Path):
    assert path.exists(), f"Missing fixture file: {path}"
    stats_path = tmp_path / f"stats_{lang}.json"

    result = subprocess.run(
        [
            sys.executable,
            "data/scripts/validate_jsonl.py",
            "--path",
            str(path),
            "--lang",
            lang,
            "--write-stats",
            str(stats_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        pytest.fail(f"Validation failed for {lang}: {result.stdout}\n{result.stderr}")

    assert stats_path.exists(), "Validator should emit stats JSON when requested"
