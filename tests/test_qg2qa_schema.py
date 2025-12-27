import pytest

pytest.importorskip("torch")

from qg_bilingual.eval.normalize import NormalizationConfig, Normalizer
from qg_bilingual.eval.qg2qa import (
    QAExample,
    Prediction,
    Thresholds,
    _aggregate,
    _build_detail_row,
)


def test_single_example_scores_pass():
    normalizer = Normalizer(NormalizationConfig())
    thresholds = Thresholds(f1_pass=0.6, conf_pass=0.2)

    example = QAExample(
        id="1",
        question="Which city is the capital of France?",
        context="France's capital city is Paris, which is known for the Eiffel Tower.",
        gold_answer="Paris",
    )
    prediction = Prediction(pred="Paris", confidence=0.9, used_no_answer=False)

    detail = _build_detail_row(example, prediction, normalizer, thresholds, include_unanswerable=False)
    summary = _aggregate([detail])

    assert detail["em"] > 0
    assert detail["f1"] > 0
    assert summary["pass_rate"] == 1
