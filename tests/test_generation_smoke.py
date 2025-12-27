import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from qg_bilingual.generation.generate import generate_questions


@pytest.mark.slow
@pytest.mark.parametrize("strategy", ["beam", "topp"])
def test_t5_generate_non_empty(strategy: str, tmp_path):
    cfg_model = {
        "name": "google/flan-t5-small",
        "max_input_len": 128,
        "batch_size": 1,
        "seed": 0,
    }
    cfg_dec = {
        "strategy": strategy,
        "num_beams": 2,
        "length_penalty": 1.0,
        "top_p": 0.9,
        "temperature": 0.9,
        "max_new_tokens": 32,
        "min_new_tokens": 8,
    }
    cfg_task = {"mode": "aware", "lang": "en", "run_dir": str(tmp_path)}
    records = [
        {
            "context": (
                "Paris is the capital and most populous city of France, situated on the Seine river in the north of the "
                "country; it is renowned for its art, gastronomy, and culture, attracting millions of tourists annually."
            ),
            "answer": "France",
            "lang": "en",
        }
    ]

    outputs = generate_questions(records, cfg_model, cfg_dec, cfg_task, debug=True)
    assert outputs, "Generation should return at least one result"
    question = outputs[0]["question"]
    assert isinstance(question, str)
    assert question.strip(), "Generated question should not be empty"
    assert question.strip().endswith("?")
    assert (tmp_path / "debug_gen.txt").exists()
