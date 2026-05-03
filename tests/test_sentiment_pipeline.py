import pytest
import pandas as pd

import sentiment_analysis as sentiment_module
from sentiment_analysis import (
    _normalize_label,
    _chunked,
    run_sentiment_evaluation,
)
import sentiment_analysis_llm as llm_module
from sentiment_analysis_llm import (
    _normalize_label as llm_normalize_label,
    _chunked as llm_chunked,
    run_sentiment_evaluation as run_llm_sentiment_evaluation,
)

def test_normalize_positive():
    assert _normalize_label("positive", 0.9) == "positive"

def test_normalize_negative():
    assert _normalize_label("negative", 0.9) == "negative"

def test_label_0_maps_negative():
    assert _normalize_label("LABEL_0", 0.8) == "negative"

def test_label_1_maps_positive():
    assert _normalize_label("LABEL_1", 0.8) == "positive"

def test_fallback_logic():
    assert _normalize_label("unknown", 0.7) == "positive"
    assert _normalize_label("unknown", 0.3) == "negative"


def test_chunking_basic():
    data = [1, 2, 3, 4, 5]
    chunks = list(_chunked(data, 2))

    assert chunks == [[1, 2], [3, 4], [5]]

def test_chunking_empty():
    assert list(_chunked([], 3)) == []

def test_chunking_exact():
    data = [1, 2, 3, 4]
    chunks = list(_chunked(data, 2))

    assert chunks == [[1, 2], [3, 4]]


class _FakeSentimentPipeline:
    def __call__(self, texts):
        return [
            {
                "label": "NEGATIVE" if ("kötü" in text.lower() or "berbat" in text.lower()) else "POSITIVE",
                "score": 0.9,
            }
            for text in texts
        ]


def test_evaluation_pipeline(tmp_path, monkeypatch):
    test_file = tmp_path / "test_data.csv"

    df = pd.DataFrame({
        "text": [
            "Türkiye çok güzel",
            "Bu hizmet çok kötü",
            "Harika bir deneyim",
            "Berbat bir sistem"
        ],
        "label": [
            "positive",
            "negative",
            "positive",
            "negative"
        ]
    })

    df.to_csv(test_file, index=False)

    monkeypatch.setattr(sentiment_module, "_build_sentiment_pipeline", lambda: _FakeSentimentPipeline())

    result = run_sentiment_evaluation(
        context={"base_dir": str(tmp_path)},
        evaluation_config={
            "dataset_path": "test_data.csv",
            "text_column": "text",
            "label_column": "label",
        },
    )

    assert "metrics" in result
    assert "accuracy" in result["metrics"]
    assert "macro_avg" in result["metrics"]

    assert 0 <= result["metrics"]["accuracy"] <= 1
    assert 0 <= result["metrics"]["macro_avg"]["precision"] <= 1
    assert 0 <= result["metrics"]["macro_avg"]["recall"] <= 1
    assert 0 <= result["metrics"]["macro_avg"]["f1"] <= 1


def test_error_analysis_present(tmp_path, monkeypatch):
    test_file = tmp_path / "test_data.csv"

    df = pd.DataFrame({
        "text": ["iyi", "kötü"],
        "label": ["positive", "negative"]
    })

    df.to_csv(test_file, index=False)

    monkeypatch.setattr(sentiment_module, "_build_sentiment_pipeline", lambda: _FakeSentimentPipeline())

    result = run_sentiment_evaluation(
        context={"base_dir": str(tmp_path)},
        evaluation_config={
            "dataset_path": "test_data.csv",
            "text_column": "text",
            "label_column": "label",
        },
    )

    assert "error_analysis" in result
    assert "misclassified_examples" in result["error_analysis"]
    assert "hardest_errors" in result["error_analysis"]


def test_llm_normalize_positive():
    assert llm_normalize_label("positive") == ("positive", 0.90)


def test_llm_normalize_negative_with_think_tag():
    assert llm_normalize_label("<think>reasoning</think>negative") == ("negative", 0.90)


def test_llm_fallback_logic():
    assert llm_normalize_label("unknown", 0.7) == ("positive", 0.50)
    assert llm_normalize_label("unknown", 0.3) == ("negative", 0.50)


def test_llm_chunking_basic():
    data = [1, 2, 3, 4, 5]
    chunks = list(llm_chunked(data, 2))

    assert chunks == [[1, 2], [3, 4], [5]]


class _FakeLlmBatchPredictor:
    def __call__(self, tokenizer, model, texts):
        return [
            ("positive" if "güzel" in text.lower() or "harika" in text.lower() or "iyi" in text.lower() else "negative", 0.90)
            for text in texts
        ]


def test_llm_evaluation_pipeline(tmp_path, monkeypatch):
    test_file = tmp_path / "llm_test_data.csv"

    df = pd.DataFrame({
        "text": [
            "Türkiye çok güzel",
            "Bu hizmet çok kötü",
            "Harika bir deneyim",
            "Berbat bir sistem",
        ],
        "label": [
            "positive",
            "negative",
            "positive",
            "negative",
        ],
    })

    df.to_csv(test_file, index=False)

    monkeypatch.setattr(llm_module, "_build_sentiment_pipeline", lambda: (object(), object()))
    monkeypatch.setattr(llm_module, "_infer_batch", _FakeLlmBatchPredictor())

    result = run_llm_sentiment_evaluation(
        context={"base_dir": str(tmp_path)},
        evaluation_config={
            "dataset_path": "llm_test_data.csv",
            "text_column": "text",
            "label_column": "label",
        },
    )

    assert "metrics" in result
    assert "accuracy" in result["metrics"]
    assert "error_analysis" in result
    assert "misclassified_examples" in result["error_analysis"]