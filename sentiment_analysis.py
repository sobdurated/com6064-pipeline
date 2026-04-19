import gc
import csv
import os
from typing import Any, Dict, Iterable, List, Tuple

import torch
from pymongo import UpdateOne
from transformers import pipeline


SOURCE_COLLECTION = os.getenv("MONGO_SOURCE_COLLECTION", "posts_processed")
MODEL_NAME = "savasy/bert-base-turkish-sentiment-cased"
BATCH_SIZE = int(os.getenv("SENTIMENT_BATCH_SIZE", "16"))
MAX_TEXT_LEN = int(os.getenv("SENTIMENT_MAX_TEXT_LEN", "512"))


def _normalize_label(raw_label: str, score: float) -> str:
    label = (raw_label or "").strip().lower()
    if "negative" in label or label == "label_0":
        return "negative"
    if "positive" in label or label == "label_1":
        return "positive"

    return "positive" if score >= 0.5 else "negative"


def _chunked(records: List[Dict], size: int) -> Iterable[List[Dict]]:
    for idx in range(0, len(records), size):
        yield records[idx : idx + size]


def _load_posts_for_inference(collection) -> List[Dict]:
    query = {
        "text": {"$type": "string", "$ne": ""},
        "sentiment.label": "neutral",
    }
    projection = {"_id": 1, "text": 1}
    return list(collection.find(query, projection))


def _build_sentiment_pipeline():
    device = 0 if torch.cuda.is_available() else -1
    print(f"loading model: {MODEL_NAME}")
    return pipeline(
        task="text-classification",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        truncation=True,
        max_length=MAX_TEXT_LEN,
        device=device,
    )


def _resolve_dataset_path(dataset_path: str, context: Dict[str, Any]) -> str:
    raw_path = (dataset_path or "").strip()
    if not raw_path:
        raise ValueError("evaluation.dataset_path must be provided")

    if os.path.isabs(raw_path):
        return raw_path

    base_dir = str(context.get("base_dir") or "")
    if not base_dir:
        return raw_path
    return os.path.join(base_dir, raw_path)


def _load_labeled_samples(csv_path: str, text_column: str, label_column: str, id_column: str) -> List[Dict[str, str]]:
    samples: List[Dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("evaluation CSV has no header row")

        if text_column not in reader.fieldnames:
            raise ValueError(f"missing text column '{text_column}' in evaluation CSV")
        if label_column not in reader.fieldnames:
            raise ValueError(f"missing label column '{label_column}' in evaluation CSV")

        for row in reader:
            text = str(row.get(text_column, "") or "").strip()
            raw_label = str(row.get(label_column, "") or "").strip().lower()
            normalized_true = _normalize_label(raw_label, 0.0)

            if not text:
                continue

            if normalized_true not in {"positive", "negative"}:
                continue

            samples.append(
                {
                    "entry_id": str(row.get(id_column, "") or ""),
                    "text": text,
                    "true_label": normalized_true,
                }
            )

    return samples


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _compute_metrics(true_labels: List[str], pred_labels: List[str]) -> Dict[str, Any]:
    tp = tn = fp = fn = 0

    for true_label, pred_label in zip(true_labels, pred_labels):
        if true_label == "positive" and pred_label == "positive":
            tp += 1
        elif true_label == "negative" and pred_label == "negative":
            tn += 1
        elif true_label == "negative" and pred_label == "positive":
            fp += 1
        elif true_label == "positive" and pred_label == "negative":
            fn += 1

    total = len(true_labels)
    accuracy = _safe_ratio(tp + tn, total)

    precision_pos = _safe_ratio(tp, tp + fp)
    recall_pos = _safe_ratio(tp, tp + fn)
    f1_pos = _safe_ratio(2 * precision_pos * recall_pos, precision_pos + recall_pos)

    precision_neg = _safe_ratio(tn, tn + fn)
    recall_neg = _safe_ratio(tn, tn + fp)
    f1_neg = _safe_ratio(2 * precision_neg * recall_neg, precision_neg + recall_neg)

    macro_precision = (precision_pos + precision_neg) / 2
    macro_recall = (recall_pos + recall_neg) / 2
    macro_f1 = (f1_pos + f1_neg) / 2

    return {
        "support": total,
        "accuracy": accuracy,
        "confusion_matrix": {
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
        },
        "per_class": {
            "positive": {
                "precision": precision_pos,
                "recall": recall_pos,
                "f1": f1_pos,
                "support": tp + fn,
            },
            "negative": {
                "precision": precision_neg,
                "recall": recall_neg,
                "f1": f1_neg,
                "support": tn + fp,
            },
        },
        "macro_avg": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1,
        },
    }


def run_sentiment_evaluation(context: Dict[str, Any], evaluation_config: Dict[str, Any]) -> Dict[str, Any]:
    dataset_path = _resolve_dataset_path(
        dataset_path=str(evaluation_config.get("dataset_path", "")),
        context=context,
    )
    text_column = str(evaluation_config.get("text_column", "content"))
    label_column = str(evaluation_config.get("label_column", "label"))
    id_column = str(evaluation_config.get("id_column", "entry_id"))
    error_limit = int(evaluation_config.get("error_limit", 25))
    batch_size = int(evaluation_config.get("batch_size", BATCH_SIZE))

    print(f"running evaluation from CSV: {dataset_path}")
    samples = _load_labeled_samples(
        csv_path=dataset_path,
        text_column=text_column,
        label_column=label_column,
        id_column=id_column,
    )

    if not samples:
        print("no valid labeled rows found in evaluation dataset")
        return {
            "dataset_path": dataset_path,
            "total_samples": 0,
            "metrics": {},
            "error_analysis": {
                "total_misclassified": 0,
                "misclassified_examples": [],
                "hardest_errors": [],
            },
        }

    print(f"evaluation samples: {len(samples)}")
    sentiment_pipe = _build_sentiment_pipeline()

    results: List[Dict[str, Any]] = []
    for sample_batch in _chunked(samples, batch_size):
        texts = [item["text"][:MAX_TEXT_LEN] for item in sample_batch]
        predictions = sentiment_pipe(texts)

        for item, pred in zip(sample_batch, predictions):
            raw_label = str(pred.get("label", ""))
            score = float(pred.get("score", 0.0))
            predicted_label = _normalize_label(raw_label, score)
            results.append(
                {
                    "entry_id": item["entry_id"],
                    "text": item["text"],
                    "true_label": item["true_label"],
                    "predicted_label": predicted_label,
                    "score": score,
                    "is_error": item["true_label"] != predicted_label,
                }
            )

    true_labels = [item["true_label"] for item in results]
    pred_labels = [item["predicted_label"] for item in results]
    metrics = _compute_metrics(true_labels=true_labels, pred_labels=pred_labels)

    misclassified = [item for item in results if item["is_error"]]
    misclassified_sorted = sorted(misclassified, key=lambda item: item["score"], reverse=True)

    error_examples = [
        {
            "entry_id": item["entry_id"],
            "true_label": item["true_label"],
            "predicted_label": item["predicted_label"],
            "score": item["score"],
            "text": item["text"][:240],
        }
        for item in misclassified[:error_limit]
    ]

    hardest_errors = [
        {
            "entry_id": item["entry_id"],
            "true_label": item["true_label"],
            "predicted_label": item["predicted_label"],
            "score": item["score"],
            "text": item["text"][:240],
        }
        for item in misclassified_sorted[:error_limit]
    ]

    print(
        "evaluation summary "
        f"| samples={metrics.get('support', 0)} "
        f"| accuracy={metrics.get('accuracy', 0.0):.4f} "
        f"| errors={len(misclassified)}"
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "dataset_path": dataset_path,
        "text_column": text_column,
        "label_column": label_column,
        "id_column": id_column,
        "total_samples": len(results),
        "metrics": metrics,
        "error_analysis": {
            "total_misclassified": len(misclassified),
            "misclassified_examples": error_examples,
            "hardest_errors": hardest_errors,
        },
    }


def run_sentiment_pipeline(context: Dict[str, Any]) -> Tuple[int, int]:
    db = context["db"]
    collection = db[SOURCE_COLLECTION]

    print(f"database: {db.name} | collection: {SOURCE_COLLECTION}")
    sentiment_pipe = _build_sentiment_pipeline()

    posts = _load_posts_for_inference(collection)
    total_posts = len(posts)

    if total_posts == 0:
        print("no documents found with non-empty text in posts_processed")
        return 0, 0

    print(f"documents to process: {total_posts}")

    ops: List[UpdateOne] = []
    for post_batch in _chunked(posts, BATCH_SIZE):
        texts = [doc["text"][:MAX_TEXT_LEN] for doc in post_batch]
        predictions = sentiment_pipe(texts)

        for doc, pred in zip(post_batch, predictions):
            raw_label = pred.get("label", "")
            score = float(pred.get("score", 0.0))
            sentiment_label = _normalize_label(raw_label, score)
            print(f"post_id: {doc['_id']} | score: {score:.4f} | normalized: {sentiment_label}")

            ops.append(
                UpdateOne(
                    {"_id": doc["_id"]},
                    {
                        "$set": {
                            "sentiment.label": sentiment_label,
                            "sentiment.score": score,
                            "sentiment.model": MODEL_NAME,
                        }
                    },
                )
            )

    updated_count = 0
    if ops:
        print(f"committing bulk write for {len(ops)} operations")
        result = collection.bulk_write(ops, ordered=False)
        updated_count = result.modified_count

    print(f"updated documents: {updated_count}/{total_posts}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return total_posts, updated_count


def run(input_data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
    print(input_data)
    if isinstance(input_data, dict):
        raw_eval_cfg = input_data.get("evaluation")
        print(f"received evaluation config: {raw_eval_cfg}")
        if isinstance(raw_eval_cfg, dict) and raw_eval_cfg.get("enabled", False):
            evaluation_payload = run_sentiment_evaluation(context=context, evaluation_config=raw_eval_cfg)
            return {
                "previous": input_data,
                "sentiment": {
                    "mode": "evaluation",
                    "model": MODEL_NAME,
                    "evaluation": evaluation_payload,
                },
            }

    total_posts, updated_count = run_sentiment_pipeline(context=context)
    return {
        "previous": input_data,
        "sentiment": {
            "mode": "pipeline",
            "total_posts": total_posts,
            "updated_count": updated_count,
            "model": MODEL_NAME,
        },
    }


if __name__ == "__main__":
    raise RuntimeError("no.")

