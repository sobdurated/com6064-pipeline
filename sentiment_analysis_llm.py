import gc
import csv
import os
import re
from typing import Any, Dict, Iterable, List, Tuple

import torch
from pymongo import UpdateOne
from transformers import AutoTokenizer, AutoModelForCausalLM

SOURCE_COLLECTION = os.getenv("MONGO_SOURCE_COLLECTION", "posts_processed")
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
BATCH_SIZE = int(os.getenv("SENTIMENT_BATCH_SIZE", "8"))
MAX_NEW_TOKENS = 8

SYSTEM_PROMPT = (
    "You are a sentiment classification assistant. "
    "Classify the sentiment of the given text.\n\n"
    "Rules:\n"
    "- Reply with ONLY one word: positive OR negative\n"
    "- No punctuation, no explanation, no other words\n"
    "- If the text is ambiguous, pick the closest match\n"
    "- Respond in English regardless of the input language"
)


def _build_prompt(tokenizer, text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Text: {text}"},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def _normalize_label(raw_text: str, fallback_score: float = 0.0) -> Tuple[str, float]:
    cleaned = raw_text.strip().lower()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    first_word = cleaned.split()[0] if cleaned.split() else ""

    if "negative" in first_word:
        return "negative", 0.90
    if "positive" in first_word:
        return "positive", 0.90

    if "negative" in cleaned:
        return "negative", 0.75
    if "positive" in cleaned:
        return "positive", 0.75

    return ("positive" if fallback_score >= 0.5 else "negative"), 0.50


def _build_sentiment_pipeline():
    print(f"loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model.eval()
    print(f"model loaded | device map: {model.device}")
    return tokenizer, model


def _chunked(records: List[Dict], size: int) -> Iterable[List[Dict]]:
    for idx in range(0, len(records), size):
        yield records[idx : idx + size]


def _infer_batch(tokenizer, model, texts: List[str]) -> List[Tuple[str, float]]:
    prompts = [_build_prompt(tokenizer, text) for text in texts]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    results = []
    input_len = inputs["input_ids"].shape[1]
    for output_ids in outputs:
        new_tokens = output_ids[input_len:]
        raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        label, score = _normalize_label(raw_text)
        results.append((label, score))

    return results


def _load_posts_for_inference(collection) -> List[Dict]:
    query = {
        "text": {"$type": "string", "$ne": ""},
        "$or": [
            {"sentiment.llm.model": "pending"},
            {"sentiment.llm.label": "neutral"},
            {"sentiment.llm.model": "savasy/bert-base-turkish-sentiment-cased"},
        ],
    }
    projection = {"_id": 1, "text": 1}
    return list(collection.find(query, projection))


def _resolve_dataset_path(dataset_path: str, context: Dict[str, Any]) -> str:
    raw_path = (dataset_path or "").strip()
    if not raw_path:
        raise ValueError("evaluation.dataset_path must be provided")
    if os.path.isabs(raw_path):
        return raw_path
    base_dir = str(context.get("base_dir") or "")
    return os.path.join(base_dir, raw_path) if base_dir else raw_path


def _load_labeled_samples(
    csv_path: str, text_column: str, label_column: str, id_column: str
) -> List[Dict[str, str]]:
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
            normalized_true, _ = _normalize_label(raw_label)

            if not text or normalized_true not in {"positive", "negative"}:
                continue

            samples.append({
                "entry_id": str(row.get(id_column, "") or ""),
                "text": text,
                "true_label": normalized_true,
            })
    return samples


def _safe_ratio(n: int, d: int) -> float:
    return 0.0 if d == 0 else n / d


def _compute_metrics(true_labels: List[str], pred_labels: List[str]) -> Dict[str, Any]:
    tp = tn = fp = fn = 0
    for t, p in zip(true_labels, pred_labels):
        if t == "positive" and p == "positive":
            tp += 1
        elif t == "negative" and p == "negative":
            tn += 1
        elif t == "negative" and p == "positive":
            fp += 1
        elif t == "positive" and p == "negative":
            fn += 1

    total = len(true_labels)
    accuracy = _safe_ratio(tp + tn, total)
    precision_pos = _safe_ratio(tp, tp + fp)
    recall_pos = _safe_ratio(tp, tp + fn)
    f1_pos = _safe_ratio(2 * precision_pos * recall_pos, precision_pos + recall_pos)
    precision_neg = _safe_ratio(tn, tn + fn)
    recall_neg = _safe_ratio(tn, tn + fp)
    f1_neg = _safe_ratio(2 * precision_neg * recall_neg, precision_neg + recall_neg)

    return {
        "support": total,
        "accuracy": accuracy,
        "confusion_matrix": {"true_negative": tn, "false_positive": fp, "false_negative": fn, "true_positive": tp},
        "per_class": {
            "positive": {"precision": precision_pos, "recall": recall_pos, "f1": f1_pos, "support": tp + fn},
            "negative": {"precision": precision_neg, "recall": recall_neg, "f1": f1_neg, "support": tn + fp},
        },
        "macro_avg": {
            "precision": (precision_pos + precision_neg) / 2,
            "recall": (recall_pos + recall_neg) / 2,
            "f1": (f1_pos + f1_neg) / 2,
        },
    }


def run_sentiment_evaluation(context: Dict[str, Any], evaluation_config: Dict[str, Any]) -> Dict[str, Any]:
    dataset_path = _resolve_dataset_path(str(evaluation_config.get("dataset_path", "")), context)
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
            "error_analysis": {"total_misclassified": 0, "misclassified_examples": [], "hardest_errors": []},
        }

    print(f"evaluation samples: {len(samples)}")
    tokenizer, model = _build_sentiment_pipeline()

    results: List[Dict[str, Any]] = []
    for batch in _chunked(samples, batch_size):
        texts = [item["text"] for item in batch]
        predictions = _infer_batch(tokenizer, model, texts)

        for item, (pred_label, score) in zip(batch, predictions):
            results.append({
                "entry_id": item["entry_id"],
                "text": item["text"],
                "true_label": item["true_label"],
                "predicted_label": pred_label,
                "score": score,
                "is_error": item["true_label"] != pred_label,
            })

    true_labels = [r["true_label"] for r in results]
    pred_labels = [r["predicted_label"] for r in results]
    metrics = _compute_metrics(true_labels, pred_labels)

    misclassified = [r for r in results if r["is_error"]]
    misclassified_sorted = sorted(misclassified, key=lambda r: r["score"], reverse=True)

    def _trim(items, limit):
        return [
            {"entry_id": r["entry_id"], "true_label": r["true_label"],
             "predicted_label": r["predicted_label"], "score": r["score"], "text": r["text"][:240]}
            for r in items[:limit]
        ]

    print(
        f"evaluation summary | samples={metrics.get('support', 0)} "
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
            "misclassified_examples": _trim(misclassified, error_limit),
            "hardest_errors": _trim(misclassified_sorted, error_limit),
        },
    }


def run_sentiment_pipeline(context: Dict[str, Any]) -> Tuple[int, int]:
    db = context["db"]
    collection = db[SOURCE_COLLECTION]

    print(f"database: {db.name} | collection: {SOURCE_COLLECTION}")
    tokenizer, model = _build_sentiment_pipeline()

    posts = _load_posts_for_inference(collection)
    total_posts = len(posts)

    if total_posts == 0:
        print("no documents found with non-empty text in posts_processed")
        return 0, 0

    print(f"documents to process: {total_posts}")

    ops: List[UpdateOne] = []
    done = 0
    for post_batch in _chunked(posts, BATCH_SIZE):
        texts = [doc["text"] for doc in post_batch]
        predictions = _infer_batch(tokenizer, model, texts)

        for doc, (sentiment_label, score) in zip(post_batch, predictions):
            print(f"-------------\npost_id: {doc['_id']}\npost content: {doc['text']}\nscore: {score:.4f}\nlabel: {sentiment_label}\n")
            ops.append(
                UpdateOne(
                    {"_id": doc["_id"]},
                    {"$set": {
                        "sentiment.llm.label": sentiment_label,
                        "sentiment.llm.score": score,
                        "sentiment.llm.model": MODEL_NAME,
                    }},
                )
            )
        if ops:
            print(f"committing bulk write for {len(ops)} operations")
            result = collection.bulk_write(ops, ordered=False)
            # updated_count = result.modified_count
            done += result.modified_count
            print(f"{done}/{total_posts}; remaining {total_posts - done}")
            ops = []

    print(f"updated documents: {total_posts}/{total_posts}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return total_posts

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

    total_posts = run_sentiment_pipeline(context=context)
    return {
        "previous": input_data,
        "sentiment": {
            "mode": "pipeline",
            "total_posts": total_posts,
            "model": MODEL_NAME,
        },
    }


if __name__ == "__main__":
    raise RuntimeError("no.")
