import gc
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


def run_sentiment_pipeline(context: Dict[str, Any]) -> Tuple[int, int]:
    device = 0 if torch.cuda.is_available() else -1
    db = context["db"]
    collection = db[SOURCE_COLLECTION]

    print(f"database: {db.name} | collection: {SOURCE_COLLECTION}")

    print(f"loading model: {MODEL_NAME}")

    sentiment_pipe = pipeline(
        task="text-classification",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        truncation=True,
        max_length=MAX_TEXT_LEN,
        device=device,
    )

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
    total_posts, updated_count = run_sentiment_pipeline(context=context)
    return {
        "previous": input_data,
        "sentiment": {
            "total_posts": total_posts,
            "updated_count": updated_count,
            "model": MODEL_NAME,
        },
    }


if __name__ == "__main__":
    raise RuntimeError("run this step via pipeline.py so it can use the shared context['db']")

