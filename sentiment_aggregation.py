from collections import defaultdict
import math
from datetime import datetime
from typing import Any, Dict

STEP_NAME = "sentiment_aggregation"

SENTIMENT_MAP = {
    "positive": 1,
    "negative": -1
}

MODEL_TYPES = ["llm", "transformer"]

def run(input_data: Any, context: Dict[str, Any]) -> Any:
    print(f"running step: {STEP_NAME}")

    db = context["db"]
    posts_collection = db["posts_processed"]
    aggregates_collection = db["sentiment_aggregates"]

    # =============================
    # FILTERS 
    # =============================
    START_DATE = datetime(1950, 1, 1)
    END_DATE = datetime(2026, 12, 31)
    #policy = politika
    #tourism = turizm
    #economy = ekonomi
    #weather = hava
    FILTER_KEYWORD = None
    #positive/negative
    FILTER_SENTIMENT = None

    # =============================
    # FETCH DATA
    # =============================
    query = {
        "created_at": {"$gte": START_DATE, "$lte": END_DATE}
    }

    posts = list(posts_collection.find(query))
    print(f"\nTotal posts fetched: {len(posts)}")

    # =============================
    # PROCESS EACH MODEL
    # =============================
    for model_type in MODEL_TYPES:

        print(f"\n===== MODEL: {model_type.upper()} =====")

        filtered_posts = []

        for post in posts:
            text = post.get("text", "").lower()
            sentiment_data = post.get("sentiment", {}).get(model_type, {})
            label = sentiment_data.get("label", "").lower()

            if label not in SENTIMENT_MAP:
                continue

            if FILTER_KEYWORD and FILTER_KEYWORD.lower() not in text:
                continue

            if FILTER_SENTIMENT and label != FILTER_SENTIMENT:
                continue

            filtered_posts.append(post)

        print(f"Filtered posts: {len(filtered_posts)}")

        # =============================
        # GROUP BY PROVINCE
        # =============================
        groups = defaultdict(list)

        for post in filtered_posts:
            province = post.get("location", {}).get("province")
            if province:
                groups[province].append(post)

        # =============================
        # CALCULATE 
        # =============================
        for province, p_list in groups.items():

            N_total = len(p_list)

            sentiments = [
                SENTIMENT_MAP[p["sentiment"][model_type]["label"].lower()]
                for p in p_list
            ]

            N_positive = sentiments.count(1)
            N_negative = sentiments.count(-1)

            pos_pct = N_positive / N_total
            neg_pct = N_negative / N_total

            avg_sent = sum(sentiments) / N_total
            normalized_score = (avg_sent + 1) / 2

            document = {
                "province": province,
                "district": "ALL",
                "model_type": model_type,
                "time_window": {
                    "start": START_DATE,
                    "end": END_DATE
                },
                "total_posts": N_total,
                "average_sentiment": round(avg_sent, 2),
                "normalized_score": round(normalized_score, 2),
                "distribution": {
                    "positive": N_positive,
                    "negative": N_negative
                },
                "ratios": {
                    "positive": round(pos_pct, 2),
                    "negative": round(neg_pct, 2)
                },
                "last_updated": datetime.now()
            }

           
            print("\n--- RESULT ---")
            print(document)

            aggregates_collection.update_one(
                {
                    "province": province,
                    "district": "ALL",
                    "model_type": model_type
                },
                {"$set": document},
                upsert=True
            )

            print(f"Updated: {province} ({model_type})")

    print("\n✅ All data processed successfully!")

    return {
        "step": STEP_NAME,
        "status": "completed"
    }

# References:
# 1. https://pymongo.readthedocs.io/en/stable/
# 2. https://www.mongodb.com/docs/atlas/
# 3. https://docs.python.org/3/

# Description:
# This pipeline step retrieves posts from MongoDB and aggregates
# sentiment data per province for both llm and transformer models.
# It applies filtering (date, keyword, sentiment), computes metrics,
# and stores results in sentiment_aggregates using model_type.