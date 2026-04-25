from collections import defaultdict
import math
from datetime import datetime
from typing import Any, Dict

STEP_NAME = "sentiment_aggregation"

# =============================
# SENTIMENT MAP 
# =============================
SENTIMENT_MAP = {
    "positive": 1,
    "negative": -1
}

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

    # =============================
    # APPLY FILTERS
    # =============================
    filtered_posts = []

    for post in posts:
        text = post.get("text", "").lower()
        sentiment_label = post.get("sentiment", {}).get("label", "").lower()

        # Skip removed neutral
        if sentiment_label not in SENTIMENT_MAP:
            continue

        if FILTER_KEYWORD and FILTER_KEYWORD.lower() not in text:
            continue

        if FILTER_SENTIMENT and sentiment_label != FILTER_SENTIMENT:
            continue

        filtered_posts.append(post)

    print(f"Total posts after filtering: {len(filtered_posts)}")

    # =============================
    # GROUP BY PROVINCE
    # =============================
    groups = defaultdict(list)

    for post in filtered_posts:
        province = post.get("location", {}).get("province")
        if province:
            groups[province].append(post)

    dashboard_summary = []
    total_posts_all = 0

    # =============================
    # CALCULATE METRICS
    # =============================
    for province, p_list in groups.items():

        N_total = len(p_list)
        total_posts_all += N_total

        sentiments = [
            SENTIMENT_MAP[p["sentiment"]["label"].lower()]
            for p in p_list
        ]

        confidences = [p["sentiment"]["score"] for p in p_list]

        N_positive = sentiments.count(1)
        N_negative = sentiments.count(-1)

        pos_pct = N_positive / N_total
        neg_pct = N_negative / N_total

        avg_sent = sum(sentiments) / N_total
        polarity = (N_positive - N_negative) / N_total

        variance = sum((s - avg_sent) ** 2 for s in sentiments) / N_total
        volatility = math.sqrt(variance)

        confidence_avg = sum(confidences) / N_total
        volume_score = N_total * abs(avg_sent)

        normalized_score = (avg_sent + 1) / 2

        row = {
            "province": province,
            "post_count": N_total,
            "positive_ratio": round(pos_pct, 2),
            "negative_ratio": round(neg_pct, 2),
            "average_sentiment": round(avg_sent, 2),
            "normalized_score": round(normalized_score, 2)
        }

        dashboard_summary.append(row)

        print("\n--- PROVINCE RESULT ---")
        print(row)

        aggregates_collection.update_one(
            {"province": province, "district": "ALL"},
            {
                "$set": {
                    "province": province,
                    "district": "ALL",
                    "time_window": {
                        "start": START_DATE,
                        "end": END_DATE
                    },
                    "total_posts": N_total,
                    "average_sentiment": avg_sent,
                    "normalized_score": normalized_score,
                    "polarity": polarity,
                    "volatility": volatility,
                    "confidence": confidence_avg,
                    "volume_score": volume_score,
                    "distribution": {
                        "positive": N_positive,
                        "negative": N_negative
                    },
                    "ratios": {
                        "positive": pos_pct,
                        "negative": neg_pct
                    },
                    "last_updated": datetime.now()
                }
            },
            upsert=True
        )

    # =============================
    # DASHBOARD 
    # =============================
    if dashboard_summary:
        dashboard_cards = {
            "type": "dashboard_cards",
            "total_posts": total_posts_all,
            "avg_sentiment_score": round(
                sum(row["normalized_score"] for row in dashboard_summary) / len(dashboard_summary), 2
            ),
            "top_positive_province": max(
                dashboard_summary, key=lambda x: x["normalized_score"]
            )["province"],
            "top_negative_province": min(
                dashboard_summary, key=lambda x: x["normalized_score"]
            )["province"],
            "last_updated": datetime.now()
        }
    else:
        dashboard_cards = {}

    print("\n===== DASHBOARD CARDS =====")
    print(dashboard_cards)

    return {
        "step": STEP_NAME,
        "total_posts": total_posts_all,
        "provinces_processed": len(dashboard_summary),
        "dashboard": dashboard_cards
    }
# References:
# 1. PyMongo Documentation (MongoDB Python Driver):
#    https://pymongo.readthedocs.io/en/stable/

# 2. MongoDB Atlas Documentation:
#    https://www.mongodb.com/docs/atlas/

# 3. Python Official Documentation:
#    https://docs.python.org/3/

# 4. Sentiment Analysis Theory:
#    Pang, B., & Lee, L. (2008). Opinion Mining and Sentiment Analysis.
#    https://www.cs.cornell.edu/home/llee/omsa/omsa.pdf

# 5. Data Aggregation & Analytics Concepts:
#    Han, J., Kamber, M., & Pei, J. (2011). Data Mining: Concepts and Techniques.

# Description:
# This module is part of a multi-step data processing pipeline.
# It uses a shared MongoDB connection provided by the pipeline context,
# retrieves processed posts, applies filtering (date, keyword, sentiment),
# aggregates sentiment data at the province level, and computes metrics
# such as average sentiment, polarity, volatility, confidence score,
# and normalized sentiment score.
# The module generates dashboard-ready outputs and stores results in:
# (1) sentiment_aggregates collection for province-level metrics
