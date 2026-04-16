"""
geospatial.py
-------------
Pipeline step : geospatial
Contract      : run(input_data, context) -> GeoJSON FeatureCollection dict

Responsibilities:
  1. Read location-tagged posts from posts_processed (via shared DB context).
  2. Aggregate sentiment at BOTH province level and district level.
  3. Apply fallback rules for posts with missing / unmatched locations.
  4. Compute a map-coloring score (0.0 – 1.0) per region for the frontend.
  5. Return a GeoJSON-compatible FeatureCollection (no real coordinates —
     the frontend map layer attaches boundaries by matching province/district
     name or code).
  6. Upsert results into sentiment_aggregates.

Plugs into pipeline.py as:
    ("geospatial", "geospatial.py")
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from pymongo import UpdateOne

# ─────────────────────────────────────────────────────────────────────────────
# 1.  TURKEY PROVINCE REFERENCE  (name → ISO-3166-2 style code)
#     Codes follow the TR-XX convention used by most GeoJSON boundary datasets.
#     The frontend map layer matches features by "province_code" or "province".
# ─────────────────────────────────────────────────────────────────────────────

PROVINCE_CODES: Dict[str, str] = {
    "Adana": "TR-01", "Adıyaman": "TR-02", "Afyonkarahisar": "TR-03",
    "Ağrı": "TR-04", "Aksaray": "TR-68", "Amasya": "TR-05",
    "Ankara": "TR-06", "Antalya": "TR-07", "Ardahan": "TR-75",
    "Artvin": "TR-08", "Aydın": "TR-09", "Balıkesir": "TR-10",
    "Bartın": "TR-74", "Batman": "TR-72", "Bayburt": "TR-69",
    "Bilecik": "TR-11", "Bingöl": "TR-12", "Bitlis": "TR-13",
    "Bolu": "TR-14", "Burdur": "TR-15", "Bursa": "TR-16",
    "Çanakkale": "TR-17", "Çankırı": "TR-18", "Çorum": "TR-19",
    "Denizli": "TR-20", "Diyarbakır": "TR-21", "Düzce": "TR-81",
    "Edirne": "TR-22", "Elazığ": "TR-23", "Erzincan": "TR-24",
    "Erzurum": "TR-25", "Eskişehir": "TR-26", "Gaziantep": "TR-27",
    "Giresun": "TR-28", "Gümüşhane": "TR-29", "Hakkari": "TR-30",
    "Hatay": "TR-31", "Iğdır": "TR-76", "Isparta": "TR-32",
    "İstanbul": "TR-34", "İzmir": "TR-35", "Kahramanmaraş": "TR-46",
    "Karabük": "TR-78", "Karaman": "TR-70", "Kars": "TR-36",
    "Kastamonu": "TR-37", "Kayseri": "TR-38", "Kilis": "TR-79",
    "Kırıkkale": "TR-71", "Kırklareli": "TR-39", "Kırşehir": "TR-40",
    "Kocaeli": "TR-41", "Konya": "TR-42", "Kütahya": "TR-43",
    "Malatya": "TR-44", "Manisa": "TR-45", "Mardin": "TR-47",
    "Mersin": "TR-33", "Muğla": "TR-48", "Muş": "TR-49",
    "Nevşehir": "TR-50", "Niğde": "TR-51", "Ordu": "TR-52",
    "Osmaniye": "TR-80", "Rize": "TR-53", "Sakarya": "TR-54",
    "Samsun": "TR-55", "Şanlıurfa": "TR-63", "Siirt": "TR-56",
    "Sinop": "TR-57", "Şırnak": "TR-73", "Sivas": "TR-58",
    "Tekirdağ": "TR-59", "Tokat": "TR-60", "Trabzon": "TR-61",
    "Tunceli": "TR-62", "Uşak": "TR-64", "Van": "TR-65",
    "Yalova": "TR-77", "Yozgat": "TR-66", "Zonguldak": "TR-67",
}

# ─────────────────────────────────────────────────────────────────────────────
# 2.  FALLBACK RULES
# ─────────────────────────────────────────────────────────────────────────────

FALLBACK_PROVINCE = "Unknown"
FALLBACK_DISTRICT = "Unknown"

def apply_fallback(province: str, district: str) -> Tuple[str, str]:
    """
    Fallback rules for unmatched / empty location fields.
      - Empty province  → "Unknown"
      - Valid province but empty district → province name used as district too
        (province-level post, no district granularity available)
      - Both empty → both "Unknown"
    """
    province = (province or "").strip()
    district = (district or "").strip()

    if not province:
        return FALLBACK_PROVINCE, FALLBACK_DISTRICT

    if not district:
        return province, province     # district defaults to province name

    return province, district

# ─────────────────────────────────────────────────────────────────────────────
# 3.  MAP-COLORING SCORE
# ─────────────────────────────────────────────────────────────────────────────

def map_color_score(positive: int, neutral: int, negative: int) -> float:
    """
    Returns a float in [0.0, 1.0] representing overall sentiment polarity.

      0.0  = fully negative
      0.5  = fully neutral  (or equal positive / negative)
      1.0  = fully positive

    Formula:  score = (positive - negative) / total, rescaled to [0, 1]
    Falls back to 0.5 for empty buckets.
    """
    total = positive + neutral + negative
    if total == 0:
        return 0.5
    raw = (positive - negative) / total      # range: [-1, 1]
    return round((raw + 1) / 2, 4)           # rescale to [0, 1]


def sentiment_label_from_score(score: float) -> str:
    """Convert a map-coloring score to a human-readable label."""
    if score >= 0.6:
        return "positive"
    if score <= 0.4:
        return "negative"
    return "neutral"

# ─────────────────────────────────────────────────────────────────────────────
# 4.  AGGREGATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def empty_bucket() -> Dict[str, Any]:
    return {"scores": [], "positive": 0, "neutral": 0, "negative": 0}


def add_post_to_bucket(bucket: Dict, label: str, score: Any) -> None:
    label = (label or "").lower().strip()
    if label == "positive":
        bucket["positive"] += 1
    elif label == "negative":
        bucket["negative"] += 1
    else:
        bucket["neutral"] += 1
    if isinstance(score, (int, float)):
        bucket["scores"].append(score)


def compute_stats(bucket: Dict) -> Dict[str, Any]:
    pos   = bucket["positive"]
    neu   = bucket["neutral"]
    neg   = bucket["negative"]
    total = pos + neu + neg
    avg   = round(sum(bucket["scores"]) / len(bucket["scores"]), 4) if bucket["scores"] else 0.0
    ratio = lambda n: round(n / total, 4) if total else 0.0
    color = map_color_score(pos, neu, neg)
    return {
        "total_posts":       total,
        "average_sentiment": avg,
        "distribution":      {"positive": pos, "neutral": neu, "negative": neg},
        "ratios":            {"positive": ratio(pos), "neutral": ratio(neu), "negative": ratio(neg)},
        "map_color_score":   color,
        "sentiment_label":   sentiment_label_from_score(color),
    }

# ─────────────────────────────────────────────────────────────────────────────
# 5.  GEOJSON FEATURE BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def province_feature(province: str, stats: Dict) -> Dict:
    """
    GeoJSON Feature for a province.
    geometry is null — the frontend attaches real boundaries by matching
    'province_code' or 'province' against its boundary dataset.
    """
    return {
        "type": "Feature",
        "geometry": None,               # frontend fills in from boundary file
        "properties": {
            "level":           "province",
            "province":        province,
            "province_code":   PROVINCE_CODES.get(province, "TR-??"),
            **stats,
        }
    }


def district_feature(province: str, district: str, stats: Dict) -> Dict:
    """
    GeoJSON Feature for a district.
    geometry is null — same reason as above.
    """
    return {
        "type": "Feature",
        "geometry": None,
        "properties": {
            "level":         "district",
            "province":      province,
            "province_code": PROVINCE_CODES.get(province, "TR-??"),
            "district":      district,
            **stats,
        }
    }

# ─────────────────────────────────────────────────────────────────────────────
# 6.  DB UPSERT
# ─────────────────────────────────────────────────────────────────────────────

def upsert_aggregates(collection, features: List[Dict],
                      tw_start: datetime, tw_end: datetime) -> None:
    """Upsert each feature's stats into sentiment_aggregates."""
    bulk_ops = []
    for feat in features:
        props = feat["properties"]
        level = props["level"]

        filter_doc = {
            "province":          props["province"],
            "time_window.start": tw_start,
            "time_window.end":   tw_end,
        }
        if level == "district":
            filter_doc["district"] = props["district"]

        set_doc = {
            "province":          props["province"],
            "province_code":     props["province_code"],
            "district":          props.get("district", ""),
            "level":             level,
            "time_window":       {"start": tw_start, "end": tw_end},
            "total_posts":       props["total_posts"],
            "average_sentiment": props["average_sentiment"],
            "distribution":      props["distribution"],
            "ratios":            props["ratios"],
            "map_color_score":   props["map_color_score"],
            "sentiment_label":   props["sentiment_label"],
        }

        bulk_ops.append(UpdateOne(filter_doc, {"$set": set_doc}, upsert=True))

    if bulk_ops:
        res = collection.bulk_write(bulk_ops)
        print(f"  DB: {res.upserted_count} inserted, {res.modified_count} updated "
              f"in sentiment_aggregates.")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  PIPELINE ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run(input_data: Any, context: Dict[str, Any]) -> Dict:
    """
    Pipeline contract: run(input_data, context) -> GeoJSON FeatureCollection

    context keys used:
        context["db"]  — pymongo Database object (provided by pipeline.py)
    """
    db                   = context["db"]
    posts_col            = db["posts_processed"]
    aggregates_col       = db["sentiment_aggregates"]

    tw_start = datetime(2000, 1, 1, tzinfo=timezone.utc)
    tw_end   = datetime.now(timezone.utc)

    # ── 7a. Read all posts that have been through location tagging ──────────
    cursor = posts_col.find(
        {},
        {
            "location.province": 1,
            "location.district": 1,
            "sentiment.label":   1,
            "sentiment.score":   1,
        }
    )

    province_buckets: Dict[str, Dict]         = defaultdict(empty_bucket)
    district_buckets: Dict[Tuple, Dict]       = defaultdict(empty_bucket)
    unknown_count = 0

    for post in cursor:
        raw_province = (post.get("location") or {}).get("province", "")
        raw_district = (post.get("location") or {}).get("district", "")
        label        = (post.get("sentiment") or {}).get("label", "")
        score        = (post.get("sentiment") or {}).get("score", None)

        province, district = apply_fallback(raw_province, raw_district)

        if province == FALLBACK_PROVINCE:
            unknown_count += 1

        add_post_to_bucket(province_buckets[province], label, score)
        add_post_to_bucket(district_buckets[(province, district)], label, score)

    print(f"  Posts read: {sum(b['positive']+b['neutral']+b['negative'] for b in province_buckets.values())}")
    print(f"  Unmatched (no location): {unknown_count}")
    print(f"  Provinces found: {len(province_buckets)}")
    print(f"  Districts found: {len(district_buckets)}")

    # ── 7b. Build GeoJSON features ──────────────────────────────────────────
    province_features = []
    for province, bucket in sorted(province_buckets.items()):
        stats = compute_stats(bucket)
        feat  = province_feature(province, stats)
        province_features.append(feat)
        print(f"  [province] {province:<22} | posts={stats['total_posts']:>4} | "
              f"color={stats['map_color_score']:.3f} | {stats['sentiment_label']}")

    district_features = []
    for (province, district), bucket in sorted(district_buckets.items()):
        stats = compute_stats(bucket)
        feat  = district_feature(province, district, stats)
        district_features.append(feat)

    all_features = province_features + district_features

    # ── 7c. Upsert into sentiment_aggregates ────────────────────────────────
    upsert_aggregates(aggregates_col, all_features, tw_start, tw_end)

    # ── 7d. Build and return GeoJSON FeatureCollection ──────────────────────
    geojson_output = {
        "type":     "FeatureCollection",
        "metadata": {
            "generated_at":      tw_end.isoformat(),
            "total_provinces":   len(province_features),
            "total_districts":   len(district_features),
            "unmatched_posts":   unknown_count,
        },
        "features": all_features,
    }

    return geojson_output


# ─────────────────────────────────────────────────────────────────────────────
# 8.  STANDALONE RUNNER  (python geospatial.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pymongo import MongoClient
    import os

    MONGO_URI    = os.getenv("MONGO_URI",
        "mongodb+srv://COM6064:OnTHCZcqye91Yv1s@cluster0.bj4tnnh.mongodb.net/COM6064?appName=Cluster0")
    MONGO_DB     = os.getenv("MONGO_DB_NAME", "COM6064")

    client  = MongoClient(MONGO_URI)
    context = {"db": client[MONGO_DB]}

    print("=" * 60)
    print("  geospatial.py — standalone run")
    print("=" * 60)

    result = run(None, context)
    client.close()

    print("\n" + "=" * 60)
    print("  GeoJSON OUTPUT")
    print("=" * 60)
    print(json.dumps(result, ensure_ascii=False, indent=2))
