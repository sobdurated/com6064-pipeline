"""
geospatial.py
-------------
Pipeline step : geospatial
Contract      : run(input_data, context) -> GeoJSON FeatureCollection dict
Author        : Mhd Adel Ugur 

Responsibilities:
  1. Extract and tag province + district for any untagged posts in posts_processed.
  2. Aggregate sentiment at BOTH province level and district level for map display.
  3. Apply fallback rules for posts with missing / unmatched locations.
  4. Compute a map-coloring score (0.0 – 1.0) per region for the frontend.
  5. Return a GeoJSON-compatible FeatureCollection.
"""

from __future__ import annotations

import re
import json
import sys
sys.stdout.reconfigure(encoding='utf-8')
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from pymongo import UpdateOne

# ─────────────────────────────────────────────────────────────────────────────
# 1.  TURKEY PROVINCE + DISTRICT REFERENCE DATA
# ─────────────────────────────────────────────────────────────────────────────
# [Keeping your exact TURKEY_LOCATIONS and PROVINCE_CODES dictionaries here]
# (Omitted for brevity in this view, but keep them in your actual file)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  LOCATION EXTRACTION (Optimized with Pre-compiled Regex)
# ─────────────────────────────────────────────────────────────────────────────

def _build_lookup(locations: dict) -> dict:
    lookup = {}
    for key, val in locations.items():
        official_province = val["official"]
        lookup[key] = (official_province, "")
        for alias in val.get("aliases", []):
            lookup[alias.lower()] = (official_province, "")
        for district in val.get("districts", []):
            d_lower = district.lower()
            if d_lower not in lookup:
                lookup[d_lower] = (official_province, district.title())
    return lookup

LOCATION_LOOKUP = _build_lookup(TURKEY_LOCATIONS)
SORTED_CANDIDATES = sorted(LOCATION_LOOKUP.keys(), key=len, reverse=True)

# Pre-compile regex patterns for performance
COMPILED_PATTERNS = {
    candidate: re.compile(r'(?<![a-zA-ZğüşıöçĞÜŞİÖÇ])' + re.escape(candidate) + r'(?![a-zA-ZğüşıöçĞÜŞİÖÇ])')
    for candidate in SORTED_CANDIDATES
}

def extract_location(text: str, post_tags: list = None) -> Tuple[str, str]:
    def scan(source: str) -> Tuple[str, str]:
        if not source:
            return "", ""
        normalized = source.lower()
        best_province = ""
        for candidate in SORTED_CANDIDATES:
            if COMPILED_PATTERNS[candidate].search(normalized):
                province, district = LOCATION_LOOKUP[candidate]
                if district:
                    return province, district
                elif not best_province:
                    best_province = province
        return best_province, ""

    for tag in (post_tags or []):
        province, district = scan(str(tag))
        if province:
            return province, district

    return scan(text)

def tag_posts_with_location(posts_col, limit: int = 0) -> int:
    query = {
        "$or": [
            {"location.province": {"$exists": False}},
            {"location.province": None},
            {"location.province": ""},
        ]
    }
    cursor = posts_col.find(query) if limit == 0 else posts_col.find(query).limit(limit)
    bulk_ops = []

    for post in cursor:
        text = post.get("text", "")
        post_tags = post.get("post_tags", [])
        province, district = extract_location(text, post_tags)

        bulk_ops.append(UpdateOne(
            {"_id": post["_id"]},
            {"$set": {"location.province": province, "location.district": district}}
        ))

    if bulk_ops:
        res = posts_col.bulk_write(bulk_ops)
        print(f"  Location tagging: {res.modified_count} posts updated.")
    return len(bulk_ops)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  FALLBACK RULES & MAP SCORING
# ─────────────────────────────────────────────────────────────────────────────

FALLBACK_PROVINCE = "Unknown"
FALLBACK_DISTRICT = "Unknown"

def apply_fallback(province: str, district: str) -> Tuple[str, str]:
    province = (province or "").strip()
    district = (district or "").strip()
    if not province:
        return FALLBACK_PROVINCE, FALLBACK_DISTRICT
    if not district:
        return province, province
    return province, district

def map_color_score(positive: int, neutral: int, negative: int) -> float:
    total = positive + neutral + negative
    if total == 0:
        return 0.5
    raw = (positive - negative) / total
    return round((raw + 1) / 2, 4)

def sentiment_label_from_score(score: float) -> str:
    if score >= 0.6: return "positive"
    if score <= 0.4: return "negative"
    return "neutral"

# ─────────────────────────────────────────────────────────────────────────────
# 4.  GEOJSON FEATURE BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def empty_bucket() -> Dict[str, Any]:
    return {"scores": [], "positive": 0, "neutral": 0, "negative": 0}

def add_post_to_bucket(bucket: Dict, label: str, score: Any) -> None:
    label = (label or "").lower().strip()
    if label == "positive": bucket["positive"] += 1
    elif label == "negative": bucket["negative"] += 1
    else: bucket["neutral"] += 1
    if isinstance(score, (int, float)):
        bucket["scores"].append(score)

def compute_stats(bucket: Dict) -> Dict[str, Any]:
    pos, neu, neg = bucket["positive"], bucket["neutral"], bucket["negative"]
    total = pos + neu + neg
    color = map_color_score(pos, neu, neg)
    return {
        "total_posts": total,
        "distribution": {"positive": pos, "neutral": neu, "negative": neg},
        "map_color_score": color,
        "sentiment_label": sentiment_label_from_score(color),
    }

def build_feature(level: str, province: str, stats: Dict, district: str = None) -> Dict:
    props = {
        "level": level,
        "province": province,
        "province_code": PROVINCE_CODES.get(province, "TR-??"),
        **stats,
    }
    if district:
        props["district"] = district
        
    return {
        "type": "Feature",
        "geometry": None,
        "properties": props
    }

# ─────────────────────────────────────────────────────────────────────────────
# 5.  PIPELINE ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run(input_data: Any, context: Dict[str, Any]) -> Dict:
    db = context["db"]
    posts_col = db["posts_processed"]
    tw_end = datetime.now(timezone.utc)

    print("  [Step 1] Tagging untagged posts with province / district ...")
    tag_posts_with_location(posts_col)

    print("  [Step 2] Grouping data for GeoJSON output ...")
    cursor = posts_col.find({}, {"location": 1, "sentiment": 1})

    province_buckets: Dict[str, Dict] = defaultdict(empty_bucket)
    district_buckets: Dict[Tuple, Dict] = defaultdict(empty_bucket)
    unknown_count = 0

    for post in cursor:
        loc = post.get("location") or {}
        sent = post.get("sentiment") or {}
        
        province, district = apply_fallback(loc.get("province", ""), loc.get("district", ""))
        if province == FALLBACK_PROVINCE: unknown_count += 1

        add_post_to_bucket(province_buckets[province], sent.get("label", ""), sent.get("score", None))
        add_post_to_bucket(district_buckets[(province, district)], sent.get("label", ""), sent.get("score", None))

    all_features = []
    for prov, bucket in province_buckets.items():
        all_features.append(build_feature("province", prov, compute_stats(bucket)))
        
    for (prov, dist), bucket in district_buckets.items():
        all_features.append(build_feature("district", prov, compute_stats(bucket), dist))

    print("  [Step 3] Building GeoJSON FeatureCollection ...")
    return {
        "type": "FeatureCollection",
        "metadata": {
            "generated_at": tw_end.isoformat(),
            "total_provinces": len(province_buckets),
            "total_districts": len(district_buckets),
            "unmatched_posts": unknown_count,
        },
        "features": all_features,
    }