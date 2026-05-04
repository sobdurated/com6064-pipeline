"""
test_geospatial.py
------------------
Automated pytest suite for the Geospatial Module (Adel Ugur)
Tests location extraction, fallback logic, map scoring, and multi-model schema integration.
"""

import pytest
from geospatial import (
    extract_location,
    apply_fallback,
    map_color_score,
    sentiment_label_from_score,
    empty_bucket,
    add_post_to_bucket,
    compute_stats,
    FALLBACK_PROVINCE,
    FALLBACK_DISTRICT
)

# ==========================================
# TC-GEO-01: Location Extraction & Tagging
# ==========================================
def test_extract_location_from_text():
    """Verify province extraction from raw text."""
    text = "Üç günlük yeme içme turumu yeni tamamladım, en az otuz mekanlık bir liste... Adana"
    province, district = extract_location(text, post_tags=[])
    assert province == "Adana"
    assert district == ""

def test_extract_location_from_tags():
    """Verify exact district extraction prioritizing tags."""
    tags = ["yemek", "kadıköy", "istanbul"]
    province, district = extract_location("", post_tags=tags)
    assert province == "İstanbul"
    assert district == "Kadıköy"

# ==========================================
# TC-GEO-02: Fallback Logic Validation
# ==========================================
def test_apply_fallback_missing_district():
    """Verify that a missing district defaults to the province name."""
    province, district = apply_fallback("Adana", "")
    assert province == "Adana"
    assert district == "Adana"

def test_apply_fallback_completely_missing():
    """Verify completely unknown locations hit the global fallback."""
    province, district = apply_fallback("", None)
    assert province == FALLBACK_PROVINCE
    assert district == FALLBACK_DISTRICT

# ==========================================
# TC-GEO-03: Sentiment Math & Map Scoring
# ==========================================
def test_map_color_score():
    """Verify color score math (1.0 = purely positive, 0.0 = purely negative)."""
    assert map_color_score(positive=100, neutral=0, negative=0) == 1.0
    assert map_color_score(positive=0, neutral=0, negative=100) == 0.0
    assert map_color_score(positive=50, neutral=0, negative=50) == 0.5
    assert map_color_score(positive=0, neutral=100, negative=0) == 0.5

def test_sentiment_label_from_score():
    """Verify score thresholds assign the correct text label."""
    assert sentiment_label_from_score(0.85) == "positive"
    assert sentiment_label_from_score(0.20) == "negative"
    assert sentiment_label_from_score(0.50) == "neutral"

# ==========================================
# TC-GEO-04 & 05: Multi-Model Schema Bucketing
# ==========================================
def test_add_post_to_bucket_multi_model():
    """Verify the bucket accurately processes separate LLM and Transformer sentiments and categories."""
    bucket = empty_bucket()
    
    mock_sentiment = {
        "llm": {"label": "negative", "score": 0.9},
        "transformer": {"label": "neutral", "score": 0.0}
    }
    
    add_post_to_bucket(bucket, mock_sentiment, category="yemek")
    
    # Check LLM updates
    assert bucket["llm"]["negative"] == 1
    assert bucket["llm"]["positive"] == 0
    assert "yemek" in bucket["llm"]["categories"]
    assert bucket["llm"]["categories"]["yemek"]["negative"] == 1

    # Check Transformer updates
    assert bucket["transformer"]["neutral"] == 1
    assert bucket["transformer"]["negative"] == 0
    assert bucket["transformer"]["categories"]["yemek"]["neutral"] == 1

def test_compute_stats_output():
    """Verify bucket statistics accurately format into the final Kaan-ready GeoJSON properties."""
    bucket = empty_bucket()
    
    # Add two posts to simulate some data
    add_post_to_bucket(bucket, {"llm": {"label": "positive", "score": 0.8}, "transformer": {"label": "neutral", "score": 0.0}}, "ulaşım")
    add_post_to_bucket(bucket, {"llm": {"label": "positive", "score": 0.7}, "transformer": {"label": "neutral", "score": 0.0}}, "ulaşım")
    
    stats = compute_stats(bucket)
    
    # Assert LLM Stats calculated correctly
    assert stats["llm"]["total_posts"] == 2
    assert stats["llm"]["map_color_score"] == 1.0  # 100% positive
    assert stats["llm"]["sentiment_label"] == "positive"
    assert stats["llm"]["categories"]["ulaşım"]["total"] == 2
    
    # Assert Transformer Stats remain neutral
    assert stats["transformer"]["total_posts"] == 2
    assert stats["transformer"]["map_color_score"] == 0.5  # 100% neutral
    assert stats["transformer"]["sentiment_label"] == "neutral"