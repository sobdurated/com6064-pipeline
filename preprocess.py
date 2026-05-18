"""
Pipeline Step: Turkish Text Preprocessing, Anonymization & Validation
Generates a detailed execution report (.json) per run.
Compliant with pipeline.py: uses run(input_data, context) and context['db'].
"""
import re
import os
import json
import emoji
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from pymongo import UpdateOne
import nltk

# --- CONFIGURATION ---
REPORT_DIR = "pipeline_reports"
REPORT_MAX_ENTRIES = 500  # Cap detailed logs to prevent memory bloat

# --- NLTK SETUP ---
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

TR_STOPWORDS = set(w.lower() for w in stopwords.words('turkish'))

# ==============================================================================
# 1. MODULAR CLEANING & ANONYMIZATION ENGINE
# ==============================================================================
class TurkishTextCleaner:
    """Modular, ethically-aware text processor with configurable rules."""
    def __init__(self, config: Optional[Dict[str, bool]] = None):
        self.config = config or {
            "remove_urls": True,
            "remove_mentions": True,
            "remove_hashtags": False,
            "replace_emojis": True,
            "anonymize_pii": True,
            "normalize_turkish": True,
            "filter_stopwords": True,
            "strip_punctuation": True,
            "min_valid_length": 3
        }
        self._compile_patterns()

    def _compile_patterns(self):
        self.RE_URL = re.compile(r'https?://\S+|www\.\S+')
        self.RE_MENTION = re.compile(r'@\w+')
        self.RE_HASHTAG = re.compile(r'#\w+')
        self.RE_REPEAT = re.compile(r'(.)\1{2,}')
        self.RE_PUNCT = re.compile(r'[^\w\s]') if self.config.get("strip_punctuation") else None
        self.RE_EMAIL = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')
        self.RE_PHONE_TR = re.compile(r'\b(?:\+90|0)?\s?(?:5\d{2})\s?\d{3}\s?\d{2}\s?\d{2}\b')
        self.RE_TCKN = re.compile(r'\b\d{11}\b')
        self.RE_USERNAME = re.compile(r'\b(?:[uU]ser(?:name)?\s*[:=]?\s*\w+|[uU]ser\s+\w+)\b')

    def anonymize(self, text: str) -> str:
        if not self.config.get("anonymize_pii"): return text
        text = self.RE_EMAIL.sub('[EMAIL]', text)
        text = self.RE_PHONE_TR.sub('[PHONE]', text)
        text = self.RE_TCKN.sub('[ID]', text)
        text = self.RE_USERNAME.sub('[USERNAME]', text)
        return text

    def validate(self, text: Optional[str]) -> bool:
        if not isinstance(text, str): return False
        return len(text.strip()) >= self.config.get("min_valid_length", 3)

    def clean(self, raw_text: Any) -> Optional[str]:
        if not isinstance(raw_text, str): return None
        text = self.anonymize(raw_text)
        if self.config.get("remove_urls"): text = self.RE_URL.sub('', text)
        if self.config.get("remove_mentions"): text = self.RE_MENTION.sub('', text)
        if self.config.get("remove_hashtags"): text = self.RE_HASHTAG.sub('', text)
        if self.config.get("replace_emojis"): text = emoji.replace_emoji(text, replace=' ')
        if self.config.get("normalize_turkish"): text = text.replace('İ', 'i').replace('I', 'ı').lower()
        if self.RE_PUNCT: text = self.RE_PUNCT.sub(' ', text)
        text = self.RE_REPEAT.sub(r'\1', text)
        text = re.sub(r'\s+', ' ', text).strip()
        if self.config.get("filter_stopwords") and TR_STOPWORDS and text:
            text = ' '.join([w for w in text.split() if w not in TR_STOPWORDS])
        return text if self.validate(text) else None


# ==============================================================================
# 2. PIPELINE CONTRACT & REPORTING
# ==============================================================================
def _save_report(report: Dict[str, Any], step_name: str) -> str:
    """Saves execution report to pipeline_reports/ directory."""
    Path(REPORT_DIR).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{step_name}_report_{timestamp}.json"
    filepath = os.path.join(REPORT_DIR, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"📄 Execution report saved to: {filepath}")
    return filepath

def run(input_data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point for the pipeline.
    Args:
        input_data: Payload from previous step.
        context: Shared runtime context (contains 'db').
    Returns:
        Dict: Summary of processing and report path.
    """
    STEP_NAME = "text_preprocessing"
    print(f"🚀 Running step: {STEP_NAME}")

    # 1. Access shared DB connection (DO NOT create MongoClient)
    db = context.get("db")
    if db is None:
        raise ValueError("Database connection not found in context. Use context['db'].")

    posts_raw = db.posts_raw
    posts_processed = db.posts_processed
    posts_processed.create_index("post_id", unique=True)

    # Load processed IDs to avoid duplicate work
    print("🔍 Loading processed post IDs...")
    processed_ids = set(doc["post_id"] for doc in posts_processed.find({}, {"post_id": 1}))
    print(f"📋 {len(processed_ids)} posts already processed.")

    # Initialize modular cleaner
    cleaner = TurkishTextCleaner(config={
        "remove_urls": True, "remove_mentions": True, "remove_hashtags": False,
        "replace_emojis": True, "anonymize_pii": True, "normalize_turkish": True,
        "filter_stopwords": True, "strip_punctuation": True, "min_valid_length": 3
    })

    # Initialize Execution Report
    execution_report = {
        "step": STEP_NAME,
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "config_applied": cleaner.config,
        "summary": {
            "total_scanned": 0,
            "successfully_processed": 0,
            "skipped_already_processed": 0,
            "skipped_invalid_text": 0
        },
        "processed_sample": [],
        "skipped_sample": [],
        "report_file_path": ""
    }

    batch_size = 500
    # Fetch only necessary fields to reduce memory usage
    cursor = posts_raw.find({}, {"_id": 1, "text": 1, "post_tags": 1, "created_at": 1}).batch_size(batch_size)
    
    start_time = time.time()
    last_log = time.time()
    batch_to_process = []

    for raw_doc in cursor:
        post_id = raw_doc["_id"]
        execution_report["summary"]["total_scanned"] += 1

        # Skip if already processed
        if post_id in processed_ids:
            execution_report["summary"]["skipped_already_processed"] += 1
            if len(execution_report["skipped_sample"]) < REPORT_MAX_ENTRIES:
                execution_report["skipped_sample"].append({
                    "post_id": str(post_id), 
                    "reason": "already_in_processed_collection"
                })
            continue

        # Clean and Validate
        cleaned_text = cleaner.clean(raw_doc.get("text", ""))
        if cleaned_text is None:
            execution_report["summary"]["skipped_invalid_text"] += 1
            if len(execution_report["skipped_sample"]) < REPORT_MAX_ENTRIES:
                execution_report["skipped_sample"].append({
                    "post_id": str(post_id), 
                    "reason": "empty_or_below_min_length"
                })
            continue

        # Extract location (Index 1 = Province, Index 2 = District)
        post_tags = raw_doc.get("post_tags", [])
        location = {"province": "", "district": ""}
        if isinstance(post_tags, list):
            if len(post_tags) > 1: location["province"] = str(post_tags[1]).strip()
            if len(post_tags) > 2: location["district"] = str(post_tags[2]).strip()

        created_at = raw_doc.get("created_at")
        if not isinstance(created_at, datetime): 
            created_at = datetime.now(timezone.utc)

        batch_to_process.append({
            "post_id": post_id, 
            "cleaned_text": cleaned_text,
            "location": location, 
            "created_at": created_at
        })

        # Log sample for report
        if len(execution_report["processed_sample"]) < REPORT_MAX_ENTRIES:
            execution_report["processed_sample"].append({
                "post_id": str(post_id),
                "actions_applied": [k for k, v in cleaner.config.items() if v and k != "min_valid_length"],
                "location": location
            })

        # Flush batch to DB
        if len(batch_to_process) >= batch_size:
            operations = [UpdateOne(
                {"post_id": item["post_id"]},
                {"$set": {
                    "text": item["cleaned_text"],
                    # UPDATED SCHEMA: Nested sentiment structure
                    "sentiment": {
                        "llm": {
                            "label": "neutral",
                            "score": 0.0,
                            "model": "pending"
                        },
                        "transformer": {
                            "label": "neutral",
                            "score": 0.0,
                            "model": "pending"
                        }
                    },
                    "location": item["location"], 
                    "created_at": item["created_at"],
                    "_anonymized": True, 
                    "_processing_step": STEP_NAME
                }}, 
                upsert=True
            ) for item in batch_to_process]
            
            if operations:
                posts_processed.bulk_write(operations, ordered=False)
                execution_report["summary"]["successfully_processed"] += len(operations)
                processed_ids.update(item["post_id"] for item in batch_to_process)
            
            batch_to_process.clear()

            if time.time() - last_log >= 2.0:
                elapsed = time.time() - start_time
                rate = execution_report["summary"]["successfully_processed"] / elapsed if elapsed > 0 else 0
                print(f"  ⚡ {execution_report['summary']['successfully_processed']} processed | {rate:.0f} posts/sec")
                last_log = time.time()

    # Final flush for remaining items
    if batch_to_process:
        operations = [UpdateOne(
            {"post_id": item["post_id"]},
            {"$set": {
                "text": item["cleaned_text"],
                # UPDATED SCHEMA: Nested sentiment structure
                "sentiment": {
                    "llm": {
                        "label": "neutral",
                        "score": 0.0,
                        "model": "pending"
                    },
                    "transformer": {
                        "label": "neutral",
                        "score": 0.0,
                        "model": "pending"
                    }
                },
                "location": item["location"], 
                "created_at": item["created_at"],
                "_anonymized": True, 
                "_processing_step": STEP_NAME
            }}, 
            upsert=True
        ) for item in batch_to_process]
        
        if operations:
            posts_processed.bulk_write(operations, ordered=False)
            execution_report["summary"]["successfully_processed"] += len(operations)

    # Save Report
    execution_report["duration_seconds"] = round(time.time() - start_time, 2)
    execution_report["report_file_path"] = _save_report(execution_report, STEP_NAME)

    print("\n" + "= "*60)
    print("📊 PIPELINE STEP COMPLETE")
    print("= "*60)
    print(f"✅ Processed: {execution_report['summary']['successfully_processed']}")
    print(f"🚫 Skipped (existing): {execution_report['summary']['skipped_already_processed']}")
    print(f"🚫 Skipped (invalid): {execution_report['summary']['skipped_invalid_text']}")
    print(f"⏱️  Duration: {execution_report['duration_seconds']}s")
    print("= "*60)

    # Return output_data for next pipeline step
    return {
        "step": STEP_NAME,
        "status": "completed",
        "summary": execution_report["summary"],
        "duration_seconds": execution_report["duration_seconds"],
        "report_file": execution_report["report_file_path"],
        "previous_input": input_data
    }

# ==============================================================================
# 3. LOCAL TESTING BLOCK
# ==============================================================================
if __name__ == "__main__":
    from pymongo import MongoClient
    
    # Local test only: creates a temporary client strictly for debugging
    MONGO_URI = "mongodb+srv://COM6064:OnTHCZcqye91Yv1s@cluster0.bj4tnnh.mongodb.net/COM6064?appName=Cluster0"
    client = MongoClient(MONGO_URI)
    db = client["COM6064"]
    
    mock_context = {
        "base_dir": ".",
        "mongo_client": client,
        "db": db
    }
    mock_input = None
    
    try:
        result = run(mock_input, mock_context)
        print("\n📥 Local test output:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        client.close()
        print("🔌 Local test DB connection closed.")
