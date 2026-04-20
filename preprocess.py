"""
Pipeline Step: Text Preprocessing & Location Extraction
Adapted to run via pipeline.py using runpy.
"""
import re
import emoji
import time
import json
from datetime import datetime, timezone
from typing import Any, Dict, List
from pymongo import UpdateOne

# --- CONFIGURATION & NLTK SETUP ---
try:
    import nltk
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    TR_STOPWORDS = set(w.lower() for w in stopwords.words('turkish'))
except Exception as e:
    TR_STOPWORDS = set()
    print(f"⚠️ NLTK stopwords loading failed: {e}. Stopwords filtering disabled.")

# Precompile Regex Patterns
URL_RE = re.compile(r'https?://\S+|www\.\S+')
MENTION_RE = re.compile(r'@\w+')
HASHTAG_RE = re.compile(r'#\w+')
RT_RE = re.compile(r'\bRT\b', re.IGNORECASE)
REPEAT_RE = re.compile(r'(.)\1{2,}')
SPACE_RE = re.compile(r'\s+')

# --- HELPER FUNCTIONS ---
def extract_location(post_tags: list) -> dict:
    """Strictly selects index 1 as province and index 2 as district."""
    location = {"province": "", "district": ""}
    if not post_tags or not isinstance(post_tags, list):
        return location
    if len(post_tags) > 1:
        location["province"] = str(post_tags[1]).strip()
    if len(post_tags) > 2:
        location["district"] = str(post_tags[2]).strip()
    return location

def clean_text(text: str) -> str:
    """Applies noise removal, normalization, and stopword filtering."""
    if not isinstance(text, str):
        return ""
    text = URL_RE.sub('', text)
    text = emoji.replace_emoji(text, replace=' ')
    text = MENTION_RE.sub('', text)
    text = HASHTAG_RE.sub('', text)
    text = RT_RE.sub('', text)
    text = text.replace('İ', 'i').replace('I', 'ı').lower()
    text = REPEAT_RE.sub(r'\1', text)
    text = SPACE_RE.sub(' ', text).strip()
    
    if TR_STOPWORDS and text:
        tokens = text.split()
        text = ' '.join([w for w in tokens if w not in TR_STOPWORDS])
    return text

# --- PIPELINE CONTRACT FUNCTION ---
def run(input_data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
    STEP_NAME = "text_preprocessing"
    print(f"🚀 Running step: {STEP_NAME}")

    # 1. Access shared DB connection (DO NOT create a new MongoClient)
    db = context.get("db")
    if db is None:
        raise ValueError("Database connection not found in context. Use context['db'].")

    posts_raw = db.posts_raw
    posts_processed = db.posts_processed
    posts_processed.create_index("post_id", unique=True)

    print("🔍 Loading processed post IDs...")
    processed_ids = set(doc["post_id"] for doc in posts_processed.find({}, {"post_id": 1}))
    print(f"📋 {len(processed_ids)} posts already processed.")

    batch_size = 500
    cursor = posts_raw.find({}, {"_id": 1, "text": 1, "post_tags": 1, "created_at": 1}).batch_size(batch_size)

    total_processed = 0
    start_time = time.time()
    last_log = time.time()
    batch_to_process = []

    # 2. Process documents sequentially (safe for shared pymongo connection)
    for raw_doc in cursor:
        post_id = raw_doc["_id"]
        if post_id in processed_ids:
            continue

        text = raw_doc.get("text", "")
        post_tags = raw_doc.get("post_tags", [])
        created_at_raw = raw_doc.get("created_at", datetime.now(timezone.utc))

        cleaned_text = clean_text(text)
        location = extract_location(post_tags)

        if not isinstance(created_at_raw, datetime):
            created_at_raw = datetime.now(timezone.utc)

        batch_to_process.append({
            "post_id": post_id,
            "cleaned_text": cleaned_text,
            "location": location,
            "created_at": created_at_raw
        })

        if len(batch_to_process) >= batch_size:
            operations = [
                UpdateOne(
                    {"post_id": item["post_id"]},
                    {"$set": {
                        "text": item["cleaned_text"],
                        "sentiment": {"label": "neutral", "score": 0.0, "model": "pending"},
                        "location": item["location"],
                        "created_at": item["created_at"]
                    }},
                    upsert=True
                ) for item in batch_to_process
            ]

            if operations:
                posts_processed.bulk_write(operations, ordered=False)
                total_processed += len(operations)
                processed_ids.update(item["post_id"] for item in batch_to_process)

            batch_to_process.clear()

            if time.time() - last_log >= 2.0:
                elapsed = time.time() - start_time
                rate = total_processed / elapsed if elapsed > 0 else 0
                print(f"  ⚡ {total_processed} processed | {rate:.0f} posts/sec")
                last_log = time.time()

    # 3. Flush remaining batch
    if batch_to_process:
        operations = [
            UpdateOne(
                {"post_id": item["post_id"]},
                {"$set": {
                    "text": item["cleaned_text"],
                    "sentiment": {"label": "neutral", "score": 0.0, "model": "pending"},
                    "location": item["location"],
                    "created_at": item["created_at"]
                }},
                upsert=True
            ) for item in batch_to_process
        ]
        if operations:
            posts_processed.bulk_write(operations, ordered=False)
            total_processed += len(operations)

    print("\n" + "= "*60)
    print("📊 PIPELINE STEP COMPLETE")
    print("= "*60)
    print(f"✅ Total processed this run: {total_processed}")
    print(f"⏱️  Time elapsed: {time.time() - start_time:.2f}s")
    print(f"📦 Total in DB: {posts_processed.count_documents({})}")
    print("= "*60)

    # 4. Return output_data for the next pipeline step
    output_data = {
        "step": STEP_NAME,
        "status": "completed",
        "processed_count": total_processed,
        "time_elapsed": round(time.time() - start_time, 2),
        "previous_input": input_data  # Passthrough for downstream steps
    }
    return output_data

# --- LOCAL TESTING BLOCK ---
if __name__ == "__main__":
    from pymongo import MongoClient
    
    # Local test only: creates a temporary client just for this run
    MONGO_URI = "mongodb+srv://COM6064:OnTHCZcqye91Yv1s@cluster0.bj4tnnh.mongodb.net/COM6064?appName=Cluster0"
    client = MongoClient(MONGO_URI)
    db = client["COM6064"]
    
    mock_context = {
        "base_dir": ".",
        "mongo_client": client,
        "db": db
    }
    mock_input = None  # Simulates first step in pipeline
    
    try:
        result = run(mock_input, mock_context)
        print("\n📥 Local test output:")
        print(json.dumps(result, indent=2, ensure_ascii=True))
    finally:
        client.close()
        print("🔌 Local test DB connection closed.")