import csv
import json
import os
import re
import time
import hashlib
import threading
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests
import cloudscraper
from bs4 import BeautifulSoup
from pymongo import UpdateOne, MongoClient
from dotenv import load_dotenv


STEP_NAME = "fetch_scrape"

REQUEST_DELAY = 2.5
REQUEST_TIMEOUT = 20
MAX_RETRIES = 3
PROVINCE_RESULTS_COUNT = 10
DISTRICT_RESULTS_COUNT = 5
PARALLEL_FETCH_WORKERS = 4
MAX_PAGES_PER_TOPIC = 5

DEFAULT_LANGUAGE = "tr"
DEFAULT_SOURCE = "eksisozluk"

# MongoDB collection used for clean valid scraped posts only
DEFAULT_RAW_COLLECTION = "posts_raw"

# Local backup sections. MongoDB still uses one collection only.
RAW_SECTION = "raw"
CLEAN_SECTION = "clean"
INVALID_SECTION = "invalid"
REPORTS_SECTION = "reports"
LOGS_SECTION = "logs"

# Validation rules for local clean/invalid split
MIN_VALID_TEXT_LENGTH = 15

# Safety switch. Keep False for demos/tests so fetch never scans everything by accident.
ALLOW_AUTOMATIC_FETCH = False
BULK_TASK_WARNING_THRESHOLD = 100
ALL_VALUE = "all"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def normalize_lookup(text: str) -> str:
    text = str(text).strip().lower()
    for a, b in [("ı", "i"), ("ğ", "g"), ("ü", "u"), ("ş", "s"), ("ö", "o"), ("ç", "c")]:
        text = text.replace(a, b)
    return re.sub(r"\s+", " ", text)


def safe_slug(text: str) -> str:
    text = normalize_lookup(text)
    text = re.sub(r"[^a-z0-9\s_-]", "", text)
    return re.sub(r"\s+", "_", text).strip("_") or "unlisted"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def iso_to_datetime(iso_value: str) -> datetime:
    value = str(iso_value).replace("Z", "+00:00")
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def try_parse_date_to_iso(raw_value: Any, fallback_iso: Optional[str] = None) -> Optional[str]:
    value = str(raw_value or "").strip()
    if not value:
        return fallback_iso

    cleaned = value.replace(" tarihinde", "").replace("tarihinde", "").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)

    patterns = [
        "%d.%m.%Y %H:%M",
        "%d.%m.%Y %H:%M:%S",
        "%d.%m.%Y",
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
    ]

    for pattern in patterns:
        try:
            dt = datetime.strptime(cleaned, pattern)
            return dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
        except ValueError:
            continue

    iso_like = cleaned.replace(" ", "T")
    try:
        dt = datetime.fromisoformat(iso_like.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    except ValueError:
        return fallback_iso


def make_post_id(source: str, topic_url: str, entry_id: str, page: int, text: str) -> str:
    base = f"{source}|{topic_url}|{entry_id}|{page}|{text.strip()}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_list_simple(filename: Path) -> List[str]:
    with filename.open(encoding="utf-8-sig", newline="") as f:
        return [row[0].strip() for row in csv.reader(f) if row and row[0].strip()]


def load_category_names(filename: Path) -> List[str]:
    categories: List[str] = []
    seen = set()
    with filename.open(encoding="utf-8-sig", newline="") as f:
        sample = f.read(2048)
        f.seek(0)
        if "category" in sample and "word" in sample:
            for row in csv.DictReader(f):
                cat = str(row.get("category", "")).strip()
                if cat and cat not in seen:
                    categories.append(cat)
                    seen.add(cat)
        else:
            for row in csv.reader(f):
                if row and row[0].strip():
                    cat = row[0].strip()
                    if cat.lower() == "category":
                        continue
                    if cat not in seen:
                        categories.append(cat)
                        seen.add(cat)
    return categories


def load_districts_map(filename: Path) -> Dict[str, List[str]]:
    df = pd.read_csv(filename)
    missing = {"province", "district"} - set(df.columns)
    if missing:
        raise ValueError(f"{filename} missing columns: {', '.join(sorted(missing))}")

    district_map: Dict[str, List[str]] = defaultdict(list)
    for _, row in df.iterrows():
        province = str(row["province"]).strip()
        district = str(row["district"]).strip()
        if province and district and province.lower() != "nan" and district.lower() != "nan":
            district_map[province].append(district)

    clean: Dict[str, List[str]] = {}
    for province, districts in district_map.items():
        seen = set()
        unique: List[str] = []
        for district in districts:
            key = normalize_lookup(district)
            if key not in seen:
                unique.append(district)
                seen.add(key)
        clean[province] = sorted(unique, key=normalize_lookup)
    return dict(sorted(clean.items(), key=lambda kv: normalize_lookup(kv[0])))


def load_csv_column_set(path: Path, column_name: str) -> set:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return set()
    if column_name not in df.columns:
        return set()
    return set(df[column_name].dropna().astype(str).tolist())


def save_set_to_csv(items: set, path: Path, column_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({column_name: sorted(items)}).to_csv(path, index=False, encoding="utf-8-sig")


def ensure_csv_with_header(path: Path, column_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        pd.DataFrame({column_name: []}).to_csv(path, index=False, encoding="utf-8-sig")


def build_query(level: str, province: str, district: Optional[str], category: str) -> str:
    if level == "district":
        return f"site:eksisozluk.com {normalize_lookup(province)} {normalize_lookup(district or '')} {normalize_lookup(category)}".strip()
    return f"site:eksisozluk.com {normalize_lookup(province)} {normalize_lookup(category)}"


def make_fetch_key(level: str, province: str, district: Optional[str], category: str) -> str:
    return f"{province} | {district} | {category}" if level == "district" else f"{province} | {category}"


def make_completed_key(base_url: str, level: str, province: str, district: Optional[str], category: str) -> str:
    return f"{base_url}|||{level}|||{province}|||{district or ''}|||{category}"


def make_request_key(level: str, province: str, district: Optional[str], category: str) -> str:
    return f"{level}|||{province}|||{district or ''}|||{category}"


def normalize_topic_url(url: str) -> str:
    url = str(url).strip()
    parsed = urlparse(url)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}".rstrip("/")
    return re.sub(r"[?#].*$", "", re.sub(r"\?p=\d+$", "", url)).rstrip("/")


def extract_start_page(url: str) -> int:
    try:
        params = parse_qs(urlparse(str(url).strip()).query)
        if "p" in params and params["p"]:
            return max(1, int(params["p"][0]))
    except Exception:
        pass
    return 1


def paged_url(base_url: str, page: int) -> str:
    return f"{base_url}?p={page}"


def create_scraper() -> cloudscraper.CloudScraper:
    scraper = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "windows", "desktop": True}
    )
    scraper.headers.update({"Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7"})
    return scraper


def load_api_keys(base_dir: Path) -> List[str]:
    env_keys = os.getenv("SERPER_API_KEYS", "").strip()
    if env_keys:
        keys = [k.strip() for k in re.split(r"[\s,]+", env_keys) if k.strip()]
        if keys:
            return keys

    key_file = base_dir / "config" / "serper_keys.txt"
    if key_file.exists():
        raw = key_file.read_text(encoding="utf-8").strip()
        keys = [k.strip() for k in re.split(r"[\s,]+", raw) if k.strip()]
        if keys:
            return keys

    raise RuntimeError(
        "No Serper API keys found. Set SERPER_API_KEYS or create serper_keys.txt in the pipeline folder."
    )


def search_google(query: str, num_results: int, api_keys: List[str]) -> Tuple[List[str], int]:
    url = "https://google.serper.dev/search"
    last_error = None
    payload = {
        "q": query,
        "gl": "tr",
        "hl": "tr",
        "num": int(num_results),
    }

    for index, api_key in enumerate(api_keys):
        headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        except Exception as exc:
            last_error = exc
            continue

        try:
            data = response.json()
        except Exception:
            data = {}

        if response.status_code >= 400:
            details = str(data)[:300] if data else response.text[:300]
            message = f"Serper error for key #{index + 1}: HTTP {response.status_code} | {details}"
            if response.status_code in {400, 401, 403, 429}:
                last_error = RuntimeError(message)
                continue
            raise RuntimeError(message)

        links: List[str] = []
        seen = set()
        for item in data.get("organic", []):
            link = str(item.get("link", "")).strip()
            if not link or "eksisozluk.com" not in link or link in seen:
                continue
            seen.add(link)
            links.append(link)

        return links, index

    raise RuntimeError(f"All Serper API keys failed. Last error: {last_error}")


def fetch_url(scraper: cloudscraper.CloudScraper, url: str) -> Optional[str]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = scraper.get(url, timeout=REQUEST_TIMEOUT)
            if response.status_code == 404:
                return None
            if response.status_code in (403, 429, 500, 502, 503, 504):
                if attempt < MAX_RETRIES:
                    time.sleep(REQUEST_DELAY * attempt * 2)
                    continue
                return None
            response.raise_for_status()
            return response.text
        except Exception:
            if attempt < MAX_RETRIES:
                time.sleep(REQUEST_DELAY * attempt * 2)
    return None


def extract_title(soup: BeautifulSoup) -> str:
    for sel in ["span[itemprop='name']"]:
        title_node = soup.select_one(sel)
        if title_node:
            return title_node.get_text(strip=True)
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(" ", strip=True)
    title = soup.find("title")
    if title:
        return title.get_text(strip=True)
    return ""


def find_entry_nodes(soup: BeautifulSoup):
    selectors = [
        "ul#entry-item-list > li",
        "ul#entry-item-list li[data-id]",
        "li[id^='entry-item']",
        "li[data-id]",
        "#entry-item-list li",
    ]
    for sel in selectors:
        nodes = soup.select(sel)
        if nodes:
            return nodes
    return []


def extract_entries(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    seen_ids = set()

    for node in find_entry_nodes(soup):
        content_tag = node.select_one(".content") or node.select_one("div.content") or node.find("div", class_="content")
        if not content_tag:
            continue

        content = content_tag.get_text(" ", strip=True)
        if not content:
            continue

        node_id = (node.get("data-id") or node.get("id") or "").strip()
        if node_id and node_id in seen_ids:
            continue
        if node_id:
            seen_ids.add(node_id)

        author_tag = node.select_one(".entry-author") or node.find("a", class_="entry-author") or node.find("span", class_="entry-author")
        date_tag = node.select_one(".entry-date") or node.find("a", class_="entry-date") or node.find("span", class_="entry-date") or node.find("time")

        results.append(
            {
                "entry_id": node_id,
                "author": author_tag.get_text(" ", strip=True) if author_tag else "",
                "date": date_tag.get_text(" ", strip=True) if date_tag else "",
                "content": content,
            }
        )
    return results


def normalize_text_for_dedup(text: Any) -> str:
    value = str(text or "").strip().lower()
    return re.sub(r"\s+", " ", value)


def make_text_fingerprint(text: Any) -> str:
    normalized = normalize_text_for_dedup(text)
    if not normalized:
        return ""
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def validate_record(
    record: Dict[str, Any],
    level: str,
    province: str,
    district: Optional[str],
    category: str,
) -> Dict[str, Any]:
    issues: List[str] = []

    text_value = str(record.get("text", "")).strip()
    if not text_value:
        issues.append("missing_text")
    elif len(text_value) < MIN_VALID_TEXT_LENGTH:
        issues.append("text_too_short")

    if not str(record.get("source", "")).strip():
        issues.append("missing_source")
    if not str(record.get("language", "")).strip():
        issues.append("missing_language")

    created_at_meta = record.get("created_at", {})
    created_at_value = created_at_meta.get("$date") if isinstance(created_at_meta, dict) else None
    if not created_at_value:
        issues.append("missing_created_at")

    validation = record.get("validation", {})
    if validation.get("timestamp_used_fallback"):
        issues.append("fallback_timestamp_used")

    if not str(category).strip():
        issues.append("missing_category")
    if not str(province).strip():
        issues.append("missing_province")
    if level == "district" and not str(district or "").strip():
        issues.append("missing_district")

    blocking_issues = [issue for issue in issues if issue != "fallback_timestamp_used"]
    record["is_valid"] = len(blocking_issues) == 0
    record["validation"] = {
        **validation,
        "issues": issues,
        "level": level,
        "province": province,
        "district": district,
        "category": category,
    }
    return record


def convert_entries_to_raw_posts(
    base_url: str,
    level: str,
    province: str,
    district: Optional[str],
    category: str,
    entries: List[Dict[str, Any]],
    collected_at_iso: str,
) -> List[Dict[str, Any]]:
    raw_posts: List[Dict[str, Any]] = []
    seen_text_fingerprints = set()

    for entry in entries:
        text = str(entry.get("content", "")).strip()
        page = int(entry.get("page", 0) or 0)
        entry_id = str(entry.get("entry_id", "")).strip()

        text_fingerprint = make_text_fingerprint(text)
        if text_fingerprint and text_fingerprint in seen_text_fingerprints:
            continue
        if text_fingerprint:
            seen_text_fingerprints.add(text_fingerprint)

        parsed_created_at_iso = try_parse_date_to_iso(entry.get("date", ""), fallback_iso=None)
        timestamp_used_fallback = not bool(parsed_created_at_iso)
        created_at_iso = parsed_created_at_iso or collected_at_iso

        tags: List[str] = []
        if category.strip():
            tags.append(category.strip())
        if province.strip():
            tags.append(province.strip())
        if district and district.strip():
            tags.append(district.strip())

        record = {
            "post_id": make_post_id(DEFAULT_SOURCE, base_url, entry_id, page, text),
            "text": text,
            "created_at": {"$date": created_at_iso},
            "collected_at": {"$date": collected_at_iso},
            "source": DEFAULT_SOURCE,
            "language": DEFAULT_LANGUAGE,
            "post_tags": tags,
            "location": {
                "province": province,
                "district": district,
            },
            "category": category,
            "topic_url": base_url,
            "entry_id": entry_id,
            "page": page,
            "author": str(entry.get("author", "")).strip(),
            "level": level,
            "validation": {
                "raw_created_at": str(entry.get("date", "")).strip(),
                "timestamp_used_fallback": timestamp_used_fallback,
                "text_fingerprint": text_fingerprint,
            },
        }

        raw_posts.append(validate_record(record, level, province, district, category))

    return raw_posts


def convert_post_for_mongo(post: Dict[str, Any]) -> Dict[str, Any]:
    created_at_obj = post.get("created_at", {})
    collected_at_obj = post.get("collected_at", {})

    created_at_iso = created_at_obj.get("$date") if isinstance(created_at_obj, dict) else created_at_obj
    collected_at_iso = collected_at_obj.get("$date") if isinstance(collected_at_obj, dict) else collected_at_obj

    return {
        "post_id": post.get("post_id"),
        "text": post.get("text", ""),
        "created_at": iso_to_datetime(created_at_iso),
        "collected_at": iso_to_datetime(collected_at_iso),
        "source": post.get("source", DEFAULT_SOURCE),
        "language": post.get("language", DEFAULT_LANGUAGE),
        "post_tags": post.get("post_tags", []),
        "location": post.get("location", {}),
        "category": post.get("category"),
        "topic_url": post.get("topic_url"),
        "page": post.get("page"),
        "author": post.get("author", ""),
        "level": post.get("level"),
    }


def insert_raw_posts_to_mongo(db: Any, raw_posts: List[Dict[str, Any]], collection_name: str = DEFAULT_RAW_COLLECTION) -> Dict[str, int]:
    """Insert or update scraped posts in one MongoDB collection only."""
    if db is None or not raw_posts:
        return {"inserted_or_updated": 0, "errors": 0}

    collection = db[collection_name]

    try:
        collection.create_index("post_id", unique=True)
    except Exception as exc:
        logger.warning("MongoDB index creation failed for %s: %s", collection_name, exc)

    operations = []
    for post in raw_posts:
        try:
            mongo_doc = convert_post_for_mongo(post)
            if not mongo_doc.get("post_id"):
                continue
            operations.append(
                UpdateOne(
                    {"post_id": mongo_doc["post_id"]},
                    {"$set": mongo_doc},
                    upsert=True,
                )
            )
        except Exception as exc:
            logger.warning("Skipped invalid post during MongoDB conversion: %s", exc)

    if not operations:
        return {"inserted_or_updated": 0, "errors": 0}

    try:
        result = collection.bulk_write(operations, ordered=False)
        changed = int(result.upserted_count) + int(result.modified_count)
        logger.info("MongoDB saved %s post(s) into collection '%s'", changed, collection_name)
        return {"inserted_or_updated": changed, "errors": 0}
    except Exception as exc:
        logger.error("MongoDB bulk write failed for collection '%s': %s", collection_name, exc)
        return {"inserted_or_updated": 0, "errors": len(operations)}


def get_section_dirs(output_dir: Path) -> Dict[str, Path]:
    section_dirs = {
        RAW_SECTION: output_dir / RAW_SECTION,
        CLEAN_SECTION: output_dir / CLEAN_SECTION,
        INVALID_SECTION: output_dir / INVALID_SECTION,
        REPORTS_SECTION: output_dir / REPORTS_SECTION,
        LOGS_SECTION: output_dir / LOGS_SECTION,
    }
    for section_dir in section_dirs.values():
        section_dir.mkdir(parents=True, exist_ok=True)
    return section_dirs


def get_output_path(base_dir: Path, level: str, province: str, district: Optional[str], category: str, suffix: str) -> Path:
    if level == "district" and district:
        parent = base_dir / f"{safe_slug(province)}_districts" / safe_slug(district)
    else:
        parent = base_dir / safe_slug(province)
    parent.mkdir(parents=True, exist_ok=True)
    return parent / f"{safe_slug(category)}_{suffix}.json"


def load_json_list(path: Path) -> List[Dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    return data if isinstance(data, list) else []


def export_bucket_posts(
    section_dirs: Dict[str, Path],
    raw_posts_by_bucket: Dict[Tuple[str, str, Optional[str], str], List[Dict[str, Any]]],
) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, int]]:
    written_paths: List[str] = []
    clean_written_posts: List[Dict[str, Any]] = []
    counts = {
        "new_raw_posts": 0,
        "new_clean_posts": 0,
        "new_invalid_posts": 0,
        "duplicates_skipped": 0,
    }

    for (level, province, district, category), raw_posts in raw_posts_by_bucket.items():
        raw_path = get_output_path(section_dirs[RAW_SECTION], level, province, district, category, "raw_posts")
        clean_path = get_output_path(section_dirs[CLEAN_SECTION], level, province, district, category, "clean_posts")
        invalid_path = get_output_path(section_dirs[INVALID_SECTION], level, province, district, category, "invalid_posts")

        existing_raw = load_json_list(raw_path)
        existing_clean = load_json_list(clean_path)
        existing_invalid = load_json_list(invalid_path)

        existing_raw_ids = {item.get("post_id") for item in existing_raw if isinstance(item, dict) and item.get("post_id")}
        existing_clean_ids = {item.get("post_id") for item in existing_clean if isinstance(item, dict) and item.get("post_id")}
        existing_invalid_ids = {item.get("post_id") for item in existing_invalid if isinstance(item, dict) and item.get("post_id")}

        new_raw: List[Dict[str, Any]] = []
        new_clean: List[Dict[str, Any]] = []
        new_invalid: List[Dict[str, Any]] = []

        for post in raw_posts:
            post_id = post.get("post_id")
            if not post_id:
                continue
            if post_id in existing_raw_ids:
                counts["duplicates_skipped"] += 1
                continue

            new_raw.append(post)
            if post.get("is_valid"):
                if post_id not in existing_clean_ids:
                    new_clean.append(post)
            else:
                if post_id not in existing_invalid_ids:
                    new_invalid.append(post)

        save_json(raw_path, existing_raw + new_raw)
        save_json(clean_path, existing_clean + new_clean)
        save_json(invalid_path, existing_invalid + new_invalid)

        written_paths.extend([str(raw_path), str(clean_path), str(invalid_path)])
        clean_written_posts.extend(new_clean)

        counts["new_raw_posts"] += len(new_raw)
        counts["new_clean_posts"] += len(new_clean)
        counts["new_invalid_posts"] += len(new_invalid)

        logger.info(
            "Local export | raw=%s | clean=%s | invalid=%s | bucket=%s/%s/%s",
            len(new_raw),
            len(new_clean),
            len(new_invalid),
            province,
            district or "",
            category,
        )

    return written_paths, clean_written_posts, counts


def append_log_event(log_path: Path, event_type: str, message: str, **meta: Any) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "time": utc_now_iso(),
        "event_type": event_type,
        "message": message,
        "meta": meta,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def collect_unique_posts_from_files(file_paths: List[Path]) -> List[Dict[str, Any]]:
    combined: List[Dict[str, Any]] = []
    seen_ids = set()

    for file_path in file_paths:
        try:
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        if not isinstance(data, list):
            continue

        for item in data:
            if not isinstance(item, dict):
                continue
            post_id = item.get("post_id")
            if not post_id or post_id in seen_ids:
                continue
            seen_ids.add(post_id)
            combined.append(item)

    return combined


def export_district_merged(section_base_dir: Path, province: str, district: Optional[str], suffix: str, merged_suffix: str) -> Optional[Path]:
    if district:
        folder = section_base_dir / f"{safe_slug(province)}_districts" / safe_slug(district)
        merge_filename = f"{safe_slug(district)}_{merged_suffix}.json"
    else:
        folder = section_base_dir / safe_slug(province)
        merge_filename = f"{safe_slug(province)}_{merged_suffix}.json"

    if not folder.exists():
        return None

    source_files = [
        folder / name
        for name in os.listdir(folder)
        if name.endswith(suffix) and name != merge_filename
    ]
    if not source_files:
        return None

    combined = collect_unique_posts_from_files(source_files)
    merged_path = folder / merge_filename
    save_json(merged_path, combined)
    return merged_path


def export_province_combined_backups(section_base_dir: Path, suffix: str, merged_suffix: str) -> List[Path]:
    exported_paths: List[Path] = []

    if not section_base_dir.exists():
        return exported_paths

    for entry in os.scandir(section_base_dir):
        if not entry.is_dir():
            continue

        folder_name = entry.name
        folder_path = Path(entry.path)

        if folder_name.endswith("_districts"):
            province_slug = folder_name[:-10]
            source_files: List[Path] = []
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.endswith(suffix):
                        source_files.append(Path(root) / file)
            if not source_files:
                continue
            combined = collect_unique_posts_from_files(source_files)
            combined_path = folder_path / f"{province_slug}_{merged_suffix}.json"
            save_json(combined_path, combined)
            exported_paths.append(combined_path)
        else:
            source_files = [Path(item.path) for item in os.scandir(folder_path) if item.is_file() and item.name.endswith(suffix)]
            if not source_files:
                continue
            combined = collect_unique_posts_from_files(source_files)
            combined_path = folder_path / f"{folder_name}_{merged_suffix}.json"
            save_json(combined_path, combined)
            exported_paths.append(combined_path)

    return exported_paths


def build_combined_backup_data(section_base_dir: Path, suffix: str) -> List[Dict[str, Any]]:
    file_paths: List[Path] = []
    if not section_base_dir.exists():
        return []
    for root, _, files in os.walk(section_base_dir):
        for file in files:
            if file.endswith(suffix):
                file_paths.append(Path(root) / file)
    return collect_unique_posts_from_files(file_paths)


def export_all_backups(section_dirs: Dict[str, Path]) -> Tuple[Path, List[Path], int, Path, List[Path], int]:
    clean_province_paths = export_province_combined_backups(
        section_dirs[CLEAN_SECTION],
        suffix="_clean_posts.json",
        merged_suffix="all_clean_posts",
    )
    invalid_province_paths = export_province_combined_backups(
        section_dirs[INVALID_SECTION],
        suffix="_invalid_posts.json",
        merged_suffix="all_invalid_posts",
    )

    clean_combined = build_combined_backup_data(section_dirs[CLEAN_SECTION], suffix="_clean_posts.json")
    invalid_combined = build_combined_backup_data(section_dirs[INVALID_SECTION], suffix="_invalid_posts.json")

    clean_combined_file = section_dirs[CLEAN_SECTION] / "ALL_DATA.json"
    invalid_combined_file = section_dirs[INVALID_SECTION] / "ALL_INVALID_DATA.json"

    save_json(clean_combined_file, clean_combined)
    save_json(invalid_combined_file, invalid_combined)

    return (
        clean_combined_file,
        clean_province_paths,
        len(clean_combined),
        invalid_combined_file,
        invalid_province_paths,
        len(invalid_combined),
    )


def scrape_topic_pages(
    base_url: str,
    visited_pages: set,
    visited_pages_file: Path,
    visited_pages_lock: threading.Lock,
) -> Tuple[str, List[Dict[str, Any]], List[int]]:
    scraper = create_scraper()
    all_entries: List[Dict[str, Any]] = []
    scraped_pages: List[int] = []
    title = ""

    for page in range(1, MAX_PAGES_PER_TOPIC + 1):
        target_url = paged_url(base_url, page)

        with visited_pages_lock:
            if target_url in visited_pages:
                continue

        html = fetch_url(scraper, target_url)
        if not html:
            continue

        soup = BeautifulSoup(html, "html.parser")
        if not title:
            title = extract_title(soup)

        entries = extract_entries(soup)
        if not entries:
            logger.info("No entries found | topic=%s | page=%s", base_url, page)
            continue

        for entry in entries:
            entry["page"] = page
            all_entries.append(entry)

        scraped_pages.append(page)
        logger.info("Page scraped | topic=%s | page=%s | entries=%s", base_url, page, len(entries))

        with visited_pages_lock:
            visited_pages.add(target_url)
            save_set_to_csv(visited_pages, visited_pages_file, "url")

        time.sleep(0.4)

    return title, all_entries, scraped_pages


def generate_tasks(base_dir: Path, include_province_level: bool = False) -> List[Dict[str, Any]]:
    provinces_file = base_dir / "data" / "provinces.csv"
    districts_file = base_dir / "data" / "districts.csv"
    categories_file = base_dir / "data" / "categories.csv"

    for path in [provinces_file, districts_file, categories_file]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    provinces = load_list_simple(provinces_file)
    categories = load_category_names(categories_file)
    district_map = load_districts_map(districts_file)

    tasks: List[Dict[str, Any]] = []
    if include_province_level:
        for province in provinces:
            for category in categories:
                tasks.append(
                    {
                        "level": "province",
                        "province": province,
                        "district": None,
                        "category": category,
                        "num_results": PROVINCE_RESULTS_COUNT,
                    }
                )

    for province, districts in district_map.items():
        for district in districts:
            for category in categories:
                tasks.append(
                    {
                        "level": "district",
                        "province": province,
                        "district": district,
                        "category": category,
                        "num_results": DISTRICT_RESULTS_COUNT,
                    }
                )
    return tasks


def is_all_value(value: Any) -> bool:
    return normalize_lookup(str(value or "")) in {"all", "hepsi", "*"}


def find_match_by_normalized(value: str, options: List[str], label: str) -> str:
    wanted = normalize_lookup(value)
    for option in options:
        if normalize_lookup(option) == wanted:
            return option
    raise ValueError(f"Unknown {label}: {value}")


def load_expansion_sources(base_dir: Path) -> Tuple[List[str], Dict[str, List[str]], List[str]]:
    provinces_file = base_dir / "data" / "provinces.csv"
    districts_file = base_dir / "data" / "districts.csv"
    categories_file = base_dir / "data" / "categories.csv"

    for path in [provinces_file, districts_file, categories_file]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    provinces = load_list_simple(provinces_file)
    district_map = load_districts_map(districts_file)
    categories = load_category_names(categories_file)
    return provinces, district_map, categories


def expand_one_input_request(item: Dict[str, Any], base_dir: Path) -> List[Dict[str, Any]]:
    """Expand province='all', district='all', and category='all' into real pipeline tasks."""
    if not isinstance(item, dict):
        raise ValueError("Each request must be a dictionary")

    provinces, district_map, all_categories = load_expansion_sources(base_dir)

    level = str(item.get("level", "district")).strip().lower() or "district"
    province_value = str(item.get("province", "")).strip()
    district_raw = item.get("district")
    district_value = str(district_raw).strip() if district_raw not in (None, "", "None") else None
    category_value = str(item.get("category", "")).strip()

    if not province_value:
        raise ValueError("Each request must include province. Use province='all' for all provinces.")
    if not category_value:
        raise ValueError("Each request must include category. Use category='all' for all categories.")

    selected_categories = all_categories if is_all_value(category_value) else [category_value]

    if is_all_value(province_value):
        selected_provinces = list(district_map.keys())
    else:
        selected_provinces = [find_match_by_normalized(province_value, list(district_map.keys()), "province")]

    num_results_default = DISTRICT_RESULTS_COUNT if level == "district" else PROVINCE_RESULTS_COUNT
    num_results = int(item.get("num_results") or num_results_default)
    max_topics = int(item.get("max_topics") or 0) or None

    tasks: List[Dict[str, Any]] = []

    if level == "province":
        for province in selected_provinces:
            for category in selected_categories:
                tasks.append({
                    "level": "province",
                    "province": province,
                    "district": None,
                    "category": category,
                    "num_results": num_results,
                    "max_topics": max_topics,
                })
        return tasks

    if district_value is None:
        raise ValueError("District-level request must include district. Use district='all' for all districts.")

    if is_all_value(district_value):
        for province in selected_provinces:
            for district in district_map.get(province, []):
                for category in selected_categories:
                    tasks.append({
                        "level": "district",
                        "province": province,
                        "district": district,
                        "category": category,
                        "num_results": num_results,
                        "max_topics": max_topics,
                    })
        return tasks

    wanted = normalize_lookup(district_value)
    matched_any = False
    for province in selected_provinces:
        matched_district = None
        for district in district_map.get(province, []):
            if normalize_lookup(district) == wanted:
                matched_district = district
                break
        if matched_district is None:
            continue

        matched_any = True
        for category in selected_categories:
            tasks.append({
                "level": "district",
                "province": province,
                "district": matched_district,
                "category": category,
                "num_results": num_results,
                "max_topics": max_topics,
            })

    if not matched_any:
        raise ValueError(f"Unknown district '{district_value}' for selected province scope")

    return tasks


def input_to_tasks(input_data: Any, base_dir: Path) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(input_data, dict) or not input_data:
        return None

    if "requests" in input_data:
        raw_requests = input_data.get("requests")
        if not isinstance(raw_requests, list):
            raise ValueError("input_data['requests'] must be a list")

        tasks: List[Dict[str, Any]] = []
        for item in raw_requests:
            tasks.extend(expand_one_input_request(item, base_dir))
        return tasks

    if "province" in input_data or "category" in input_data or "district" in input_data:
        return expand_one_input_request(input_data, base_dir)

    return None

def build_direct_input_from_args(args) -> Dict[str, Any]:
    if not args.province and not args.category and not args.district:
        return {}

    if not args.province or not args.category:
        raise SystemExit("When using direct CLI arguments, --province and --category are required.")
    if args.level == "district" and not args.district:
        raise SystemExit("When level is district, --district is required.")

    result = {
        "level": args.level,
        "province": args.province,
        "district": args.district,
        "category": args.category,
    }

    if args.num_results is not None:
        result["num_results"] = args.num_results
    if args.max_topics is not None:
        result["max_topics"] = args.max_topics
    if args.include_province_level:
        result["include_province_level"] = True

    return result


def parse_scalar_value(value: str) -> Any:
    value = value.strip().strip('"').strip("'")
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def parse_loose_object(raw_object: str) -> Dict[str, Any]:
    cleaned = raw_object.strip()
    if cleaned.startswith("{") and cleaned.endswith("}"):
        cleaned = cleaned[1:-1]

    parsed: Dict[str, Any] = {}
    current = ""
    depth = 0
    parts: List[str] = []

    for char in cleaned:
        if char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1

        if char == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += char

    if current:
        parts.append(current)

    for part in parts:
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        key = key.strip().strip('"').strip("'")
        value = value.strip()
        if not key:
            continue
        parsed[key] = parse_scalar_value(value)

    return parsed


def parse_loose_requests(raw_input: str) -> Optional[Dict[str, Any]]:
    text = raw_input.strip()
    if not text.startswith("{") or "requests" not in text:
        return None

    match = re.search(r"requests\s*:\s*\[(.*)\]\s*}", text, flags=re.DOTALL)
    if not match:
        return None

    inner = match.group(1).strip()
    objects: List[str] = []
    current = ""
    depth = 0

    for char in inner:
        if char == "{":
            depth += 1
        if depth > 0:
            current += char
        if char == "}":
            depth -= 1
            if depth == 0 and current:
                objects.append(current)
                current = ""

    requests = [parse_loose_object(obj) for obj in objects if obj.strip()]
    return {"requests": requests}


def parse_pipeline_input(input_data: Any) -> Any:
    """Accept dict input, JSON string input, Python dict string input, loose pipeline strings, and loose requests lists."""
    if not isinstance(input_data, str):
        return input_data

    raw_input = input_data.strip()
    if not raw_input:
        return {}

    try:
        return json.loads(raw_input)
    except Exception:
        pass

    try:
        import ast
        return ast.literal_eval(raw_input)
    except Exception:
        pass

    loose_requests = parse_loose_requests(raw_input)
    if loose_requests is not None:
        return loose_requests

    parsed = parse_loose_object(raw_input)
    if parsed:
        return parsed

    raise ValueError(f"Invalid input passed to fetch step: {raw_input}")


def run(input_data: Any, context: Dict[str, Any]) -> Any:
    input_data = parse_pipeline_input(input_data)

    base_dir = Path(context.get("base_dir") or ".")
    db = context.get("db")

    output_dir = base_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    section_dirs = get_section_dirs(output_dir)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = section_dirs[LOGS_SECTION] / f"fetch_run_{run_id}.jsonl"
    report_path = section_dirs[REPORTS_SECTION] / f"fetch_report_{run_id}.json"

    append_log_event(log_path, "run_started", "Fetch run started", input_data=input_data)

    visited_pages_file = base_dir / "state" / "visited_pages.csv"
    completed_topics_file = base_dir / "state" / "completed_topics.csv"
    completed_requests_file = base_dir / "state" / "completed_requests.csv"

    ensure_csv_with_header(visited_pages_file, "url")
    ensure_csv_with_header(completed_topics_file, "key")
    ensure_csv_with_header(completed_requests_file, "key")

    include_province_level = False
    global_max_topics = None
    if isinstance(input_data, dict):
        include_province_level = bool(input_data.get("include_province_level", False))
        global_max_topics = input_data.get("max_topics")

    explicit_tasks = input_to_tasks(input_data, base_dir)
    api_keys = load_api_keys(base_dir)

    if explicit_tasks is None and not ALLOW_AUTOMATIC_FETCH:
        raise ValueError(
            "Fetch stopped because no valid input was provided. "
            "Use --input with province, district, category, num_results, and max_topics. "
            "This safety guard prevents accidental full-project API usage."
        )

    all_tasks = explicit_tasks if explicit_tasks is not None else generate_tasks(
        base_dir, include_province_level=include_province_level
    )
    mode = "input" if explicit_tasks is not None else "automatic"

    logger.info("Generated tasks: %s | estimated Serper API calls: %s", len(all_tasks), len(all_tasks))
    append_log_event(
        log_path,
        "tasks_generated",
        "Fetch tasks generated",
        task_count=len(all_tasks),
        estimated_api_calls=len(all_tasks),
        mode=mode,
    )

    confirm_bulk = bool(input_data.get("confirm_bulk", False)) if isinstance(input_data, dict) else False
    if len(all_tasks) > BULK_TASK_WARNING_THRESHOLD and not confirm_bulk:
        raise ValueError(
            f"Bulk fetch blocked for safety: {len(all_tasks)} tasks would run. "
            "Add confirm_bulk=true in the input if you really want this bulk run."
        )

    visited_pages = load_csv_column_set(visited_pages_file, "url")
    completed_topics = load_csv_column_set(completed_topics_file, "key")
    completed_requests = load_csv_column_set(completed_requests_file, "key")

    visited_pages_lock = threading.Lock()
    completed_topics_lock = threading.Lock()

    remaining_tasks: List[Dict[str, Any]] = []
    skipped_completed_requests = 0
    skipped_completed_topics = 0
    skipped_duplicate_search_urls = 0

    for task in all_tasks:
        query = build_query(task["level"], task["province"], task["district"], task["category"])
        request_key = make_request_key(task["level"], task["province"], task["district"], task["category"])

        task["query"] = query
        task["fetch_key"] = make_fetch_key(task["level"], task["province"], task["district"], task["category"])
        task["request_key"] = request_key

        if request_key in completed_requests:
            skipped_completed_requests += 1
            append_log_event(log_path, "request_skipped", "Request already completed", request_key=request_key)
            continue

        remaining_tasks.append(task)

    total_tasks = len(remaining_tasks)
    fetched_queries = 0
    processed_topics = 0
    total_raw_posts = 0
    total_clean_posts = 0
    total_invalid_posts = 0
    total_urls_found = 0
    used_key_indexes = set()
    mongo_upserted_count = 0
    local_duplicates_skipped = 0

    written_paths: List[str] = []
    merged_paths: List[str] = []

    grouped_topics: Dict[Tuple[str, str], List[Tuple[str, int, str, str, Optional[str], str, str]]] = defaultdict(list)
    seen_topic_keys = set()

    for task in remaining_tasks:
        logger.info("Searching Serper: %s", task["query"])
        append_log_event(log_path, "query_started", "Serper query started", query=task["query"], fetch_key=task["fetch_key"])

        links, key_index = search_google(task["query"], task["num_results"], api_keys)
        used_key_indexes.add(key_index)
        total_urls_found += len(links)
        fetched_queries += 1

        logger.info("Found %s URL(s) for: %s", len(links), task["fetch_key"])
        append_log_event(log_path, "query_finished", "Serper query finished", query=task["query"], urls_found=len(links))

        if not links:
            completed_requests.add(task["request_key"])
            save_set_to_csv(completed_requests, completed_requests_file, "key")
            time.sleep(REQUEST_DELAY)
            continue

        task_max_topics = task.get("max_topics") or global_max_topics or len(links)
        try:
            task_max_topics = int(task_max_topics)
        except Exception:
            task_max_topics = len(links)

        added_any_topic = False
        added_for_this_request = 0

        for raw_url in links:
            raw_url = str(raw_url).strip()
            if not raw_url:
                continue

            base_url = normalize_topic_url(raw_url)
            start_page = extract_start_page(raw_url)
            completed_key = make_completed_key(base_url, task["level"], task["province"], task["district"], task["category"])

            if completed_key in completed_topics:
                skipped_completed_topics += 1
                append_log_event(log_path, "topic_skipped", "Topic already completed", completed_key=completed_key, url=base_url)
                continue

            if completed_key in seen_topic_keys:
                skipped_duplicate_search_urls += 1
                append_log_event(log_path, "duplicate_url_skipped", "Duplicate search URL skipped", completed_key=completed_key, url=base_url)
                continue

            seen_topic_keys.add(completed_key)
            group_key = (task["province"], task["district"] or "")
            grouped_topics[group_key].append(
                (
                    base_url,
                    start_page,
                    task["level"],
                    task["province"],
                    task["district"],
                    task["category"],
                    task["request_key"],
                )
            )
            added_any_topic = True
            added_for_this_request += 1

            if added_for_this_request >= task_max_topics:
                break

        if not added_any_topic:
            completed_requests.add(task["request_key"])
            save_set_to_csv(completed_requests, completed_requests_file, "key")

        time.sleep(REQUEST_DELAY)

    request_success: Dict[str, bool] = {task["request_key"]: False for task in remaining_tasks}

    for (province, district_str), group_items in grouped_topics.items():

        def _fetch_one(item: Tuple[str, int, str, str, Optional[str], str, str]):
            base_url, start_page, level, prov, dist, category, request_key = item
            completed_key = make_completed_key(base_url, level, prov, dist, category)

            with completed_topics_lock:
                if completed_key in completed_topics:
                    return None

            logger.info(
                "Scraping topic started | province=%s | district=%s | category=%s | url=%s",
                prov,
                dist or "",
                category,
                base_url,
            )
            append_log_event(log_path, "topic_started", "Topic scraping started", province=prov, district=dist, category=category, url=base_url)

            _, entries, scraped_pages = scrape_topic_pages(
                base_url=base_url,
                visited_pages=visited_pages,
                visited_pages_file=visited_pages_file,
                visited_pages_lock=visited_pages_lock,
            )

            return base_url, level, prov, dist, category, entries, completed_key, request_key, scraped_pages

        with ThreadPoolExecutor(max_workers=PARALLEL_FETCH_WORKERS) as executor:
            futures = [executor.submit(_fetch_one, item) for item in group_items]

            for future in as_completed(futures):
                result = future.result()
                if result is None:
                    continue

                base_url, level, prov, dist, category, entries, completed_key, request_key, scraped_pages = result

                if entries:
                    collected_at_iso = utc_now_iso()
                    raw_posts = convert_entries_to_raw_posts(
                        base_url,
                        level,
                        prov,
                        dist,
                        category,
                        entries,
                        collected_at_iso,
                    )

                    bucket = (level, prov, dist, category)
                    one_topic_bucket: Dict[Tuple[str, str, Optional[str], str], List[Dict[str, Any]]] = defaultdict(list)
                    one_topic_bucket[bucket].extend(raw_posts)

                    paths, clean_written_posts, export_counts = export_bucket_posts(section_dirs, one_topic_bucket)
                    written_paths.extend(paths)

                    mongo_result = insert_raw_posts_to_mongo(
                        db,
                        clean_written_posts,
                        collection_name=DEFAULT_RAW_COLLECTION,
                    )
                    mongo_upserted_count += mongo_result.get("inserted_or_updated", 0)

                    total_raw_posts += export_counts["new_raw_posts"]
                    total_clean_posts += export_counts["new_clean_posts"]
                    total_invalid_posts += export_counts["new_invalid_posts"]
                    local_duplicates_skipped += export_counts["duplicates_skipped"]
                    processed_topics += 1
                    request_success[request_key] = True

                    clean_merged = export_district_merged(
                        section_dirs[CLEAN_SECTION],
                        prov,
                        dist,
                        suffix="_clean_posts.json",
                        merged_suffix="all_clean_posts",
                    )
                    invalid_merged = export_district_merged(
                        section_dirs[INVALID_SECTION],
                        prov,
                        dist,
                        suffix="_invalid_posts.json",
                        merged_suffix="all_invalid_posts",
                    )
                    for merged in [clean_merged, invalid_merged]:
                        if merged:
                            merged_path = str(merged)
                            if merged_path not in merged_paths:
                                merged_paths.append(merged_path)

                    logger.info(
                        "Topic saved immediately | url=%s | entries=%s | raw=%s | clean=%s | invalid=%s | mongo_changed=%s",
                        base_url,
                        len(entries),
                        export_counts["new_raw_posts"],
                        export_counts["new_clean_posts"],
                        export_counts["new_invalid_posts"],
                        mongo_result.get("inserted_or_updated", 0),
                    )
                    append_log_event(
                        log_path,
                        "topic_saved",
                        "Topic saved immediately",
                        url=base_url,
                        entries=len(entries),
                        raw=export_counts["new_raw_posts"],
                        clean=export_counts["new_clean_posts"],
                        invalid=export_counts["new_invalid_posts"],
                        mongo_changed=mongo_result.get("inserted_or_updated", 0),
                    )

                    with completed_topics_lock:
                        completed_topics.add(completed_key)
                        save_set_to_csv(completed_topics, completed_topics_file, "key")

                elif scraped_pages:
                    request_success[request_key] = True
                    logger.info("Topic visited but no valid entries found | url=%s", base_url)
                    append_log_event(log_path, "topic_empty", "Topic visited but no valid entries found", url=base_url)

                    with completed_topics_lock:
                        completed_topics.add(completed_key)
                        save_set_to_csv(completed_topics, completed_topics_file, "key")

    for request_key, success in request_success.items():
        if success or request_key not in completed_requests:
            completed_requests.add(request_key)

    save_set_to_csv(completed_requests, completed_requests_file, "key")

    (
        combined_output_file,
        province_combined_paths,
        combined_clean_count,
        invalid_combined_output_file,
        invalid_province_paths,
        combined_invalid_count,
    ) = export_all_backups(section_dirs)

    result_payload = {
        "step": STEP_NAME,
        "status": "ok",
        "mode": mode,
        "base_dir": str(base_dir),
        "output_dir": str(output_dir),
        "raw_output_dir": str(section_dirs[RAW_SECTION]),
        "clean_output_dir": str(section_dirs[CLEAN_SECTION]),
        "invalid_output_dir": str(section_dirs[INVALID_SECTION]),
        "logs_dir": str(section_dirs[LOGS_SECTION]),
        "reports_dir": str(section_dirs[REPORTS_SECTION]),
        "log_file": str(log_path),
        "report_file": str(report_path),
        "combined_output_file": str(combined_output_file),
        "invalid_combined_output_file": str(invalid_combined_output_file),
        "province_combined_files": [str(p) for p in province_combined_paths],
        "invalid_province_combined_files": [str(p) for p in invalid_province_paths],
        "district_merged_files": merged_paths,
        "written_files": written_paths,
        "mongo_collection": DEFAULT_RAW_COLLECTION,
        "mongo_source": "clean valid posts only",
        "total_generated_tasks": len(all_tasks),
        "run_task_count": total_tasks,
        "fetched_query_count": fetched_queries,
        "processed_topic_count": processed_topics,
        "skipped_completed_requests": skipped_completed_requests,
        "skipped_completed_topics": skipped_completed_topics,
        "skipped_duplicate_search_urls": skipped_duplicate_search_urls,
        "local_duplicates_skipped": local_duplicates_skipped,
        "total_urls_found": total_urls_found,
        "new_raw_posts_in_this_run": total_raw_posts,
        "new_clean_posts_in_this_run": total_clean_posts,
        "new_invalid_posts_in_this_run": total_invalid_posts,
        "combined_clean_post_count": combined_clean_count,
        "combined_invalid_post_count": combined_invalid_count,
        "used_api_key_count": len(used_key_indexes),
        "mongo_upserted_count": mongo_upserted_count,
        "resume_files": {
            "visited_pages_file": str(visited_pages_file),
            "completed_topics_file": str(completed_topics_file),
            "completed_requests_file": str(completed_requests_file),
        },
    }

    save_json(report_path, result_payload)
    append_log_event(log_path, "run_finished", "Fetch run finished", summary=result_payload)

    return result_payload


if __name__ == "__main__":
    import argparse

    load_dotenv()

    parser = argparse.ArgumentParser(description="Run fetch step directly with MongoDB support")
    parser.add_argument("--input", default=None, help="Optional JSON input")
    parser.add_argument("--max-topics", type=int, default=None, help="Optional limit on requests to run")
    parser.add_argument("--include-province-level", action="store_true", help="Include province-level tasks in automatic mode")

    parser.add_argument("--level", choices=["district", "province"], default="district", help="Task level")
    parser.add_argument("--province", default=None, help="Province name")
    parser.add_argument("--district", default=None, help="District name")
    parser.add_argument("--category", default=None, help="Category name")
    parser.add_argument("--num-results", type=int, default=None, help="Number of Serper results to request")

    args = parser.parse_args()

    demo_input = None

    if args.input:
        raw = str(args.input).strip()
        if raw.startswith("'") and raw.endswith("'"):
            raw = raw[1:-1]
        try:
            demo_input = parse_pipeline_input(raw)
        except Exception as exc:
            raise SystemExit(f"Invalid input for --input: {exc}")
    else:
        demo_input = build_direct_input_from_args(args)

    if demo_input is None:
        demo_input = {}

    if args.max_topics is not None and isinstance(demo_input, dict):
        demo_input["max_topics"] = args.max_topics
    if args.include_province_level and isinstance(demo_input, dict):
        demo_input["include_province_level"] = True

    mongo_uri = os.getenv("MONGO_URI")
    mongo_db_name = os.getenv("MONGO_DB_NAME", "COM6064")

    if not mongo_uri:
        raise SystemExit("Missing MONGO_URI in .env")

    mongo_client = MongoClient(mongo_uri)
    mongo_db = mongo_client[mongo_db_name]

    try:
        demo_context = {
            "base_dir": ".",
            "db": mongo_db,
            "mongo_client": mongo_client,
        }
        result = run(demo_input, demo_context)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        mongo_client.close()
