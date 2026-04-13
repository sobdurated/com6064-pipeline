import csv
import json
import os
import re
import time
import hashlib
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

STEP_NAME = "fetch_scrape"

REQUEST_DELAY = 2.5
REQUEST_TIMEOUT = 20
MAX_RETRIES = 3
PROVINCE_RESULTS_COUNT = 10
DISTRICT_RESULTS_COUNT = 5
PARALLEL_FETCH_WORKERS = 4
DEFAULT_LANGUAGE = "tr"
DEFAULT_SOURCE = "eksisozluk"


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
        return f"site:eksisozluk.com {province.lower()} {str(district or '').lower()} {category.lower()}".strip()
    return f"site:eksisozluk.com {province.lower()} {category.lower()}"


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

    for index, api_key in enumerate(api_keys):
        headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "q": query,
            "gl": "tr",
            "hl": "tr",
            "num": num_results,
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
            data = response.json()
        except Exception as exc:
            last_error = exc
            continue

        if response.status_code >= 400:
            message = f"Serper error for key #{index + 1}: HTTP {response.status_code}"
            if response.status_code in {401, 403, 429}:
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
    for entry in entries:
        text = str(entry.get("content", "")).strip()
        if not text:
            continue

        page = int(entry.get("page", 0) or 0)
        entry_id = str(entry.get("entry_id", "")).strip()
        created_at_iso = try_parse_date_to_iso(entry.get("date", ""), fallback_iso=collected_at_iso)

        tags: List[str] = []
        if category.strip():
            tags.append(category.strip())
        if province.strip():
            tags.append(province.strip())
        if district and district.strip():
            tags.append(district.strip())

        raw_posts.append(
            {
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
            }
        )
    return raw_posts


def get_output_path(output_dir: Path, level: str, province: str, district: Optional[str], category: str) -> Path:
    if level == "district" and district:
        parent = output_dir / f"{safe_slug(province)}_districts" / safe_slug(district)
    else:
        parent = output_dir / safe_slug(province)
    parent.mkdir(parents=True, exist_ok=True)
    return parent / f"{safe_slug(category)}_raw_posts.json"


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


def export_raw_posts(output_dir: Path, raw_posts_by_bucket: Dict[Tuple[str, str, Optional[str], str], List[Dict[str, Any]]]) -> List[Path]:
    written_paths: List[Path] = []
    for (level, province, district, category), raw_posts in raw_posts_by_bucket.items():
        output_path = get_output_path(output_dir, level, province, district, category)
        existing: List[Dict[str, Any]] = []
        if output_path.exists() and output_path.stat().st_size > 0:
            try:
                with output_path.open("r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:
                existing = []

        existing_ids = {item.get("post_id") for item in existing if isinstance(item, dict) and item.get("post_id")}
        new_posts = [item for item in raw_posts if item.get("post_id") not in existing_ids]
        save_json(output_path, existing + new_posts)
        written_paths.append(output_path)
    return written_paths


def export_district_merged(output_dir: Path, province: str, district: Optional[str]) -> Optional[Path]:
    if district:
        district_folder = output_dir / f"{safe_slug(province)}_districts" / safe_slug(district)
        merge_filename = f"{safe_slug(district)}_all_categories_raw_posts.json"
    else:
        district_folder = output_dir / safe_slug(province)
        merge_filename = f"{safe_slug(province)}_all_categories_raw_posts.json"

    if not district_folder.exists():
        return None

    source_files = [
        district_folder / f
        for f in os.listdir(district_folder)
        if f.endswith("_raw_posts.json") and f != merge_filename
    ]
    if not source_files:
        return None

    combined = collect_unique_posts_from_files(source_files)
    merged_path = district_folder / merge_filename
    save_json(merged_path, combined)
    return merged_path


def export_province_combined_backups(output_dir: Path) -> List[Path]:
    exported_paths: List[Path] = []

    for entry in os.scandir(output_dir):
        if not entry.is_dir():
            continue

        folder_name = entry.name
        folder_path = Path(entry.path)

        if folder_name.endswith("_districts"):
            province_slug = folder_name[:-10]
            source_files: List[Path] = []
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.endswith("_raw_posts.json"):
                        source_files.append(Path(root) / file)
            if not source_files:
                continue
            combined = collect_unique_posts_from_files(source_files)
            combined_path = folder_path / f"{province_slug}_all_districts_raw_posts.json"
            save_json(combined_path, combined)
            exported_paths.append(combined_path)
        else:
            source_files = [Path(item.path) for item in os.scandir(folder_path) if item.is_file() and item.name.endswith("_raw_posts.json")]
            if not source_files:
                continue
            combined = collect_unique_posts_from_files(source_files)
            combined_path = folder_path / f"{folder_name}_all_categories_raw_posts.json"
            save_json(combined_path, combined)
            exported_paths.append(combined_path)

    return exported_paths


def build_combined_backup_data(output_dir: Path) -> List[Dict[str, Any]]:
    file_paths: List[Path] = []
    for root, _, files in os.walk(output_dir):
        for file in files:
            if file.endswith("_raw_posts.json"):
                file_paths.append(Path(root) / file)
    return collect_unique_posts_from_files(file_paths)


def export_all_backups(output_dir: Path) -> Tuple[Path, List[Path], int]:
    province_combined_paths = export_province_combined_backups(output_dir)
    combined = build_combined_backup_data(output_dir)
    combined_output_file = output_dir / "ALL_DATA.json"
    save_json(combined_output_file, combined)
    return combined_output_file, province_combined_paths, len(combined)


def scrape_topic_single_page(
    scraper: cloudscraper.CloudScraper,
    base_url: str,
    start_page: int,
    visited_pages: set,
    visited_pages_file: Path,
) -> Tuple[str, List[Dict[str, Any]]]:
    all_entries: List[Dict[str, Any]] = []
    target_url = paged_url(base_url, start_page)

    if target_url in visited_pages:
        return "", []

    html = fetch_url(scraper, target_url)
    if not html:
        return "", []

    soup = BeautifulSoup(html, "html.parser")
    title = extract_title(soup)
    entries = extract_entries(soup)
    for entry in entries:
        entry["page"] = start_page
        all_entries.append(entry)

    visited_pages.add(target_url)
    save_set_to_csv(visited_pages, visited_pages_file, "url")
    return title, all_entries


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


def input_to_tasks(input_data: Any) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(input_data, dict) or not input_data:
        return None

    if "requests" in input_data:
        raw_requests = input_data.get("requests")
        if not isinstance(raw_requests, list):
            raise ValueError("input_data['requests'] must be a list")
        tasks: List[Dict[str, Any]] = []
        for item in raw_requests:
            if not isinstance(item, dict):
                raise ValueError("Each request must be a dictionary")
            level = str(item.get("level", "district")).strip().lower() or "district"
            province = str(item.get("province", "")).strip()
            district = item.get("district")
            district = str(district).strip() if district not in (None, "", "None") else None
            category = str(item.get("category", "")).strip()
            if not province or not category:
                raise ValueError("Each request must include province and category")
            if level == "district" and not district:
                raise ValueError("District-level request must include district")
            tasks.append(
                {
                    "level": level,
                    "province": province,
                    "district": district,
                    "category": category,
                    "num_results": int(item.get("num_results") or (DISTRICT_RESULTS_COUNT if level == "district" else PROVINCE_RESULTS_COUNT)),
                }
            )
        return tasks

    if "province" in input_data or "category" in input_data or "district" in input_data:
        level = str(input_data.get("level", "district")).strip().lower() or "district"
        province = str(input_data.get("province", "")).strip()
        district = input_data.get("district")
        district = str(district).strip() if district not in (None, "", "None") else None
        category = str(input_data.get("category", "")).strip()
        if not province or not category:
            raise ValueError("input_data must include province and category when using direct parameter mode")
        if level == "district" and not district:
            raise ValueError("District-level parameter mode must include district")
        return [
            {
                "level": level,
                "province": province,
                "district": district,
                "category": category,
                "num_results": int(input_data.get("num_results") or (DISTRICT_RESULTS_COUNT if level == "district" else PROVINCE_RESULTS_COUNT)),
            }
        ]

    return None


def run(input_data: Any, context: Dict[str, Any]) -> Any:
    base_dir = Path(context.get("base_dir") or ".")
    output_dir = base_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    visited_pages_file = base_dir / "state" / "visited_pages.csv"
    completed_topics_file = base_dir / "state" / "completed_topics.csv"
    completed_requests_file = base_dir / "state" / "completed_requests.csv"

    ensure_csv_with_header(visited_pages_file, "url")
    ensure_csv_with_header(completed_topics_file, "key")
    ensure_csv_with_header(completed_requests_file, "key")

    include_province_level = False
    max_topics = None
    if isinstance(input_data, dict):
        include_province_level = bool(input_data.get("include_province_level", False))
        max_topics = input_data.get("max_topics")

    explicit_tasks = input_to_tasks(input_data)

    api_keys = load_api_keys(base_dir)
    scraper = create_scraper()

    all_tasks = explicit_tasks if explicit_tasks is not None else generate_tasks(
        base_dir, include_province_level=include_province_level
    )
    mode = "input" if explicit_tasks is not None else "automatic"

    visited_pages = load_csv_column_set(visited_pages_file, "url")
    completed_topics = load_csv_column_set(completed_topics_file, "key")
    completed_requests = load_csv_column_set(completed_requests_file, "key")

    remaining_tasks: List[Dict[str, Any]] = []
    skipped_completed_requests = 0
    skipped_completed_topics = 0

    for task in all_tasks:
        query = build_query(task["level"], task["province"], task["district"], task["category"])
        request_key = make_request_key(task["level"], task["province"], task["district"], task["category"])
        task["query"] = query
        task["fetch_key"] = make_fetch_key(task["level"], task["province"], task["district"], task["category"])
        task["request_key"] = request_key

        if request_key in completed_requests:
            skipped_completed_requests += 1
            continue

        remaining_tasks.append(task)

    if max_topics is not None:
        remaining_tasks = remaining_tasks[: int(max_topics)]

    total_tasks = len(remaining_tasks)
    fetched_topics = 0
    processed_topics = 0
    total_new_posts = 0
    total_urls_found = 0
    used_key_indexes = set()
    written_paths: List[str] = []
    merged_paths: List[str] = []

    grouped_topics: Dict[Tuple[str, str], List[Tuple[str, int, str, str, Optional[str], str, str]]] = defaultdict(list)

    for task in remaining_tasks:
        links, key_index = search_google(task["query"], task["num_results"], api_keys)
        used_key_indexes.add(key_index)
        total_urls_found += len(links)
        fetched_topics += 1

        if not links:
            completed_requests.add(task["request_key"])
            save_set_to_csv(completed_requests, completed_requests_file, "key")
            time.sleep(REQUEST_DELAY)
            continue

        added_any_topic = False
        for raw_url in links:
            raw_url = str(raw_url).strip()
            if not raw_url:
                continue
            base_url = normalize_topic_url(raw_url)
            start_page = extract_start_page(raw_url)
            completed_key = make_completed_key(base_url, task["level"], task["province"], task["district"], task["category"])
            if completed_key in completed_topics:
                skipped_completed_topics += 1
                continue

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

        if not added_any_topic:
            completed_requests.add(task["request_key"])
            save_set_to_csv(completed_requests, completed_requests_file, "key")

        time.sleep(REQUEST_DELAY)

    request_success: Dict[str, bool] = {}
    for task in remaining_tasks:
        request_success[task["request_key"]] = False

    for (province, district_str), group_items in grouped_topics.items():
        raw_posts_by_bucket: Dict[Tuple[str, str, Optional[str], str], List[Dict[str, Any]]] = defaultdict(list)

        def _fetch_one(item: Tuple[str, int, str, str, Optional[str], str, str]):
            base_url, start_page, level, prov, dist, category, request_key = item
            completed_key = make_completed_key(base_url, level, prov, dist, category)
            if completed_key in completed_topics:
                return None
            _, entries = scrape_topic_single_page(scraper, base_url, start_page, visited_pages, visited_pages_file)
            return base_url, level, prov, dist, category, entries, completed_key, request_key

        with ThreadPoolExecutor(max_workers=PARALLEL_FETCH_WORKERS) as executor:
            futures = [executor.submit(_fetch_one, item) for item in group_items]
            for future in as_completed(futures):
                result = future.result()
                if result is None:
                    continue

                base_url, level, prov, dist, category, entries, completed_key, request_key = result
                completed_topics.add(completed_key)
                save_set_to_csv(completed_topics, completed_topics_file, "key")

                collected_at_iso = utc_now_iso()
                if entries:
                    raw_posts = convert_entries_to_raw_posts(base_url, level, prov, dist, category, entries, collected_at_iso)
                    total_new_posts += len(raw_posts)
                    processed_topics += 1
                    bucket = (level, prov, dist, category)
                    raw_posts_by_bucket[bucket].extend(raw_posts)

                request_success[request_key] = True

        if raw_posts_by_bucket:
            paths = export_raw_posts(output_dir, raw_posts_by_bucket)
            written_paths.extend([str(p) for p in paths])
            merged = export_district_merged(output_dir, province, district_str or None)
            if merged:
                merged_paths.append(str(merged))

    for request_key in request_success.keys():
        completed_requests.add(request_key)
    save_set_to_csv(completed_requests, completed_requests_file, "key")

    combined_output_file, province_combined_paths, combined_post_count = export_all_backups(output_dir)

    return {
        "step": STEP_NAME,
        "status": "ok",
        "mode": mode,
        "base_dir": str(base_dir),
        "output_dir": str(output_dir),
        "combined_output_file": str(combined_output_file),
        "province_combined_files": [str(p) for p in province_combined_paths],
        "district_merged_files": merged_paths,
        "written_category_files": written_paths,
        "total_generated_tasks": len(all_tasks),
        "run_task_count": total_tasks,
        "fetched_query_count": fetched_topics,
        "processed_topic_count": processed_topics,
        "skipped_completed_requests": skipped_completed_requests,
        "skipped_completed_topics": skipped_completed_topics,
        "total_urls_found": total_urls_found,
        "new_posts_in_this_run": total_new_posts,
        "combined_post_count": combined_post_count,
        "used_api_key_count": len(used_key_indexes),
        "resume_files": {
            "visited_pages_file": str(visited_pages_file),
            "completed_topics_file": str(completed_topics_file),
            "completed_requests_file": str(completed_requests_file),
        },
    }
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run fetch step in automatic or targeted mode")
    parser.add_argument("--input", default=None, help="Optional JSON input for targeted mode")
    parser.add_argument("--max-topics", type=int, default=None, help="Optional limit on requests to run")
    parser.add_argument("--include-province-level", action="store_true", help="Include province-level tasks in automatic mode")
    args = parser.parse_args()

    demo_input = None
    if args.input:
        try:
            demo_input = json.loads(args.input)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON for --input: {exc}")

    if demo_input is None:
        demo_input = {}

    if args.max_topics is not None:
        demo_input["max_topics"] = args.max_topics
    if args.include_province_level:
        demo_input["include_province_level"] = True

    demo_context = {"base_dir": ".", "db": None, "mongo_client": None}
    print(json.dumps(run(demo_input, demo_context), indent=2, ensure_ascii=False))
