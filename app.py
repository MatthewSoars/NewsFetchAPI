import os
import hashlib
import httpx
import json
from fastapi import FastAPI, BackgroundTasks, Query, HTTPException
import joblib
import logging
import feedparser
from datetime import datetime
import asyncio
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import urllib.parse
import tldextract
import whois
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

combined_feed: List[Dict[str, Any]] = []
denied_urls: List[str] = []
feed_lock = asyncio.Lock()
model_lock = asyncio.Lock()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

binary_model = None
detailed_model = None
vectorizer = None
mlb = None

# URLs to download the files from DigitalOcean Spaces
BINARY_MODEL_URL = "https://construct-api.ams3.cdn.digitaloceanspaces.com/v3_binary_classifier_model.pkl"
DETAILED_MODEL_URL = "https://construct-api.ams3.cdn.digitaloceanspaces.com/v3_detailed_classifier_model.pkl"
VECTOR_URL = "https://construct-api.ams3.cdn.digitaloceanspaces.com/v3_tfidf_vectorizer.pkl"
MLB_URL = "https://construct-api.ams3.cdn.digitaloceanspaces.com/v3_mlb.pkl"

# Expected file hashes
EXPECTED_BINARY_MODEL_HASH = "6da8dd301d1c15447461d1c31958d9712cc0ecc6e1583885f490744eb7cee79a"
EXPECTED_DETAILED_MODEL_HASH = "5a99719406cbe9f8635c1fd553bc775d635435939c37d3eb3e84f46f803c8927"
EXPECTED_VECTOR_HASH = "268e4882fffb12c2eaef1e26100b4be261f2b53ec892b1013c97a24763a1b1b6"
EXPECTED_MLB_HASH = "1885a2719893874ab63ab45e061910d0ae978cdf89e29ce85ef40f38c0fc625d"

CACHE_FILE = 'domain_country_cache.txt'
COMBINED_FEED_FILE = 'combined_feed.json'
tld_country_map: Dict[str, str] = {}
country_continent_map: Dict[str, Dict[str, str]] = {}
domain_country_cache: Dict[str, str] = {}


def generate_file_hash(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


async def download_file(url, filepath):
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        with open(filepath, 'wb') as f:
            f.write(response.content)


@app.on_event("startup")
async def startup_event() -> None:
    global binary_model, detailed_model, vectorizer, mlb, tld_country_map, country_continent_map, domain_country_cache, combined_feed
    try:
        binary_model_path = 'v3_binary_classifier_model.pkl'
        detailed_model_path = 'v3_detailed_classifier_model.pkl'
        vectorizer_path = 'v3_tfidf_vectorizer.pkl'
        mlb_path = 'v3_mlb.pkl'
        tld_country_map_path = 'tld_country_map.txt'
        country_continent_map_path = 'country_continent_map.txt'

        logger.info("Downloading binary model...")
        await download_file(BINARY_MODEL_URL, binary_model_path)
        logger.info("Binary model downloaded.")

        logger.info("Downloading detailed model...")
        await download_file(DETAILED_MODEL_URL, detailed_model_path)
        logger.info("Detailed model downloaded.")

        logger.info("Downloading vectorizer...")
        await download_file(VECTOR_URL, vectorizer_path)
        logger.info("Vectorizer downloaded.")

        logger.info("Downloading MultiLabelBinarizer...")
        await download_file(MLB_URL, mlb_path)
        logger.info("MultiLabelBinarizer downloaded.")

        logger.info("Verifying file integrity...")
        if generate_file_hash(binary_model_path) != EXPECTED_BINARY_MODEL_HASH:
            raise ValueError("Binary model file is corrupted.")
        if generate_file_hash(detailed_model_path) != EXPECTED_DETAILED_MODEL_HASH:
            raise ValueError("Detailed model file is corrupted.")
        if generate_file_hash(vectorizer_path) != EXPECTED_VECTOR_HASH:
            raise ValueError("Vectorizer file is corrupted.")
        if generate_file_hash(mlb_path) != EXPECTED_MLB_HASH:
            raise ValueError("MultiLabelBinarizer file is corrupted.")
        logger.info("File integrity verified.")

        logger.info("Loading binary model...")
        binary_model = joblib.load(binary_model_path)
        logger.info("Binary model loaded successfully.")

        logger.info("Loading detailed model...")
        detailed_model = joblib.load(detailed_model_path)
        logger.info("Detailed model loaded successfully.")

        logger.info("Loading vectorizer...")
        vectorizer = joblib.load(vectorizer_path)
        logger.info("Vectorizer loaded successfully.")

        logger.info("Loading MultiLabelBinarizer...")
        mlb = joblib.load(mlb_path)
        logger.info("MultiLabelBinarizer loaded successfully.")

        logger.info("Loading country maps and cache...")
        if not os.path.exists(tld_country_map_path):
            logger.error(f"TLD country map file not found: {tld_country_map_path}")
            return
        if not os.path.exists(country_continent_map_path):
            logger.error(f"Country continent map file not found: {country_continent_map_path}")
            return

        tld_country_map = load_country_map(tld_country_map_path)
        country_continent_map = load_country_map(country_continent_map_path)
        domain_country_cache = load_cache()
        logger.info("Country maps and cache loaded successfully.")

        logger.info("Loading combined feed from file...")
        combined_feed = load_combined_feed()
        logger.info("Combined feed loaded successfully.")

        logger.info("Starting the background task for refreshing feed...")
        asyncio.create_task(refresh_feed_background_task())
        logger.info("Background task started successfully.")
    except Exception as e:
        logger.error(f"Error during startup event: {e}")
        raise


def get_root_domain(url: str) -> str:
    parsed_url = urllib.parse.urlparse(url)
    netloc = parsed_url.netloc
    extract_result = tldextract.extract(netloc)
    root_domain = f"{extract_result.domain}.{extract_result.suffix}"
    return root_domain


def load_country_map(filename: str) -> Dict[str, Dict[str, str]]:
    country_map = {}
    try:
        with open(filename, 'r') as file:
            for line in file:
                code, country, continent = line.strip().split(',')
                country_map[country.strip()] = {'code': code.strip(), 'continent': continent.strip()}
    except Exception as e:
        logger.error(f"Error loading map from {filename}: {e}")
    return country_map


def load_cache() -> Dict[str, str]:
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as file:
            for line in file:
                domain, country = line.strip().split(',')
                cache[domain.strip()] = country.strip()
    return cache


def save_cache(cache: Dict[str, str]) -> None:
    with open(CACHE_FILE, 'w') as file:
        for domain, country in cache.items():
            file.write(f"{domain},{country}\n")


def get_country_from_url(url: str) -> str:
    global tld_country_map, domain_country_cache
    try:
        root_domain = get_root_domain(url)
        if root_domain in domain_country_cache:
            country = domain_country_cache[root_domain]
        else:
            tld = root_domain.split('.')[-1]
            country = tld_country_map.get(tld, "Unknown")
            if country == "Unknown":
                try:
                    w = whois.whois(root_domain)
                    country = w.country if w and w.country else "Unknown"
                except Exception as e:
                    logger.error(f"WHOIS lookup failed for {root_domain}: {e}")
            domain_country_cache[root_domain] = country
            save_cache(domain_country_cache)
        return country
    except Exception as e:
        logger.error(f"Error fetching country for {url}: {e}")
        return "Unknown"


def get_continent_from_country(country: str) -> str:
    global country_continent_map
    return country_continent_map.get(country, {}).get('continent', "unknown")


def url_friendly_format(text: str) -> str:
    return text.strip().lower().replace(" ", "-")


async def fetch_feed_data(rss_feed_url: str, headers: Dict[str, str]) -> Optional[bytes]:
    async with httpx.AsyncClient(follow_redirects=True, verify=False) as client:
        try:
            response = await client.get(rss_feed_url, headers=headers)
            response.raise_for_status()
            return response.content
        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to fetch the RSS feed for {rss_feed_url} with status code {e.response.status_code}")
            denied_urls.append(rss_feed_url)
        except httpx.RequestError as e:
            logger.error(f"Error fetching RSS feed: {e}")
            denied_urls.append(rss_feed_url)
    return None


async def parse_feed_entries(entries: List[Dict[str, Any]], rss_feed_url: str) -> List[Dict[str, Any]]:
    global binary_model, detailed_model, vectorizer, mlb

    if not entries:
        return []

    texts = [entry.get("title", "") + ' ' + entry.get("description", "") for entry in entries if
             entry.get("title") or entry.get("description")]
    if not texts:
        return []

    X = vectorizer.transform(texts)
    is_construction = binary_model.predict(X)

    if sum(is_construction) > 0:
        detailed_classifications = mlb.inverse_transform(detailed_model.predict(X[is_construction == 1]))
    else:
        detailed_classifications = []

    parsed_entries = []

    detailed_index = 0
    for entry, is_const in zip(entries, is_construction):
        title = entry.get("title", "")
        description = entry.get("description", "")
        pub_date_str = entry.get("published", "")

        pub_date = None
        formats_to_try = [
            "%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S", "%a, %d %b %Y %H:%M %z",
            "%a, %d %b %Y %H:%M", "%a, %d %b %Y %H:%M:%S GMT", "%Y-%m-%dT%H:%M:%S%z"
        ]
        for date_format in formats_to_try:
            try:
                pub_date = datetime.strptime(pub_date_str, date_format)
                break
            except ValueError:
                continue

        formatted_pub_date = pub_date.strftime("%Y-%m-%d %H:%M:%S") if pub_date else None

        image_link = None
        if "media_content" in entry:
            for media in entry["media_content"]:
                if "url" in media:
                    image_link = media["url"]
                    break

        if not image_link and "media_thumbnail" in entry:
            image_link = entry["media_thumbnail"][0]["url"]

        if not image_link and "links" in entry:
            for link in entry["links"]:
                if link["rel"] == "enclosure" and link["type"].startswith("image"):
                    image_link = link["href"]
                    break

        article_url = entry.get("link", rss_feed_url)

        if is_const:
            classification = detailed_classifications[detailed_index]
            detailed_index += 1
        else:
            classification = ["non-construction"]

        classification_str = ", ".join([url_friendly_format(cls.strip()) for cls in classification])

        country = get_country_from_url(article_url)
        continent = get_continent_from_country(country)

        combined_info = (
            f"Article Title: {title}, "
            f"Country: {country}, "
            f"Continent: {continent}, "
            f"Classification: {classification_str}"
        )
        logger.info(combined_info)

        parsed_entries.append({
            "title": title,
            "description": description,
            "pub_date": formatted_pub_date,
            "url": article_url,
            "image_link": image_link or "",
            "classification": classification_str,
            "country": country,
            "continent": continent,
            "is_construction": bool(is_const)
        })

    return parsed_entries


async def update_combined_feed() -> None:
    global combined_feed, denied_urls

    try:
        with open("Accepted.txt", "r") as file:
            rss_feed_urls = file.read().splitlines()
    except FileNotFoundError as e:
        logger.error(f"Accepted.txt not found: {e}")
        return

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }

    updated_feed = []
    updated_denied_urls = []

    tasks = []
    for rss_feed_url in rss_feed_urls:
        tasks.append(fetch_feed_data(rss_feed_url, headers))

    results = await asyncio.gather(*tasks)

    for rss_feed_url, feed_data in zip(rss_feed_urls, results):
        if feed_data:
            parsed_feed = feedparser.parse(feed_data)
            if parsed_feed.bozo:
                logger.error(f"Failed to parse feed {rss_feed_url}: {parsed_feed.bozo_exception}")
                updated_denied_urls.append(rss_feed_url)
                continue
            parsed_entries = await parse_feed_entries(parsed_feed.entries, rss_feed_url)
            updated_feed.extend(parsed_entries)

    async with feed_lock:
        combined_feed = updated_feed
        denied_urls = updated_denied_urls
        logger.info(f"Combined feed updated with {len(updated_feed)} entries. {len(updated_denied_urls)} feeds denied.")
        save_combined_feed(combined_feed)


def load_combined_feed() -> List[Dict[str, Any]]:
    if os.path.exists(COMBINED_FEED_FILE):
        with open(COMBINED_FEED_FILE, 'r') as file:
            return json.load(file)
    return []


def save_combined_feed(feed: List[Dict[str, Any]]) -> None:
    with open(COMBINED_FEED_FILE, 'w') as file:
        json.dump(feed, file)


@app.get("/combined_feed")
async def get_combined_feed(
        page: int = Query(1, ge=1),
        size: int = Query(10, ge=1),
        classifications: Optional[List[str]] = Query(None),
        continents: Optional[List[str]] = Query(None)
) -> Dict[str, Any]:
    async with feed_lock:
        filtered_feed = combined_feed
        if classifications:
            url_friendly_classifications = [url_friendly_format(cls) for cls in classifications]
            filtered_feed = [item for item in combined_feed if
                             url_friendly_format(item['classification']) in url_friendly_classifications]
        if continents:
            url_friendly_continents = [url_friendly_format(cont) for cont in continents]
            filtered_feed = [item for item in filtered_feed if
                             url_friendly_format(item['continent']) in url_friendly_continents]

        start_idx = (page - 1) * size
        end_idx = start_idx + size
        if start_idx >= len(filtered_feed):
            raise HTTPException(status_code=404, detail="Page not found")
        total_pages = (len(filtered_feed) + size - 1) // size  # Calculate total number of pages
        return {
            "page": page,
            "size": size,
            "total_items": len(filtered_feed),
            "total_pages": total_pages,
            "feed": filtered_feed[start_idx:end_idx]
        }


@app.get("/fullfeed")
async def get_full_feed(
        classifications: Optional[List[str]] = Query(None),
        continents: Optional[List[str]] = Query(None)
) -> List[Dict[str, Any]]:
    async with feed_lock:
        filtered_feed = combined_feed
        if classifications:
            url_friendly_classifications = [url_friendly_format(cls) for cls in classifications]
            filtered_feed = [item for item in combined_feed if
                             url_friendly_format(item['classification']) in url_friendly_classifications]
        if continents:
            url_friendly_continents = [url_friendly_format(cont) for cont in continents]
            filtered_feed = [item for item in filtered_feed if
                             url_friendly_format(item['continent']) in url_friendly_continents]
        return filtered_feed


@app.post("/refresh_feed")
async def refresh_feed(background_tasks: BackgroundTasks) -> Dict[str, str]:
    background_tasks.add_task(update_combined_feed)
    return {"message": "Feed refresh scheduled"}


class Article(BaseModel):
    title: str
    description: str


@app.post("/classify_article")
async def classify_article(article: Article) -> Dict[str, Any]:
    async with model_lock:
        text = article.title + ' ' + article.description
        X = vectorizer.transform([text])
        is_construction = binary_model.predict(X)[0]
        if is_construction == 1:
            original_classification = mlb.inverse_transform(detailed_model.predict(X))[0]
            classification = ", ".join([url_friendly_format(cls.strip()) for cls in original_classification])
        else:
            original_classification = ["non-construction"]
            classification = "non-construction"
    return {"classification": original_classification}


async def refresh_feed_background_task() -> None:
    while True:
        try:
            logger.info("Refreshing combined feed in background task...")
            await update_combined_feed()
            logger.info("Combined feed refreshed successfully.")
        except Exception as e:
            logger.error(f"Error in background feed refresh task: {e}")
        await asyncio.sleep(600)
