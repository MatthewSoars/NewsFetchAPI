import os
import hashlib
import httpx
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

app = FastAPI()

combined_feed: List[Dict[str, Any]] = []
denied_urls: List[str] = []
feed_lock = asyncio.Lock()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None
vectorizer = None
mlb = None

# URLs to download the files from DigitalOcean Spaces
MODEL_URL = "https://construct-api.ams3.cdn.digitaloceanspaces.com/text_classifier_model.pkl"
VECTOR_URL = "https://construct-api.ams3.cdn.digitaloceanspaces.com/tfidf_vectorizer.pkl"
MLB_URL = "https://construct-api.ams3.cdn.digitaloceanspaces.com/mlb.pkl"

# Expected file hashes
EXPECTED_MODEL_HASH = "eca04b84942bcb1aed6bd8f49a6b310b97c497b258006bd20d93860d9b81909b"
EXPECTED_VECTOR_HASH = "2fa997a8fe5fb0930fd0fa31c9e6d01202fe4389753ae866b09e41945797e243"
EXPECTED_MLB_HASH = "3ffeacff2262897dab960a184728ad03d26e84a102f96741dd4bc7e34a63e8ae"

CACHE_FILE = 'domain_country_cache.txt'
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
    global model, vectorizer, mlb, tld_country_map, country_continent_map, domain_country_cache
    try:
        model_path = 'text_classifier_model.pkl'
        vectorizer_path = 'tfidf_vectorizer.pkl'
        mlb_path = 'mlb.pkl'
        tld_country_map_path = 'tld_country_map.txt'
        country_continent_map_path = 'country_continent_map.txt'

        logger.info("Downloading model...")
        await download_file(MODEL_URL, model_path)
        logger.info("Model downloaded.")

        logger.info("Downloading vectorizer...")
        await download_file(VECTOR_URL, vectorizer_path)
        logger.info("Vectorizer downloaded.")

        logger.info("Downloading MultiLabelBinarizer...")
        await download_file(MLB_URL, mlb_path)
        logger.info("MultiLabelBinarizer downloaded.")

        logger.info("Verifying file integrity...")
        if generate_file_hash(model_path) != EXPECTED_MODEL_HASH:
            raise ValueError("Model file is corrupted.")
        if generate_file_hash(vectorizer_path) != EXPECTED_VECTOR_HASH:
            raise ValueError("Vectorizer file is corrupted.")
        if generate_file_hash(mlb_path) != EXPECTED_MLB_HASH:
            raise ValueError("MultiLabelBinarizer file is corrupted.")
        logger.info("File integrity verified.")

        logger.info("Loading model...")
        model = joblib.load(model_path)
        logger.info("Model loaded successfully.")

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

        logger.info("Updating combined feed...")
        await update_combined_feed()
        logger.info("Combined feed updated successfully.")

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
        logger.info(f"Root domain extracted: {root_domain}")

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
        logger.info(f"Country determined: {country}")
        return country
    except Exception as e:
        logger.error(f"Error fetching country for {url}: {e}")
        return "Unknown"


def get_continent_from_country(country: str) -> str:
    global country_continent_map
    return country_continent_map.get(country, {}).get('continent', "unknown")


def url_friendly_format(text: str) -> str:
    return text.lower().replace(" ", "-")


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


def parse_feed_entry(entry: Dict[str, Any], rss_feed_url: str) -> Dict[str, Any]:
    global model, vectorizer, mlb

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

    text = title + ' ' + description
    X = vectorizer.transform([text])
    original_classification = mlb.inverse_transform(model.predict(X))[0]
    classification = url_friendly_format(", ".join(original_classification))

    country = get_country_from_url(article_url)
    continent = get_continent_from_country(country)

    logger.info(f"Article Title: {title}, Country: {country}, Continent: {continent}")

    return {
        "title": title,
        "description": description,
        "pub_date": formatted_pub_date,
        "url": article_url,
        "image_link": image_link or "",
        "classification": classification,
        "country": country,
        "continent": continent
    }


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

    for rss_feed_url in rss_feed_urls:
        feed_data = await fetch_feed_data(rss_feed_url, headers)
        if feed_data:
            parsed_feed = feedparser.parse(feed_data)
            if parsed_feed.bozo:
                logger.error(f"Failed to parse feed {rss_feed_url}: {parsed_feed.bozo_exception}")
                updated_denied_urls.append(rss_feed_url)
                continue
            for entry in parsed_feed.entries:
                try:
                    feed_entry = parse_feed_entry(entry, rss_feed_url)
                    updated_feed.append(feed_entry)
                except Exception as e:
                    logger.error(f"Error parsing entry in feed {rss_feed_url}: {e}")

    async with feed_lock:
        combined_feed = updated_feed
        denied_urls = updated_denied_urls
        logger.info(f"Combined feed updated with {len(updated_feed)} entries. {len(updated_denied_urls)} feeds denied.")


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
            filtered_feed = [item for item in combined_feed if url_friendly_format(item['classification']) in url_friendly_classifications]
        if continents:
            url_friendly_continents = [url_friendly_format(cont) for cont in continents]
            filtered_feed = [item for item in filtered_feed if url_friendly_format(item['continent']) in url_friendly_continents]

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


@app.post("/refresh_feed")
async def refresh_feed(background_tasks: BackgroundTasks) -> Dict[str, str]:
    background_tasks.add_task(update_combined_feed)
    return {"message": "Feed refresh scheduled"}


class Article(BaseModel):
    title: str
    description: str


@app.post("/classify_article")
async def classify_article(article: Article) -> Dict[str, Any]:
    text = article.title + ' ' + article.description
    X = vectorizer.transform([text])
    original_classification = mlb.inverse_transform(model.predict(X))[0]
    classification = url_friendly_format(", ".join(original_classification))
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
