import os
import hashlib
import httpx
import json
from fastapi import FastAPI, BackgroundTasks, Query, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
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
from functools import lru_cache
from starlette.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Global Variables
combined_feed: List[Dict[str, Any]] = []
denied_urls: List[str] = []
feed_lock = asyncio.Lock()
model_lock = asyncio.Lock()

# Logger Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Machine Learning Models
binary_model = None
detailed_model = None
vectorizer = None
mlb = None

# URLs to download the files from DigitalOcean Spaces
BINARY_MODEL_URL = "https://construct-api.ams3.cdn.digitaloceanspaces.com/v5_binary_classifier_model.pkl"
DETAILED_MODEL_URL = "https://construct-api.ams3.cdn.digitaloceanspaces.com/v5_detailed_classifier_model.pkl"
VECTOR_URL = "https://construct-api.ams3.cdn.digitaloceanspaces.com/v5_tfidf_vectorizer.pkl"
MLB_URL = "https://construct-api.ams3.cdn.digitaloceanspaces.com/v5_mlb.pkl"

# Expected file hashes
EXPECTED_HASHES = {
    BINARY_MODEL_URL: "917b6b5ab244237b19a1ea4f8e57864a64bd80ca71f89539ae6a6451f82fb1e8",
    DETAILED_MODEL_URL: "548c60cd44b3a5ec0e44028630f0aa7e561d2eecf1a8b7550713006b9d8a33f7",
    VECTOR_URL: "dad6ec379eb9d703bb74bb301841b443280824b30e3370643538f01bed8de47e",
    MLB_URL: "273f02d42a1e334d1c85f3f383fbf286856663a6d3d41b77cbcb23286716ff58",
}

CACHE_FILE = 'domain_country_cache.txt'
COMBINED_FEED_FILE = 'combined_feed.json'
TLD_COUNTRY_MAP_FILE = 'tld_country_map.txt'
COUNTRY_CONTINENT_MAP_FILE = 'country_continent_map.txt'

# Cache and mappings
tld_country_map: Dict[str, str] = {}
country_continent_map: Dict[str, Dict[str, str]] = {}
domain_country_cache: Dict[str, str] = {}

# New RSS links file
RSS_LINKS_FILE = "Accepted.txt"

@app.get("/rss_links", response_model=List[str])
async def get_rss_links() -> List[str]:
    """
    Get the current list of RSS feed links.
    """
    if not os.path.exists(RSS_LINKS_FILE):
        return []
    with open(RSS_LINKS_FILE, "r") as file:
        links = file.read().splitlines()
    return links

@app.post("/rss_links")
async def add_rss_link(link: str) -> Dict[str, str]:
    """
    Add a new RSS feed link to the list.
    """
    if not link:
        raise HTTPException(status_code=400, detail="Link cannot be empty.")
    if not os.path.exists(RSS_LINKS_FILE):
        with open(RSS_LINKS_FILE, "w") as file:
            file.write(link + "\n")
    else:
        with open(RSS_LINKS_FILE, "r") as file:
            links = file.read().splitlines()
        if link in links:
            raise HTTPException(status_code=400, detail="Link already exists.")
        links.append(link)
        with open(RSS_LINKS_FILE, "w") as file:
            file.write("\n".join(links) + "\n")
    return {"message": "Link added successfully."}

@app.delete("/rss_links")
async def remove_rss_link(link: str) -> Dict[str, str]:
    """
    Remove an RSS feed link from the list.
    """
    if not link:
        raise HTTPException(status_code=400, detail="Link cannot be empty.")
    if not os.path.exists(RSS_LINKS_FILE):
        raise HTTPException(status_code=404, detail="No RSS links found.")
    with open(RSS_LINKS_FILE, "r") as file:
        links = file.read().splitlines()
    if link not in links:
        raise HTTPException(status_code=404, detail="Link not found.")
    links.remove(link)
    with open(RSS_LINKS_FILE, "w") as file:
        file.write("\n".join(links) + "\n")
    return {"message": "Link removed successfully."}

@app.get("/manage_links", response_class=HTMLResponse)
async def manage_links(request: Request):
    """
    Render the page to manage RSS feed links.
    """
    if not os.path.exists(RSS_LINKS_FILE):
        links = []
    else:
        with open(RSS_LINKS_FILE, "r") as file:
            links = file.read().splitlines()
    return templates.TemplateResponse("manage_links.html", {"request": request, "links": links})

@app.post("/manage_links/add")
async def add_link(request: Request, link: str = Form(...)):
    """
    Add an RSS link through the web interface.
    """
    try:
        await add_rss_link(link)
    except HTTPException as e:
        return templates.TemplateResponse("manage_links.html", {"request": request, "links": await get_rss_links(), "error": e.detail})
    return templates.TemplateResponse("manage_links.html", {"request": request, "links": await get_rss_links(), "message": "Link added successfully."})

@app.post("/manage_links/delete")
async def delete_link(request: Request, link: str = Form(...)):
    """
    Remove an RSS link through the web interface.
    """
    try:
        await remove_rss_link(link)
    except HTTPException as e:
        return templates.TemplateResponse("manage_links.html", {"request": request, "links": await get_rss_links(), "error": e.detail})
    return templates.TemplateResponse("manage_links.html", {"request": request, "links": await get_rss_links(), "message": "Link removed successfully."})

def generate_file_hash(filepath: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


async def download_file(url: str, filepath: str) -> None:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        with open(filepath, 'wb') as f:
            f.write(response.content)


async def verify_and_load_models() -> None:
    global binary_model, detailed_model, vectorizer, mlb
    file_paths = {
        BINARY_MODEL_URL: 'v3_binary_classifier_model.pkl',
        DETAILED_MODEL_URL: 'v3_detailed_classifier_model.pkl',
        VECTOR_URL: 'v3_tfidf_vectorizer.pkl',
        MLB_URL: 'v3_mlb.pkl'
    }

    for url, path in file_paths.items():
        await download_file(url, path)
        if generate_file_hash(path) != EXPECTED_HASHES[url]:
            raise ValueError(f"File from {url} is corrupted.")

    binary_model = joblib.load(file_paths[BINARY_MODEL_URL])
    detailed_model = joblib.load(file_paths[DETAILED_MODEL_URL])
    vectorizer = joblib.load(file_paths[VECTOR_URL])
    mlb = joblib.load(file_paths[MLB_URL])


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


def get_root_domain(url: str) -> str:
    parsed_url = urllib.parse.urlparse(url)
    netloc = parsed_url.netloc
    extract_result = tldextract.extract(netloc)
    root_domain = f"{extract_result.domain}.{extract_result.suffix}"
    return root_domain


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


@lru_cache(maxsize=128)
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

    all_detailed_classifications = mlb.inverse_transform(detailed_model.predict(X))

    parsed_entries = []
    for entry, is_const, detailed_classification in zip(entries, is_construction, all_detailed_classifications):
        if not is_const:
            continue

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

        # Assign a general "general construction" classification if the detailed classification is empty
        classification_str = ", ".join([url_friendly_format(cls.strip()) for cls in
                                        detailed_classification]) if detailed_classification else "general construction"

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


@app.on_event("startup")
async def startup_event() -> None:
    global tld_country_map, country_continent_map, domain_country_cache, combined_feed
    try:
        logger.info("Verifying and loading models...")
        await verify_and_load_models()
        logger.info("Models loaded successfully.")

        logger.info("Loading country maps and cache...")
        tld_country_map = load_country_map(TLD_COUNTRY_MAP_FILE)
        country_continent_map = load_country_map(COUNTRY_CONTINENT_MAP_FILE)
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

        # Calculate unique classifications in the current filtered feed
        unique_classifications = set([item['classification'] for item in filtered_feed])

        total_items = len(filtered_feed)
        return {
            "total_items": total_items,
            "classifications": list(unique_classifications),
            "continents": continents or "All",
            "feed": filtered_feed
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
    async with model_lock:
        text = article.title + ' ' + article.description
        X = vectorizer.transform([text])
        is_construction = binary_model.predict(X)[0]
        if is_construction == 1:
            original_classification = mlb.inverse_transform(detailed_model.predict(X))[0]
            classification = ", ".join([url_friendly_format(cls.strip()) for cls in
                                        original_classification]) if original_classification else "general construction"
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