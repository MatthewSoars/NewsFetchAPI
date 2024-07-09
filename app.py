from fastapi import FastAPI, BackgroundTasks, Query, HTTPException
import httpx
import feedparser
from datetime import datetime
import asyncio
from typing import List, Dict, Any, Optional
import logging
import joblib
import os
from pydantic import BaseModel
from urllib.parse import urlparse
import socket
from ip2geotools.databases.noncommercial import DbIpCity

app = FastAPI()

combined_feed: List[Dict[str, Any]] = []
denied_urls: List[str] = []
feed_lock = asyncio.Lock()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None
vectorizer = None


@app.on_event("startup")
async def startup_event() -> None:
    global model, vectorizer
    try:
        logger.info("Starting the application and loading the model and vectorizer...")

        model_path = 'text_classifier_model.pkl'
        vectorizer_path = 'tfidf_vectorizer.pkl'

        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return
        if not os.path.exists(vectorizer_path):
            logger.error(f"Vectorizer file not found: {vectorizer_path}")
            return

        logger.info("Model and vectorizer files found. Attempting to load...")

        try:
            model = joblib.load(model_path)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        try:
            vectorizer = joblib.load(vectorizer_path)
            logger.info("Vectorizer loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load vectorizer: {e}")
            raise

        logger.info("Updating combined feed...")
        await update_combined_feed()
        logger.info("Combined feed updated successfully.")

        logger.info("Starting the background task for refreshing feed...")
        asyncio.create_task(refresh_feed_background_task())
        logger.info("Background task started successfully.")
    except Exception as e:
        logger.error(f"Error during startup event: {e}")
        raise


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


def get_continent_from_ip(ip: str) -> Optional[str]:
    try:
        response = DbIpCity.get(ip, api_key='free')
        country_code = response.country
        continent_mapping = {
            'AF': 'Africa',
            'AN': 'Antarctica',
            'AS': 'Asia',
            'EU': 'Europe',
            'NA': 'North America',
            'OC': 'Oceania',
            'SA': 'South America'
        }
        return continent_mapping.get(country_code)
    except Exception as e:
        logger.error(f"Error getting continent from IP: {e}")
    return None


def get_domain_location(url: str) -> Optional[str]:
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        ip = socket.gethostbyname(domain)
        return get_continent_from_ip(ip)
    except Exception as e:
        logger.error(f"Error getting domain location: {e}")
    return None


def parse_feed_entry(entry: Dict[str, Any], rss_feed_url: str) -> Dict[str, Any]:
    global model, vectorizer

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

    text = title + ' ' + description
    X = vectorizer.transform([text])
    classification = model.predict(X)[0]

    location = get_domain_location(rss_feed_url)

    return {
        "title": title,
        "description": description,
        "pub_date": formatted_pub_date,
        "url": rss_feed_url,
        "image_link": image_link or "",
        "classification": classification,
        "location": location or "Unknown"
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
        classifications: Optional[List[str]] = Query(None)
) -> Dict[str, Any]:
    async with feed_lock:
        filtered_feed = combined_feed
        if classifications:
            filtered_feed = [item for item in combined_feed if item['classification'] in classifications]

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
    classification = model.predict(X)[0]
    return {"classification": classification}


async def refresh_feed_background_task() -> None:
    while True:
        try:
            logger.info("Refreshing combined feed in background task...")
            await update_combined_feed()
            logger.info("Combined feed refreshed successfully.")
        except Exception as e:
            logger.error(f"Error in background feed refresh task: {e}")
        await asyncio.sleep(600)
