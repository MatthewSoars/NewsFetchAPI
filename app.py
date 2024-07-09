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
from ip2geotools.databases.noncommercial import DbIpCity
import socket
import urllib.parse
import tldextract

app = FastAPI()

combined_feed: List[Dict[str, Any]] = []
denied_urls: List[str] = []
feed_lock = asyncio.Lock()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None
vectorizer = None

continent_mapping = {
    "AF": "Africa",
    "AS": "Asia",
    "EU": "Europe",
    "NA": "North America",
    "SA": "South America",
    "OC": "Oceania",
    "AN": "Antarctica"
}

country_to_continent = {
    "DZ": "AF", "AO": "AF", "BJ": "AF", "BW": "AF", "BF": "AF", "BI": "AF", "CM": "AF", "CV": "AF", "CF": "AF",
    "TD": "AF", "KM": "AF", "CG": "AF", "DJ": "AF", "EG": "AF", "GQ": "AF", "ER": "AF", "ET": "AF", "GA": "AF",
    "GM": "AF", "GH": "AF", "GN": "AF", "GW": "AF", "CI": "AF", "KE": "AF", "LS": "AF", "LR": "AF", "LY": "AF",
    "MG": "AF", "MW": "AF", "ML": "AF", "MR": "AF", "MU": "AF", "YT": "AF", "MA": "AF", "MZ": "AF", "NA": "AF",
    "NE": "AF", "NG": "AF", "RE": "AF", "RW": "AF", "SH": "AF", "ST": "AF", "SN": "AF", "SC": "AF", "SL": "AF",
    "SO": "AF", "ZA": "AF", "SS": "AF", "SD": "AF", "SZ": "AF", "TZ": "AF", "TG": "AF", "TN": "AF", "UG": "AF",
    "EH": "AF", "ZM": "AF", "ZW": "AF", "AQ": "AN", "AE": "AS", "AM": "AS", "AZ": "AS", "BH": "AS", "BD": "AS",
    "BT": "AS", "BN": "AS", "KH": "AS", "CN": "AS", "CY": "AS", "GE": "AS", "IN": "AS", "ID": "AS", "IR": "AS",
    "IQ": "AS", "IL": "AS", "JP": "AS", "JO": "AS", "KZ": "AS", "KW": "AS", "KG": "AS", "LA": "AS", "LB": "AS",
    "MY": "AS", "MV": "AS", "MN": "AS", "MM": "AS", "NP": "AS", "KP": "AS", "OM": "AS", "PK": "AS", "PH": "AS",
    "QA": "AS", "SA": "AS", "SG": "AS", "KR": "AS", "LK": "AS", "SY": "AS", "TW": "AS", "TJ": "AS", "TH": "AS",
    "TL": "AS", "TR": "AS", "TM": "AS", "AE": "AS", "UZ": "AS", "VN": "AS", "YE": "AS", "AD": "EU", "AL": "EU",
    "AT": "EU", "BY": "EU", "BE": "EU", "BA": "EU", "BG": "EU", "HR": "EU", "CY": "EU", "CZ": "EU", "DK": "EU",
    "EE": "EU", "FO": "EU", "FI": "EU", "FR": "EU", "DE": "EU", "GI": "EU", "GR": "EU", "HU": "EU", "IS": "EU",
    "IE": "EU", "IT": "EU", "LV": "EU", "LI": "EU", "LT": "EU", "LU": "EU", "MT": "EU", "MD": "EU", "MC": "EU",
    "ME": "EU", "NL": "EU", "MK": "EU", "NO": "EU", "PL": "EU", "PT": "EU", "RO": "EU", "RU": "EU", "SM": "EU",
    "RS": "EU", "SK": "EU", "SI": "EU", "ES": "EU", "SE": "EU", "CH": "EU", "UA": "EU", "GB": "EU", "VA": "EU",
    "AG": "NA", "BS": "NA", "BB": "NA", "BZ": "NA", "BM": "NA", "CA": "NA", "CR": "NA", "CU": "NA", "DM": "NA",
    "DO": "NA", "SV": "NA", "GD": "NA", "GT": "NA", "HT": "NA", "HN": "NA", "JM": "NA", "MX": "NA", "NI": "NA",
    "PA": "NA", "PR": "NA", "KN": "NA", "LC": "NA", "VC": "NA", "TT": "NA", "US": "NA", "UM": "NA", "VG": "NA",
    "VI": "NA", "AR": "SA", "BO": "SA", "BR": "SA", "CL": "SA", "CO": "SA", "EC": "SA", "FK": "SA", "GF": "SA",
    "GY": "SA", "PY": "SA", "PE": "SA", "SR": "SA", "UY": "SA", "VE": "SA", "AS": "OC", "AU": "OC", "CK": "OC",
    "FJ": "OC", "PF": "OC", "GU": "OC", "KI": "OC", "MH": "OC", "FM": "OC", "NR": "OC", "NC": "OC", "NZ": "OC",
    "NU": "OC", "NF": "OC", "MP": "OC", "PW": "OC", "PG": "OC", "PN": "OC", "WS": "OC", "SB": "OC", "TK": "OC",
    "TO": "OC", "TV": "OC", "VU": "OC", "WF": "OC"
}


def get_root_domain(url: str) -> str:
    parsed_url = urllib.parse.urlparse(url)
    netloc = parsed_url.netloc
    extract_result = tldextract.extract(netloc)
    root_domain = f"{extract_result.domain}.{extract_result.suffix}"
    return root_domain


def get_continent_from_url(url: str) -> str:
    try:
        root_domain = get_root_domain(url)
        hostname = socket.gethostbyname(root_domain)
        response = DbIpCity.get(hostname, api_key='free')
        country_code = response.country
        continent_code = country_to_continent.get(country_code, "Unknown")
        return continent_mapping.get(continent_code, "Unknown")
    except Exception as e:
        logger.error(f"Error fetching continent for {url}: {e}")
        return "Unknown"


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

    continent = get_continent_from_url(rss_feed_url)

    return {
        "title": title,
        "description": description,
        "pub_date": formatted_pub_date,
        "url": rss_feed_url,
        "image_link": image_link or "",
        "classification": classification,
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
