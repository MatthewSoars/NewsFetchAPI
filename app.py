from fastapi import FastAPI, BackgroundTasks
import httpx
import feedparser
from datetime import datetime
import asyncio
from typing import List, Dict, Any, Optional
import joblib
import os

app = FastAPI()

combined_feed: List[Dict[str, Any]] = []
denied_urls: List[str] = []

feed_lock = asyncio.Lock()

# Load the trained model
model_path = 'classification_model.pkl'
if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(f"Required module not found: {e}")
else:
    raise FileNotFoundError(f"Model file not found: {model_path}")


async def fetch_feed_data(rss_feed_url: str, headers: Dict[str, str]) -> Optional[bytes]:
    async with httpx.AsyncClient(verify=False) as client:
        try:
            response = await client.get(rss_feed_url, headers=headers, follow_redirects=True)
            if response.status_code == 200:
                return response.content
            else:
                print(f"Failed to fetch the RSS feed for {rss_feed_url} with status code {response.status_code}")
                denied_urls.append(rss_feed_url)
        except httpx.RequestError as e:
            print(f"Error fetching RSS feed: {e}")
            denied_urls.append(rss_feed_url)
    return None


def parse_feed_entry(entry: Dict[str, Any], rss_feed_url: str) -> Dict[str, Any]:
    title = entry.get("title", "")
    link = entry.get("link", "")
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

    # Predict classification using the model
    text = f"{title} {description}"
    classification = model.predict([text])[0]

    # Extract image link if available
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

    return {
        "title": title,
        "description": description,
        "pub_date": formatted_pub_date,
        "url": rss_feed_url,
        "image_link": image_link or "",
        "classification": classification
    }


async def update_combined_feed() -> None:
    global combined_feed, denied_urls

    try:
        with open("Accepted.txt", "r") as file:
            rss_feed_urls = file.read().splitlines()
    except FileNotFoundError as e:
        print(f"Accepted.txt not found: {e}")
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
            if parsed_feed.bozo:  # Check if there was a parsing error
                print(f"Failed to parse feed {rss_feed_url}: {parsed_feed.bozo_exception}")
                updated_denied_urls.append(rss_feed_url)
                continue
            for entry in parsed_feed.entries:
                feed_entry = parse_feed_entry(entry, rss_feed_url)
                updated_feed.append(feed_entry)

    async with feed_lock:
        combined_feed = updated_feed
        denied_urls = updated_denied_urls


@app.on_event("startup")
async def startup_event() -> None:
    # Initial feed update at startup
    await update_combined_feed()
    # Schedule the feed refresh task
    asyncio.create_task(refresh_feed_background_task())


@app.get("/combined_feed")
async def get_combined_feed() -> List[Dict[str, Any]]:
    async with feed_lock:
        return combined_feed


@app.post("/refresh_feed")
async def refresh_feed(background_tasks: BackgroundTasks) -> Dict[str, str]:
    background_tasks.add_task(update_combined_feed)
    return {"message": "Feed refresh scheduled"}


async def refresh_feed_background_task() -> None:
    while True:
        await update_combined_feed()
        await asyncio.sleep(600)  # 600 seconds = 10 minutes
