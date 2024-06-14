from fastapi import FastAPI, BackgroundTasks
import httpx
import xml.etree.ElementTree as ET
from datetime import datetime
from xml.etree.ElementTree import ParseError
import asyncio
import random
from typing import List, Optional, Dict, Any

app = FastAPI()

classifications = [
    "Architectural", "Bricks & Blocks", "Cladding", "Construction Software", "Education",
    "Environmental", "General Construction", "Glass and Glazing", "Green issues", "Groundworks",
    "House Building", "Interiors", "Plant & Machinery", "Plumbing and heating", "Public sector",
    "Roads & Highways", "Roofing", "Solar", "Surveying", "Sustainability", "Tools and accessories",
    "Waterproofing", "Building Regulations", "Drainage and flood control", "Concrete", "Minerals",
    "Engineering", "General Construction", "Modular Building", "BIM Technology"
]

combined_feed: List[Dict[str, Any]] = []
denied_urls: List[str] = []

feed_lock = asyncio.Lock()


async def fetch_feed_data(rss_feed_url: str, headers: Dict[str, str]) -> Optional[bytes]:
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(rss_feed_url, headers=headers)
            if response.status_code == 200:
                return response.content
            else:
                print(f"Failed to fetch the RSS feed for {rss_feed_url} with status code {response.status_code}")
                denied_urls.append(rss_feed_url)
        except httpx.RequestError as e:
            print(f"Error fetching RSS feed: {e}")
            denied_urls.append(rss_feed_url)
    return None


async def parse_feed_entry(item: ET.Element, rss_feed_url: str) -> Dict[str, Any]:
    title = item.findtext("title", default="")
    link = item.findtext("link", default="")
    description = item.findtext("description", default="")
    pub_date_str = item.findtext("pubDate", default="")

    pub_date = None
    formats_to_try = [
        "%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S", "%a, %d %b %Y %H:%M %z",
        "%a, %d %b %Y %H:%M", "%a, %d %b %Y %H:%M:%S GMT"
    ]
    for date_format in formats_to_try:
        try:
            pub_date = datetime.strptime(pub_date_str, date_format)
            break
        except ValueError:
            continue

    formatted_pub_date = pub_date.strftime("%Y-%m-%d %H:%M:%S") if pub_date else None

    classification = random.choice(classifications)

    return {
        "title": title,
        "link": link,
        "description": description,
        "pub_date": formatted_pub_date,
        "url": rss_feed_url,
        "image_link": "",
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
            try:
                root = ET.fromstring(feed_data)
                items = root.findall(".//item")
                for item in items:
                    entry = await parse_feed_entry(item, rss_feed_url)
                    updated_feed.append(entry)
            except ParseError as parse_error:
                print(f"Failed to parse XML for {rss_feed_url}: {parse_error}")
                updated_denied_urls.append(rss_feed_url)

    async with feed_lock:
        combined_feed = updated_feed
        denied_urls = updated_denied_urls


@app.on_event("startup")
async def startup_event() -> None:
    await update_combined_feed()
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
        await asyncio.sleep(600)