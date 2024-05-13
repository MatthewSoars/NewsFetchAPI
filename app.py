from fastapi import FastAPI, BackgroundTasks
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from xml.etree.ElementTree import ParseError
import asyncio
import random

app = FastAPI()

classifications = [
    "Architectural",
    "Bricks & Blocks",
    "Cladding",
    "Construction Software",
    "Education",
    "Environmental",
    "General Construction",
    "Glazing",
    "Green issues",
    "Groundworks",
    "House Building",
    "Interiors",
    "Plant & Machinery",
    "Plumbing and heating",
    "Public sector",
    "Roads & Highways",
    "Roofing",
    "Solar",
    "Surveying",
    "Sustainability",
    "Tools and accessories",
    "Waterproofing",
    "Building Regulations",
    "Drainage and flood control",
    "Concrete",
    "Minerals"
]

combined_feed = []
denied_urls = []


async def fetch_feed_data(rss_feed_url, headers):
    try:
        response = requests.get(rss_feed_url, headers=headers)
        if response.status_code == 200:
            return response.content
        else:
            print("Failed to fetch the RSS feed for", rss_feed_url)
            denied_urls.append(rss_feed_url)
    except requests.exceptions.RequestException as e:
        print("Error fetching RSS feed:", e)
        denied_urls.append(rss_feed_url)
    return None


async def parse_feed_entry(item, rss_feed_url):
    title = item.findtext("title", default="")
    link = item.findtext("link", default="")
    description = item.findtext("description", default="")
    pub_date_str = item.findtext("pubDate", default="")

    pub_date = None
    formats_to_try = [
        "%a, %d %b %Y %H:%M:%S %z",  # Original format
        "%a, %d %b %Y %H:%M:%S",  # Without timezone
        "%a, %d %b %Y %H:%M %z",  # Without seconds
        "%a, %d %b %Y %H:%M",  # Without seconds and timezone
        "%a, %d %b %Y %H:%M:%S GMT"  # With 'GMT'
    ]
    for date_format in formats_to_try:
        try:
            pub_date = datetime.strptime(pub_date_str, date_format)
            break
        except ValueError:
            continue

    formatted_pub_date = pub_date.strftime("%Y-%m-%d %H:%M:%S") if pub_date else None

    # Assign random classification
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


async def update_combined_feed():
    global combined_feed, denied_urls

    # Read RSS feed URLs from the text file
    with open("Accepted.txt", "r") as file:
        rss_feed_urls = file.read().splitlines()

    # Set custom headers with User-Agent
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }

    updated_feed = []
    updated_denied_urls = []

    # Iterate over each RSS feed URL
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

    # Update global variables
    combined_feed = updated_feed
    denied_urls = updated_denied_urls


@app.on_event("startup")
async def startup_event():
    await update_combined_feed()

@app.get("/combined_feed")
async def get_combined_feed():
    return combined_feed


@app.post("/refresh_feed")
async def refresh_feed(background_tasks: BackgroundTasks):
    background_tasks.add_task(update_combined_feed)
    return {"message": "Feed refresh scheduled"}


# Background task to refresh the feed every 10 minutes
async def refresh_feed_background_task():
    while True:
        await update_combined_feed()
        await asyncio.sleep(600)  # Sleep for 10 minutes


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(refresh_feed_background_task())
