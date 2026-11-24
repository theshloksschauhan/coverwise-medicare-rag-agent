"""Download public Medicare.gov coverage pages into data/raw as Markdown.

Usage: python scripts/fetch_data.py
"""
import re
import sys
import time
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

TOPICS = [
    "acupuncture", "ambulance-services", "ambulatory-surgical-centers", "cardiac-rehabilitation-programs",
    "cataract-surgery", "chiropractic-services", "dental-services", "diabetes-screenings",
    "diagnostic-non-laboratory-tests", "dialysis-services-supplies", "emergency-department-services",
    "flu-vaccines", "hearing-aids", "home-health-services", "hospice-care", "mammograms",
    "mental-health-care-outpatient", "nursing-home-care", "oxygen-equipment-accessories",
    "physical-therapy-services", "prescription-drugs-outpatient", "skilled-nursing-facility-care",
    "sleep-studies", "telehealth", "wheelchairs-scooters", "x-rays",
]
BASE = "https://www.medicare.gov/coverage/"
OUT = Path(__file__).resolve().parents[1] / "data" / "raw"
TAB_LABELS = {"Description", "Coverage details", "Costs"}
BOILERPLATE_STARTS = ("Ask your doctor or healthcare provider", "Other related information", "Related topics")
HEADERS = {"User-Agent": "CoverWise-research-bot/0.1 (educational project)"}


def page_to_markdown(html: str, url: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.find("h1").get_text(" ", strip=True) if soup.find("h1") else url.rsplit("/", 1)[-1])
    main = soup.find("main") or soup.body
    for tag in main.find_all(["script", "style", "nav", "footer", "form", "button", "svg"]):
        tag.decompose()
    lines = []
    for el in main.find_all(["h2", "h3", "p", "li"]):
        if el.name == "p" and el.find_parent("li"):
            continue  # already captured by the parent list item
        text = re.sub(r"\s+", " ", el.get_text(" ", strip=True))
        if not text or len(text) < 3 or text in TAB_LABELS:
            continue
        if el.name in ("h2", "h3") and text.startswith(BOILERPLATE_STARTS):
            break  # the rest of the page is generic, identical on every topic
        lines.append(f"## {text}" if el.name in ("h2", "h3") else (f"- {text}" if el.name == "li" else text))
    body = "\n".join(dict.fromkeys(lines))  # drop exact duplicate lines, keep order
    return f"---\ntitle: {title}\nsource: {url}\n---\n# {title}\n{body}\n"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with httpx.Client(headers=HEADERS, timeout=30, follow_redirects=True) as client:
        for topic in TOPICS:
            url = BASE + topic
            resp = client.get(url)
            if resp.status_code != 200:
                print(f"skip {topic}: HTTP {resp.status_code}", file=sys.stderr)
                continue
            (OUT / f"{topic}.md").write_text(page_to_markdown(resp.text, url), encoding="utf-8")
            print(f"saved {topic}")
            time.sleep(1.0)  # be polite
    return 0


if __name__ == "__main__":
    sys.exit(main())
