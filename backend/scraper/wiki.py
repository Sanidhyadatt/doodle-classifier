import asyncio
import re
from typing import Final
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup

WIKIPEDIA_SEARCH_URL: Final[str] = "https://en.wikipedia.org/wiki/Special:Search"


def _fetch_and_parse_first_paragraph(query: str) -> str:
    params = {"search": query, "go": "Go"}
    headers = {
        "User-Agent": "NeuroSketch/1.0 (educational project; +https://en.wikipedia.org/)"
    }

    response = requests.get(
        WIKIPEDIA_SEARCH_URL,
        params=params,
        headers=headers,
        timeout=10,
        allow_redirects=True,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    main_content = soup.select_one("div.mw-parser-output")
    if main_content is None:
        article_url = f"https://en.wikipedia.org/wiki/{quote_plus(query.replace(' ', '_'))}"
        response = requests.get(article_url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        main_content = soup.select_one("div.mw-parser-output")

    if main_content is None:
        return ""

    paragraph_text = ""
    for paragraph in main_content.select("p"):
        text = paragraph.get_text(" ", strip=True)
        if text:
            paragraph_text = text
            break

    if not paragraph_text:
        return ""

    cleaned = re.sub(r"\[[^\]]*\]", "", paragraph_text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _summarize_to_two_or_three_sentences(text: str) -> str:
    if not text:
        return "No definition found for the requested query."

    sentences = [
        s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s and s.strip()
    ]

    if not sentences:
        return "No definition found for the requested query."

    if len(sentences) == 1:
        return sentences[0]

    if len(sentences) == 2:
        return " ".join(sentences[:2])

    return " ".join(sentences[:3])


async def get_object_definition(query: str) -> str:
    """Fetch and summarize the first Wikipedia paragraph for a query in 2-3 sentences."""
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")

    raw_text = await asyncio.to_thread(_fetch_and_parse_first_paragraph, query.strip())
    return _summarize_to_two_or_three_sentences(raw_text)
