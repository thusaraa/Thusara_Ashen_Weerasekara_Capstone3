"""Web search, page fetch, text extraction, and the citation list.

Search tries DuckDuckGo first, then the public arXiv API when the web
results are thin. That second source matters for paper-to-paper questions
such as Raptor RAG versus HyDE.
"""

from __future__ import annotations

import ipaddress
import re
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from config import PAGE_CHARS, SEARCH_LIMIT

HEADERS = {"User-Agent": "capstone3-research-agent/1.0"}
ATOM = {"a": "http://www.w3.org/2005/Atom"}
ACTIONS = ("web_search", "fetch_page", "add_citation")


class CitationTracker:
    """Sources the agent has actually read or chosen to cite."""

    def __init__(self):
        self.sources: list[dict] = []

    def add(self, url: str, title: str, note: str = "") -> int:
        url = (url or "").strip()
        for source in self.sources:
            if source["url"] == url:
                if note and note not in source["note"]:
                    source["note"] = f"{source['note']} {note}".strip()
                if title and source["title"] == url:
                    source["title"] = title
                return source["id"]
        item = {
            "id": len(self.sources) + 1,
            "title": (title or url or "Untitled").strip(),
            "url": url,
            "note": (note or "").strip(),
        }
        self.sources.append(item)
        return item["id"]

    def render(self) -> str:
        if not self.sources:
            return "None yet."
        lines = []
        for source in self.sources:
            line = f"[{source['id']}] {source['title']} — {source['url']}"
            if source["note"]:
                line += f"\nExcerpt: {source['note'][:500]}"
            lines.append(line)
        return "\n".join(lines)


def check_public_url(url: str) -> str | None:
    """Return an error string when the URL is not a public web page."""
    parsed = urlparse((url or "").strip())
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in {"http", "https"} or not host:
        return "Only http and https URLs can be fetched."
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
        return "Local addresses are blocked."
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address and (address.is_private or address.is_loopback or address.is_link_local):
        return "Private addresses are blocked."
    return None


def extract_content(html: str) -> tuple[str, str]:
    """Pull the title and article text out of an HTML page."""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()
    noisy = []
    for tag in soup.find_all(True):
        classes = tag.get("class") or []
        if not isinstance(classes, list):
            classes = [str(classes)]
        ident = f"{' '.join(classes)} {tag.get('id') or ''}"
        if re.search(r"\b(nav|menu|sidebar|footer|header|toc|breadcrumb|cookie|share)\b", ident, re.I):
            noisy.append(tag)
    for tag in noisy:
        if tag.parent is not None:
            tag.decompose()
    root = soup.find("article") or soup.find("main") or soup.body or soup
    text = root.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return title, text[:PAGE_CHARS]


def search_duckduckgo(query: str, limit: int = SEARCH_LIMIT) -> list[dict]:
    try:
        from ddgs import DDGS
    except ImportError:
        return []
    try:
        rows = DDGS().text(query, max_results=limit)
    except Exception:
        return []
    hits = []
    for row in rows:
        url = row.get("href") or row.get("url") or ""
        if not url:
            continue
        hits.append(
            {
                "title": row.get("title") or url,
                "url": url,
                "snippet": row.get("body") or "",
            }
        )
    return hits


def search_arxiv(query: str, limit: int = SEARCH_LIMIT) -> list[dict]:
    response = requests.get(
        "https://export.arxiv.org/api/query",
        params={"search_query": f"all:{query}", "start": 0, "max_results": limit},
        headers=HEADERS,
        timeout=20,
    )
    response.raise_for_status()
    root = ET.fromstring(response.text)
    hits = []
    for entry in root.findall("a:entry", ATOM):
        title = " ".join((entry.findtext("a:title", default="", namespaces=ATOM) or "").split())
        summary = " ".join((entry.findtext("a:summary", default="", namespaces=ATOM) or "").split())
        url = ""
        for link in entry.findall("a:link", ATOM):
            if link.attrib.get("rel") == "alternate" and link.attrib.get("href"):
                url = link.attrib["href"]
                break
        if not url:
            url = entry.findtext("a:id", default="", namespaces=ATOM) or ""
        if title and url:
            hits.append({"title": title, "url": url, "snippet": summary[:400]})
    return hits


def prefer_arxiv_html(url: str) -> str:
    """arXiv abstract pages are thin. The HTML copy has the paper text."""
    match = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", url)
    if not match:
        return url
    return f"https://arxiv.org/html/{match.group(1)}"


def _dedupe(hits: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for hit in hits:
        if not hit.get("url") or hit["url"] in seen:
            continue
        seen.add(hit["url"])
        unique.append(hit)
    return unique


def _format_hits(hits: list[dict], weak: bool) -> str:
    lines = []
    for index, hit in enumerate(hits[:6], start=1):
        snippet = " ".join(hit["snippet"].split())[:300]
        lines.append(f"{index}. {hit['title']}\nURL: {hit['url']}\nSnippet: {snippet}")
    prefix = ""
    if weak:
        prefix = "Few matches. Fetch the best URL, or search the full technique name if these pages miss the question.\n\n"
    return prefix + "\n\n".join(lines)


class ResearchTools:
    def __init__(self, tracker: CitationTracker | None = None):
        self.tracker = tracker or CitationTracker()
        self.searches_since_fetch = 0

    def web_search(self, query: str) -> str:
        query = (query or "").strip()
        if not query:
            return "web_search needs a query string."
        hits = _dedupe(search_duckduckgo(query))
        if hits:
            return _format_hits(hits, weak=len(hits) < 3)
        try:
            papers = _dedupe(search_arxiv(query))
        except requests.RequestException as exc:
            return f"WEAK RESULTS: search failed ({exc}). Try a different query."
        if not papers:
            return (
                "WEAK RESULTS: no pages found. Try a more specific query, "
                "using the full name of any acronym."
            )
        warning = (
            "WEB SEARCH EMPTY. The hits below are arXiv papers. "
            "A short query can match an author name instead of the method. "
            "If the titles look unrelated, search again with the name from the question plus RAG.\n\n"
        )
        return warning + _format_hits(papers, weak=True)

    def fetch_page(self, url: str) -> str:
        blocked = check_public_url(url)
        if blocked:
            return blocked
        original = url.strip()
        target = prefer_arxiv_html(original)
        if target.lower().split("?", 1)[0].endswith(".pdf"):
            return "This URL is a PDF. Fetch the HTML abstract or article page instead."
        result = self._read_page(target)
        if target != original and result.startswith(("WEAK PAGE", "Fetch failed")):
            return self._read_page(original)
        return result

    def _read_page(self, target: str) -> str:
        try:
            response = requests.get(target, headers=HEADERS, timeout=20, allow_redirects=True)
            response.raise_for_status()
            response.encoding = response.apparent_encoding or response.encoding
        except requests.RequestException as exc:
            return f"Fetch failed for {target}: {exc}. Choose another URL."
        content_type = response.headers.get("content-type", "").lower()
        if "pdf" in content_type:
            return "This URL returned a PDF. Fetch an HTML page instead."
        title, text = extract_content(response.text[:500_000])
        if len(text) < 400:
            return (
                f"WEAK PAGE: almost no article text extracted from {target}. "
                "Try another URL, such as an arXiv HTML page or a different article."
            )
        excerpt = " ".join(text.split())[:700]
        source_id = self.tracker.add(response.url, title or target, note=excerpt)
        return (
            f"Recorded as source [{source_id}]. Title: {title or 'Untitled'}\n"
            f"URL: {response.url}\n\n{text}"
        )

    def add_citation(self, url: str, title: str = "", note: str = "") -> str:
        blocked = check_public_url(url)
        if blocked:
            return blocked
        source_id = self.tracker.add(url, title or url, note)
        return f"Recorded as source [{source_id}]."

    def dispatch(self, action: str, payload: dict) -> str:
        try:
            if action == "web_search":
                if self.searches_since_fetch >= 2:
                    return (
                        "Two searches already returned URLs. "
                        "Call fetch_page on one of those URLs before searching again."
                    )
                result = self.web_search(str(payload.get("query") or payload.get("q") or ""))
                if not result.startswith(("WEB SEARCH EMPTY", "WEAK RESULTS", "web_search needs")):
                    self.searches_since_fetch += 1
                return (
                    result
                    + "\n\nNext: call fetch_page on the best URL. "
                    "Do not search again until that page has been read."
                )
            if action == "fetch_page":
                result = self.fetch_page(str(payload.get("url") or ""))
                if result.startswith("Recorded as source"):
                    self.searches_since_fetch = 0
                return result
            if action == "add_citation":
                return self.add_citation(
                    str(payload.get("url") or ""),
                    str(payload.get("title") or ""),
                    str(payload.get("note") or ""),
                )
            return (
                f"Unknown action '{action}'. "
                "Use web_search, fetch_page, add_citation, or finish."
            )
        except Exception as exc:
            return f"Tool error ({action}): {exc}. Try a different action."
