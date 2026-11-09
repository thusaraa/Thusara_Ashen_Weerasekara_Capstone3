"""Short-term scratchpad and long-term research notes.

The scratchpad is the working memory for one task. When it grows past the
character budget, the oldest steps are dropped and a count is kept so the
model still knows those steps happened.

Long-term notes survive across runs. Upstash Redis is used when
UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN are set. Otherwise the
same records are written to memory_store/research.json.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone

import requests

from config import MEMORY_DIR, SHORT_TERM_CHARS

INDEX_KEY = "capstone3:index"
STOPWORDS = {
    "about", "after", "also", "and", "are", "because", "compare", "does",
    "from", "have", "how", "into", "that", "the", "their", "them", "then",
    "they", "this", "what", "when", "with", "versus", "using",
}


class ConversationBuffer:
    def __init__(self, char_budget: int = SHORT_TERM_CHARS):
        self.char_budget = char_budget
        self.blocks: list[str] = []
        self.dropped = 0

    def add(self, block: str) -> str:
        self.blocks.append(block.strip())
        while self._size() > self.char_budget and len(self.blocks) > 1:
            self.blocks.pop(0)
            self.dropped += 1
        return self.render()

    def render(self) -> str:
        if not self.blocks:
            return "No steps yet."
        body = "\n\n---\n\n".join(self.blocks)
        if not self.dropped:
            return body
        return (
            f"[{self.dropped} earlier steps removed to stay inside the context window]\n\n"
            + body
        )

    def _size(self) -> int:
        return sum(len(block) for block in self.blocks)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text.lower())).strip()


def _tokens(text: str) -> set[str]:
    return {word for word in _normalize(text).split() if len(word) > 3 and word not in STOPWORDS}


def _score(question: str, note: dict) -> int:
    return len(_tokens(question) & _tokens(note.get("question", "") + " " + note.get("summary", "")))


class LongTermMemory:
    def __init__(self):
        self.url = os.getenv("UPSTASH_REDIS_REST_URL", "").rstrip("/")
        self.token = os.getenv("UPSTASH_REDIS_REST_TOKEN", "")
        self.path = MEMORY_DIR / "research.json"
        self.backend = "upstash" if self.url and self.token else "local"

    def recall(self, question: str) -> str:
        notes = self._load()
        ranked = sorted(notes, key=lambda note: _score(question, note), reverse=True)
        useful = [note for note in ranked if _score(question, note) > 0][:2]
        if not useful:
            return "No earlier research is stored."
        chunks = []
        for note in useful:
            summary = " ".join(note.get("summary", "").split())[:500]
            chunks.append(f"Earlier question: {note.get('question')}\nNote: {summary}")
        return "\n\n".join(chunks)

    def save(self, question: str, summary: str, sources: list[dict]) -> None:
        notes = [
            note for note in self._load()
            if _normalize(note.get("question", "")) != _normalize(question)
        ]
        notes.append(
            {
                "question": question,
                "summary": summary[:2000],
                "sources": sources,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._store(notes[-20:])

    def _load(self) -> list[dict]:
        if self.backend == "upstash":
            try:
                raw = self._command("GET", INDEX_KEY)
                return json.loads(raw) if raw else []
            except (requests.RequestException, json.JSONDecodeError, RuntimeError):
                self.backend = "local"
        return self._read_local()

    def _store(self, notes: list[dict]) -> None:
        if self.backend == "upstash":
            try:
                self._command("SET", INDEX_KEY, json.dumps(notes))
                return
            except (requests.RequestException, RuntimeError):
                self.backend = "local"
        self._write_local(notes)

    def _command(self, *parts: str):
        response = requests.post(
            self.url,
            headers={"Authorization": f"Bearer {self.token}"},
            json=list(parts),
            timeout=15,
        )
        response.raise_for_status()
        body = response.json()
        if isinstance(body, dict) and body.get("error"):
            raise RuntimeError(body["error"])
        return body.get("result") if isinstance(body, dict) else body

    def _read_local(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []

    def _write_local(self, notes: list[dict]) -> None:
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(notes, indent=2), encoding="utf-8")
