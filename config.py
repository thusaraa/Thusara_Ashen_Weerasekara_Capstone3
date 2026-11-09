import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
MEMORY_DIR = ROOT / "memory_store"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Hard stops for one research task. The loop also stops when the model finishes.
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "8"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "12000"))
MAX_SECONDS = int(os.getenv("MAX_SECONDS", "180"))

# Oldest thoughts are dropped once the scratchpad passes this size.
SHORT_TERM_CHARS = int(os.getenv("SHORT_TERM_CHARS", "6000"))

PAGE_CHARS = 4000
SEARCH_LIMIT = 5


def chat_client_kwargs() -> dict:
    """Arguments shared by the scratch client and the LangGraph model."""
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        return {
            "model": os.getenv("CHAT_MODEL", "gpt-4o-mini"),
            "api_key": openai_key,
            "temperature": 0,
        }

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        return {
            "model": os.getenv("CHAT_MODEL", "openai/gpt-4o-mini"),
            "api_key": openrouter_key,
            "base_url": OPENROUTER_BASE_URL,
            "temperature": 0,
        }

    raise RuntimeError("Set OPENAI_API_KEY or OPENROUTER_API_KEY in .env")
