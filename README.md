# Capstone 3 — Researcher-and-Summarizer Agent

An agent takes a research question, searches the web, reads pages, and writes a short summary with citations. The loop is plain Python first. `m3_langgraph_agent.py` is that same loop as a LangGraph.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `OPENROUTER_API_KEY`. The same key used in Capstone 1 and Capstone 2 works here.

Long-term memory uses Upstash Redis when both of these are set:

```text
UPSTASH_REDIS_REST_URL
UPSTASH_REDIS_REST_TOKEN
```

Without them, notes are stored in `memory_store/research.json`. The next run can still recall an earlier topic.

## Run

```powershell
.\.venv\Scripts\python.exe test_react.py
.\.venv\Scripts\python.exe m3_react_agent.py "Compare Raptor RAG vs HyDE"
.\.venv\Scripts\python.exe run_sample.py
.\.venv\Scripts\python.exe m3_langgraph_agent.py "Compare Raptor RAG vs HyDE"
```

`run_sample.py` writes `sample_trace.md`. Add `--graph` to run the LangGraph agent instead.

## What each file does

- `m3_react_agent.py` — Thought, Action, Observation loop. Stops on a final answer, or when iterations, tokens, or wall-clock time run out.
- `tools.py` — `web_search`, `fetch_page` (BeautifulSoup extraction), and the citation list. Search uses DuckDuckGo. arXiv is used only when the web search returns nothing.
- `memory.py` — short-term scratchpad that drops the oldest steps, and long-term notes in Upstash or a local file.
- `m3_langgraph_agent.py` — recall, reason, act, and synthesize nodes over the same tools.

A run prints the stop reason (`goal_met`, `max_iterations`, `max_tokens`, or `max_seconds`), the trace, the summary, and the source list.
