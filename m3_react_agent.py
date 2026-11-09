"""Researcher-and-Summarizer, built as a plain ReAct loop.

Each turn asks the model for one Thought and one Action. The code runs that
action, appends the Observation, and asks again. Nothing here uses
AgentExecutor, create_react_agent, or LangGraph. m3_langgraph_agent.py is
the same loop expressed as a graph.

The loop stops when the model finishes, or when iterations, tokens, or
wall-clock time run out. A budget stop still produces a cited summary from
whatever was collected.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from config import MAX_ITERATIONS, MAX_SECONDS, MAX_TOKENS, chat_client_kwargs
from memory import ConversationBuffer, LongTermMemory
from tools import ResearchTools

SYSTEM = """You research a question with a ReAct loop. Each reply is one tool call, or the final answer.

Tool call, in exactly this shape:
Thought: one sentence
Action: web_search
Action Input: {"query": "Raptor RAG"}

Action is only one of: web_search, fetch_page, add_citation, finish.
fetch_page uses {"url": "https://..."}.
add_citation uses {"url": "https://...", "title": "...", "note": "..."}.

Final answer, only after the Sources list has at least two pages:
Thought: one sentence
Final Answer: the answer with inline citations such as [1] and [2]

Rules:
- Do not write a sentence in the Action line. The action is a single tool name.
- Keep each technique's name as written in the question. Add a topic word from the question, such as RAG. Do not invent an expansion.
- If the titles are about a different subject, ignore them and search again with the original name.
- After a search returns relevant URLs, call fetch_page before searching again.
- For a comparison, fetch one page about each side.
- If a search says WEAK RESULTS or a fetch fails, try a different query or URL.
- Do not invent URLs, numbers, or citations.
- The summary is a direct answer. No preamble.
"""

FORMAT_ERROR = (
    "Could not read an action. Reply in exactly this shape:\n"
    "Thought: one sentence\n"
    "Action: web_search\n"
    'Action Input: {"query": "..."}\n'
    "Action must be web_search, fetch_page, add_citation, or finish."
)
ACTIONS = {"web_search", "fetch_page", "add_citation", "finish"}


class Source(BaseModel):
    id: int
    title: str
    url: str
    note: str = ""


class ResearchReport(BaseModel):
    question: str
    summary: str
    sources: list[Source] = Field(default_factory=list)
    stop_reason: str
    iterations: int
    total_tokens: int
    elapsed_seconds: float
    memory_backend: str


@dataclass
class ParsedStep:
    thought: str
    action: str
    action_input: dict
    parse_error: str = ""


@dataclass
class RunResult:
    report: ResearchReport
    trace: list[dict] = field(default_factory=list)


def strip_fence(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def _thought(text: str) -> str:
    match = re.search(r"Thought:\s*(.*?)(?=\n(?:Action|Final Answer):|\Z)", text, re.S | re.I)
    return match.group(1).strip() if match else ""


def _load_input(raw: str) -> dict:
    raw = strip_fence(raw).strip()
    if not raw:
        return {}
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                loaded = json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                loaded = raw
        else:
            loaded = raw
    if isinstance(loaded, dict):
        return loaded
    return {"query": str(loaded), "url": str(loaded), "summary": str(loaded)}


def parse_step(text: str) -> ParsedStep:
    cleaned = strip_fence(text)
    final = re.search(r"Final Answer:\s*(.*)", cleaned, re.S | re.I)
    if final:
        return ParsedStep(
            thought=_thought(cleaned),
            action="finish",
            action_input={"summary": final.group(1).strip()},
        )
    action = re.search(r"Action:\s*([A-Za-z_]+)\s*(\{.*\}|\(.*\))?", cleaned, re.S)
    if not action:
        return ParsedStep(
            thought=_thought(cleaned) or cleaned[:400],
            action="",
            action_input={},
            parse_error=FORMAT_ERROR,
        )
    name = action.group(1).strip().lower()
    if name not in ACTIONS:
        return ParsedStep(
            thought=_thought(cleaned),
            action=name,
            action_input={},
            parse_error=FORMAT_ERROR,
        )
    inline = (action.group(2) or "").strip()
    payload_match = re.search(r"Action Input:\s*(.*)", cleaned, re.S | re.I)
    if payload_match:
        payload = _load_input(payload_match.group(1))
    elif inline.startswith("(") and inline.endswith(")"):
        inner = inline[1:-1].strip().strip('"').strip("'")
        payload = {"query": inner, "url": inner, "summary": inner}
    else:
        payload = _load_input(inline)
    if name == "finish":
        summary = str(payload.get("summary") or payload.get("query") or "").strip()
        return ParsedStep(thought=_thought(cleaned), action="finish", action_input={"summary": summary})
    return ParsedStep(thought=_thought(cleaned), action=name, action_input=payload)


def finish_block(summary: str, sources: list, last_turn: bool) -> str:
    """Refuse a final answer that has not read pages or cites missing sources."""
    if len(sources) == 0:
        return (
            "No pages have been read. Call fetch_page on a URL from the search "
            "results before answering."
        )
    valid = {source["id"] for source in sources}
    cited = {int(number) for number in re.findall(r"\[(\d+)\]", summary or "")}
    if cited - valid:
        return "Those citation numbers are not in the Sources list. Cite only ids from that list."
    if last_turn:
        return ""
    if len(sources) < 2:
        return (
            f"Only {len(sources)} source recorded. "
            "Fetch one more page before the final answer. "
            "For a comparison, read one page for each side."
        )
    if not cited:
        return "The final answer needs inline citations such as [1] that match the Sources list."
    return ""


def render_prompt(question, memory_context, sources, scratchpad, iteration, max_iterations) -> str:
    remaining = max_iterations - iteration + 1
    budget = f"Iterations left after this reply: {remaining - 1}."
    if remaining <= 1:
        budget = "This is the last turn. Reply with Final Answer only. Do not call a tool."
    return (
        f"Question: {question}\n\n"
        f"Long-term memory:\n{memory_context}\n\n"
        f"Sources:\n{sources}\n\n"
        f"Working memory:\n{scratchpad}\n\n"
        f"{budget}"
    )


def synthesis_prompt(question, sources, scratchpad) -> str:
    return (
        "Write the final summary from the research below. "
        "Cite only ids that appear in Sources, such as [1]. "
        "If Sources says None yet, do not use citation numbers. "
        "Reply in this form and nothing else:\n"
        "Final Answer: ...\n\n"
        f"Question: {question}\n\n"
        f"Sources:\n{sources}\n\n"
        f"Working memory:\n{scratchpad}"
    )


def summary_from_text(text: str) -> str:
    parsed = parse_step(text)
    if parsed.action == "finish" and parsed.action_input.get("summary"):
        return parsed.action_input["summary"]
    return strip_fence(text).strip()


def default_complete(prompt: str) -> tuple[str, int, int]:
    from openai import OpenAI

    kwargs = chat_client_kwargs()
    client = OpenAI(api_key=kwargs["api_key"], base_url=kwargs.get("base_url"))
    response = client.chat.completions.create(
        model=kwargs["model"],
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    usage = response.usage
    text = response.choices[0].message.content or ""
    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", 0) or 0
    return text, input_tokens, output_tokens


def _step_block(thought: str, action: str, action_input: dict, observation: str) -> str:
    return (
        f"Thought: {thought}\n"
        f"Action: {action}\n"
        f"Action Input: {json.dumps(action_input, ensure_ascii=True)}\n"
        f"Observation: {observation[:2200]}"
    )


def research(
    question: str,
    complete=None,
    tools: ResearchTools | None = None,
    memory: LongTermMemory | None = None,
    max_iterations: int | None = None,
    max_tokens: int | None = None,
    max_seconds: int | None = None,
) -> RunResult:
    complete = complete or default_complete
    tools = tools or ResearchTools()
    memory = memory or LongTermMemory()
    max_iterations = MAX_ITERATIONS if max_iterations is None else max_iterations
    max_tokens = MAX_TOKENS if max_tokens is None else max_tokens
    max_seconds = MAX_SECONDS if max_seconds is None else max_seconds

    buffer = ConversationBuffer()
    recalled = memory.recall(question)
    trace: list[dict] = []
    tokens = 0
    started = time.perf_counter()
    stop = "max_iterations"
    summary = ""

    for iteration in range(1, max_iterations + 1):
        elapsed = time.perf_counter() - started
        if elapsed > max_seconds:
            stop = "max_seconds"
            break
        if tokens >= max_tokens:
            stop = "max_tokens"
            break

        prompt = render_prompt(
            question, recalled, tools.tracker.render(), buffer.render(), iteration, max_iterations
        )
        text, input_tokens, output_tokens = complete(prompt)
        tokens += input_tokens + output_tokens
        parsed = parse_step(text)
        last_turn = iteration == max_iterations

        if parsed.parse_error:
            observation = parsed.parse_error
            action = "invalid"
        elif parsed.action == "finish":
            summary_text = parsed.action_input.get("summary", "").strip()
            blocked = finish_block(summary_text, tools.tracker.sources, last_turn)
            if not blocked:
                summary = summary_text
                trace.append(
                    {
                        "iteration": iteration,
                        "thought": parsed.thought,
                        "action": "finish",
                        "action_input": parsed.action_input,
                        "observation": "",
                        "raw": text,
                    }
                )
                stop = "goal_met"
                break
            observation = blocked
            action = "finish"
        elif last_turn:
            trace.append(
                {
                    "iteration": iteration,
                    "thought": parsed.thought,
                    "action": parsed.action,
                    "action_input": parsed.action_input,
                    "observation": "Stopped: iteration budget spent before another tool call.",
                    "raw": text,
                }
            )
            stop = "max_iterations"
            break
        else:
            observation = tools.dispatch(parsed.action, parsed.action_input)
            action = parsed.action

        buffer.add(_step_block(parsed.thought, action, parsed.action_input, observation))
        trace.append(
            {
                "iteration": iteration,
                "thought": parsed.thought,
                "action": action,
                "action_input": parsed.action_input,
                "observation": observation,
                "raw": text,
            }
        )

    if not summary:
        synth_text, input_tokens, output_tokens = complete(
            synthesis_prompt(question, tools.tracker.render(), buffer.render())
        )
        tokens += input_tokens + output_tokens
        summary = summary_from_text(synth_text)
        trace.append(
            {
                "iteration": len(trace) + 1,
                "thought": "Budget reached. Writing the summary from the notes already collected.",
                "action": "finish",
                "action_input": {},
                "observation": "",
                "raw": synth_text,
            }
        )

    memory.save(question, summary, tools.tracker.sources)
    elapsed = round(time.perf_counter() - started, 1)
    report = ResearchReport(
        question=question,
        summary=summary,
        sources=[Source(**source) for source in tools.tracker.sources],
        stop_reason=stop,
        iterations=max((step["iteration"] for step in trace), default=0),
        total_tokens=tokens,
        elapsed_seconds=elapsed,
        memory_backend=memory.backend,
    )
    return RunResult(report=report, trace=trace)


def render_trace(result: RunResult, engine: str = "from-scratch ReAct") -> str:
    report = result.report
    lines = [
        f"# {report.question}",
        "",
        f"Engine: {engine}",
        f"Stop: {report.stop_reason}",
        f"Iterations: {report.iterations}",
        f"Tokens: {report.total_tokens}",
        f"Elapsed seconds: {report.elapsed_seconds}",
        f"Memory: {report.memory_backend}",
        "",
        "## Trace",
        "",
    ]
    for step in result.trace:
        lines.append(f"### Step {step['iteration']}")
        lines.append(f"Thought: {step['thought']}")
        lines.append(f"Action: {step['action']}")
        if step["action_input"]:
            lines.append(f"Action Input: {json.dumps(step['action_input'], ensure_ascii=True)}")
        if step["observation"]:
            lines.append("")
            lines.append("Observation:")
            lines.append(step["observation"][:2000])
        lines.append("")
    lines.extend(["## Summary", "", report.summary, "", "## Sources", ""])
    if not report.sources:
        lines.append("No sources were recorded.")
    for source in report.sources:
        lines.append(f"{source.id}. {source.title} — {source.url}")
    lines.append("")
    return "\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run the from-scratch research agent.")
    parser.add_argument("question", nargs="?", default="Compare Raptor RAG vs HyDE")
    parser.add_argument("--trace", default="", help="Optional path for the markdown trace.")
    args = parser.parse_args()
    result = research(args.question)
    rendered = render_trace(result)
    print(rendered)
    if args.trace:
        from pathlib import Path

        Path(args.trace).write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
