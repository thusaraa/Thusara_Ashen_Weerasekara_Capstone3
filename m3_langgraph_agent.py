"""The same researcher, expressed as a LangGraph.

recall -> reason -> act -> reason ... -> synthesize

reason chooses the next ReAct step. act runs one tool. synthesize writes
the cited summary when the model finishes or a budget is spent. Tool code
and the step parser stay in tools.py and m3_react_agent.py.
"""

from __future__ import annotations

import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from config import MAX_ITERATIONS, MAX_SECONDS, MAX_TOKENS
from m3_react_agent import (
    ResearchReport,
    RunResult,
    Source,
    _step_block,
    default_complete,
    finish_block,
    parse_step,
    render_prompt,
    summary_from_text,
    synthesis_prompt,
)
from memory import ConversationBuffer, LongTermMemory
from tools import ResearchTools


class ResearchState(TypedDict):
    question: str
    memory_context: str
    scratchpad: str
    iteration: int
    tokens: int
    started: float
    thought: str
    action: str
    action_input: dict
    parse_error: str
    summary: str
    stop_reason: str
    trace: list


def build_graph(
    complete=None,
    tools: ResearchTools | None = None,
    memory: LongTermMemory | None = None,
    max_iterations: int | None = None,
    max_tokens: int | None = None,
    max_seconds: int | None = None,
):
    complete = complete or default_complete
    tools = tools or ResearchTools()
    memory = memory or LongTermMemory()
    max_iterations = MAX_ITERATIONS if max_iterations is None else max_iterations
    max_tokens = MAX_TOKENS if max_tokens is None else max_tokens
    max_seconds = MAX_SECONDS if max_seconds is None else max_seconds
    buffer = ConversationBuffer()

    def recall(state: ResearchState) -> dict:
        return {"memory_context": memory.recall(state["question"])}

    def reason(state: ResearchState) -> dict:
        elapsed = time.perf_counter() - state["started"]
        if elapsed > max_seconds:
            return {"stop_reason": "max_seconds", "action": "finish", "parse_error": ""}
        if state["tokens"] >= max_tokens:
            return {"stop_reason": "max_tokens", "action": "finish", "parse_error": ""}
        if state["iteration"] >= max_iterations:
            return {"stop_reason": "max_iterations", "action": "finish", "parse_error": ""}

        turn = state["iteration"] + 1
        prompt = render_prompt(
            state["question"],
            state["memory_context"],
            tools.tracker.render(),
            state["scratchpad"],
            turn,
            max_iterations,
        )
        text, input_tokens, output_tokens = complete(prompt)
        parsed = parse_step(text)
        update = {
            "tokens": state["tokens"] + input_tokens + output_tokens,
            "thought": parsed.thought,
            "action": parsed.action or "invalid",
            "action_input": parsed.action_input,
            "parse_error": parsed.parse_error,
            "iteration": turn,
        }
        # The last turn may only finish. A tool call here would exceed the budget.
        if turn == max_iterations and parsed.action != "finish":
            update["action"] = "finish"
            update["parse_error"] = ""
            update["summary"] = ""
            update["stop_reason"] = "max_iterations"
            update["trace"] = state["trace"] + [
                {
                    "iteration": turn,
                    "thought": parsed.thought,
                    "action": parsed.action or "invalid",
                    "action_input": parsed.action_input,
                    "observation": "Stopped: iteration budget spent before another tool call.",
                    "raw": text,
                }
            ]
            return update
        if parsed.action == "finish" and not parsed.parse_error:
            blocked = finish_block(
                parsed.action_input.get("summary", ""),
                tools.tracker.sources,
                last_turn=turn == max_iterations,
            )
            if blocked:
                update["parse_error"] = blocked
                return update
            update["summary"] = parsed.action_input.get("summary", "")
            update["stop_reason"] = "goal_met"
            update["trace"] = state["trace"] + [
                {
                    "iteration": turn,
                    "thought": parsed.thought,
                    "action": "finish",
                    "action_input": parsed.action_input,
                    "observation": "",
                    "raw": text,
                }
            ]
        return update

    def act(state: ResearchState) -> dict:
        if state["parse_error"]:
            observation = state["parse_error"]
            action = state["action"] if state["action"] in {"web_search", "fetch_page", "add_citation", "finish"} else "invalid"
        else:
            action = state["action"]
            observation = tools.dispatch(action, state["action_input"])
        scratchpad = buffer.add(
            _step_block(state["thought"], action, state["action_input"], observation)
        )
        step = {
            "iteration": state["iteration"],
            "thought": state["thought"],
            "action": action,
            "action_input": state["action_input"],
            "observation": observation,
            "raw": "",
        }
        return {"scratchpad": scratchpad, "trace": state["trace"] + [step]}

    def synthesize(state: ResearchState) -> dict:
        summary = state["summary"].strip()
        tokens = state["tokens"]
        trace = list(state["trace"])
        if not summary:
            text, input_tokens, output_tokens = complete(
                synthesis_prompt(state["question"], tools.tracker.render(), state["scratchpad"])
            )
            tokens += input_tokens + output_tokens
            summary = summary_from_text(text)
            trace.append(
                {
                    "iteration": state["iteration"] + 1,
                    "thought": "Budget reached. Writing the summary from the notes already collected.",
                    "action": "finish",
                    "action_input": {},
                    "observation": "",
                    "raw": text,
                }
            )
        memory.save(state["question"], summary, tools.tracker.sources)
        return {"summary": summary, "tokens": tokens, "trace": trace}

    def route(state: ResearchState) -> str:
        if state["action"] == "finish" and not state["parse_error"]:
            return "synthesize"
        if state["stop_reason"] in {"max_seconds", "max_tokens", "max_iterations"}:
            return "synthesize"
        return "act"

    graph = StateGraph(ResearchState)
    graph.add_node("recall", recall)
    graph.add_node("reason", reason)
    graph.add_node("act", act)
    graph.add_node("synthesize", synthesize)
    graph.add_edge(START, "recall")
    graph.add_edge("recall", "reason")
    graph.add_conditional_edges("reason", route, {"act": "act", "synthesize": "synthesize"})
    graph.add_edge("act", "reason")
    graph.add_edge("synthesize", END)
    return graph.compile(), tools, memory


def research_graph(question: str, **kwargs) -> RunResult:
    app, tools, memory = build_graph(**kwargs)
    started = time.perf_counter()
    final = app.invoke(
        {
            "question": question,
            "memory_context": "",
            "scratchpad": "No steps yet.",
            "iteration": 0,
            "tokens": 0,
            "started": started,
            "thought": "",
            "action": "",
            "action_input": {},
            "parse_error": "",
            "summary": "",
            "stop_reason": "",
            "trace": [],
        },
        config={"recursion_limit": 40},
    )
    report = ResearchReport(
        question=question,
        summary=final["summary"],
        sources=[Source(**source) for source in tools.tracker.sources],
        stop_reason=final["stop_reason"] or "max_iterations",
        iterations=max((step["iteration"] for step in final["trace"]), default=0),
        total_tokens=final["tokens"],
        elapsed_seconds=round(time.perf_counter() - started, 1),
        memory_backend=memory.backend,
    )
    return RunResult(report=report, trace=final["trace"])


def main():
    import argparse

    from m3_react_agent import render_trace

    parser = argparse.ArgumentParser(description="Run the LangGraph research agent.")
    parser.add_argument("question", nargs="?", default="Compare Raptor RAG vs HyDE")
    parser.add_argument("--trace", default="", help="Optional path for the markdown trace.")
    args = parser.parse_args()
    result = research_graph(args.question)
    rendered = render_trace(result, engine="LangGraph")
    print(rendered)
    if args.trace:
        from pathlib import Path

        Path(args.trace).write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
