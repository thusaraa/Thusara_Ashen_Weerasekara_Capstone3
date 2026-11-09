"""Run the course sample and write sample_trace.md.

Default question: Compare Raptor RAG vs HyDE.
"""

import argparse
from pathlib import Path

from m3_react_agent import render_trace, research


def main():
    parser = argparse.ArgumentParser(description="Write a cited research trace.")
    parser.add_argument("--question", default="Compare Raptor RAG vs HyDE")
    parser.add_argument("--graph", action="store_true", help="Use the LangGraph agent.")
    parser.add_argument("--output", default="sample_trace.md")
    args = parser.parse_args()

    if args.graph:
        from m3_langgraph_agent import research_graph

        result = research_graph(args.question)
        rendered = render_trace(result, engine="LangGraph")
    else:
        result = research(args.question)
        rendered = render_trace(result)

    Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
