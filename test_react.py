"""Checks that do not call a model or the network."""

import unittest

from memory import ConversationBuffer, LongTermMemory, _score
from m3_react_agent import parse_step, research
from tools import CitationTracker, ResearchTools, extract_content, prefer_arxiv_html


class ParseTests(unittest.TestCase):
    def test_final_answer(self):
        parsed = parse_step("Thought: enough sources.\nFinal Answer: HyDE [1] differs from Raptor [2].")
        self.assertEqual(parsed.action, "finish")
        self.assertIn("[1]", parsed.action_input["summary"])

    def test_action_json(self):
        parsed = parse_step(
            'Thought: look up both methods.\nAction: web_search\nAction Input: {"query": "Raptor RAG"}'
        )
        self.assertEqual(parsed.action, "web_search")
        self.assertEqual(parsed.action_input["query"], "Raptor RAG")

    def test_fenced_json(self):
        parsed = parse_step(
            "Thought: open the paper.\nAction: fetch_page\n"
            "Action Input:\n```json\n{\"url\": \"https://arxiv.org/abs/2212.10496\"}\n```"
        )
        self.assertEqual(parsed.action_input["url"], "https://arxiv.org/abs/2212.10496")

    def test_missing_action(self):
        parsed = parse_step("I think I should search.")
        self.assertTrue(parsed.parse_error)

    def test_prose_action_is_rejected(self):
        parsed = parse_step("Thought: search.\nAction: I should search the web")
        self.assertTrue(parsed.parse_error)

    def test_inline_query(self):
        parsed = parse_step('Thought: search.\nAction: web_search("Raptor RAG")')
        self.assertEqual(parsed.action, "web_search")
        self.assertEqual(parsed.action_input["query"], "Raptor RAG")


class ToolTests(unittest.TestCase):
    def test_citation_numbers_stay_stable(self):
        tracker = CitationTracker()
        first = tracker.add("https://example.com/a", "A")
        second = tracker.add("https://example.com/a", "A", note="again")
        third = tracker.add("https://example.com/b", "B")
        self.assertEqual(first, second)
        self.assertEqual(third, 2)

    def test_arxiv_abstract_url_becomes_html(self):
        self.assertEqual(
            prefer_arxiv_html("https://arxiv.org/abs/2212.10496"),
            "https://arxiv.org/html/2212.10496",
        )

    def test_local_fetch_is_blocked(self):
        message = ResearchTools().fetch_page("http://127.0.0.1/secret")
        self.assertIn("Private", message)

    def test_extract_keeps_article_and_drops_nav(self):
        html = (
            "<html><head><title>T</title></head><body>"
            "<nav>Home About</nav><article><p>HyDE embeds a hypothetical document.</p></article>"
            "</body></html>"
        )
        title, text = extract_content(html)
        self.assertEqual(title, "T")
        self.assertIn("hypothetical document", text)
        self.assertNotIn("Home About", text)


class MemoryTests(unittest.TestCase):
    def test_buffer_drops_oldest_step(self):
        buffer = ConversationBuffer(char_budget=30)
        buffer.add("a" * 20)
        rendered = buffer.add("b" * 20)
        self.assertEqual(buffer.dropped, 1)
        self.assertNotIn("aaa", rendered)
        self.assertIn("bbb", rendered)
        self.assertIn("earlier steps removed", rendered)

    def test_recall_prefers_overlapping_notes(self):
        hyde = {"question": "What is HyDE retrieval?", "summary": "HyDE embeds a hypothetical document."}
        other = {"question": "Refund policy for shoes", "summary": "Returns within 30 days."}
        self.assertGreater(_score("Compare HyDE and Raptor", hyde), _score("Compare HyDE and Raptor", other))


class LoopTests(unittest.TestCase):
    def test_goal_met(self):
        replies = [
            (
                'Thought: search.\nAction: web_search\nAction Input: {"query": "HyDE"}',
                4,
                4,
            ),
            (
                'Thought: read it.\nAction: fetch_page\nAction Input: {"url": "https://example.com/hyde"}',
                4,
                4,
            ),
            (
                'Thought: read the other.\nAction: fetch_page\nAction Input: {"url": "https://example.com/raptor"}',
                4,
                4,
            ),
            ("Thought: done.\nFinal Answer: HyDE writes a fake document before retrieval [1]. Raptor builds a tree [2].", 4, 4),
        ]

        def complete(prompt):
            text, inp, out = replies.pop(0)
            return text, inp, out

        class FakeTools:
            def __init__(self):
                self.tracker = CitationTracker()
                self.calls = []

            def dispatch(self, action, payload):
                self.calls.append(action)
                if action == "fetch_page":
                    number = self.tracker.add(payload["url"], "HyDE paper")
                    return f"Recorded as source [{number}]. HyDE generates a hypothetical document."
                return "WEAK RESULTS: 1. HyDE\nURL: https://example.com/hyde"

        class FakeMemory:
            backend = "local"

            def recall(self, question):
                return "No earlier research is stored."

            def save(self, question, summary, sources):
                self.saved = (question, summary, sources)

        tools = FakeTools()
        memory = FakeMemory()
        result = research("What is HyDE?", complete=complete, tools=tools, memory=memory)
        self.assertEqual(result.report.stop_reason, "goal_met")
        self.assertEqual(tools.calls, ["web_search", "fetch_page", "fetch_page"])
        self.assertEqual(result.report.sources[0].id, 1)
        self.assertEqual(len(result.report.sources), 2)
        self.assertIn("[1]", memory.saved[1])
        self.assertEqual(result.report.total_tokens, 32)

    def test_iteration_budget_still_summarizes(self):
        def complete(prompt):
            if prompt.startswith("Write the final summary"):
                return "Final Answer: Partial note from the searches [1].", 3, 3
            return 'Thought: again.\nAction: web_search\nAction Input: {"query": "Raptor"}', 2, 2

        class FakeTools:
            def __init__(self):
                self.tracker = CitationTracker()

            def dispatch(self, action, payload):
                self.tracker.add("https://example.com/raptor", "Raptor")
                return "1. Raptor\nURL: https://example.com/raptor"

        class FakeMemory:
            backend = "local"

            def recall(self, question):
                return "No earlier research is stored."

            def save(self, question, summary, sources):
                self.saved = summary

        memory = FakeMemory()
        result = research(
            "What is Raptor RAG?",
            complete=complete,
            tools=FakeTools(),
            memory=memory,
            max_iterations=2,
            max_tokens=1000,
            max_seconds=30,
        )
        self.assertEqual(result.report.stop_reason, "max_iterations")
        self.assertIn("Partial note", result.report.summary)
        self.assertIn("Partial note", memory.saved)

    def test_early_finish_is_sent_back_to_read(self):
        replies = [
            ("Thought: enough.\nFinal Answer: nothing yet", 1, 1),
            ('Thought: search.\nAction: web_search\nAction Input: {"query": "HyDE"}', 1, 1),
            ('Thought: read.\nAction: fetch_page\nAction Input: {"url": "https://example.com/a"}', 1, 1),
            ('Thought: read more.\nAction: fetch_page\nAction Input: {"url": "https://example.com/b"}', 1, 1),
            ("Thought: done.\nFinal Answer: HyDE [1] and Raptor [2].", 1, 1),
        ]

        def complete(prompt):
            return replies.pop(0)

        class FakeTools:
            def __init__(self):
                self.tracker = CitationTracker()
                self.calls = []

            def dispatch(self, action, payload):
                self.calls.append(action)
                if action == "fetch_page":
                    number = self.tracker.add(payload["url"], "Page")
                    return f"Recorded as source [{number}]."
                return "results"

        class FakeMemory:
            backend = "local"

            def recall(self, question):
                return "No earlier research is stored."

            def save(self, question, summary, sources):
                self.saved = summary

        tools = FakeTools()
        result = research("Compare them", complete=complete, tools=tools, memory=FakeMemory())
        self.assertEqual(result.report.stop_reason, "goal_met")
        self.assertEqual(tools.calls[0], "web_search")
        self.assertIn("No pages have been read", result.trace[0]["observation"])
        self.assertIn("[1]", result.report.summary)


class GraphTests(unittest.TestCase):
    def test_graph_stops_on_final_answer(self):
        from m3_langgraph_agent import research_graph

        replies = [
            'Thought: search.\nAction: web_search\nAction Input: {"query": "HyDE"}',
            'Thought: read.\nAction: fetch_page\nAction Input: {"url": "https://example.com/hyde"}',
            'Thought: read more.\nAction: fetch_page\nAction Input: {"url": "https://example.com/raptor"}',
            "Thought: done.\nFinal Answer: HyDE is a query transform [1]. Raptor is a tree [2].",
        ]

        def complete(prompt):
            return replies.pop(0), 1, 1

        class FakeTools:
            def __init__(self):
                self.tracker = CitationTracker()

            def dispatch(self, action, payload):
                if action == "fetch_page":
                    number = self.tracker.add(payload["url"], payload["url"])
                    return f"Recorded as source [{number}]."
                return "two results"

        class FakeMemory:
            backend = "local"

            def recall(self, question):
                return "No earlier research is stored."

            def save(self, question, summary, sources):
                self.saved = summary

        result = research_graph(
            "What is HyDE?",
            complete=complete,
            tools=FakeTools(),
            memory=FakeMemory(),
            max_iterations=4,
        )
        self.assertEqual(result.report.stop_reason, "goal_met")
        self.assertIn("[1]", result.report.summary)
        self.assertEqual(len(result.report.sources), 2)


if __name__ == "__main__":
    unittest.main()
