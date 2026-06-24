"""Search client abstraction for ResearcherAgent."""

import re

from multi_agent_research_lab.core.schemas import SourceDocument


class SearchClient:
    """Provider-agnostic search client with a deterministic local corpus.

    The lab can run without paid search APIs. Replace or extend this class with Tavily,
    Bing, SerpAPI, or an internal document retriever when those integrations are needed.
    """

    _LOCAL_CORPUS = [
        SourceDocument(
            title="Effective Multi-Agent System Design",
            url="local://multi-agent-design",
            snippet=(
                "Strong multi-agent systems use distinct roles, explicit handoffs, "
                "bounded routing, and shared state that preserves enough context for debugging."
            ),
            metadata={"source_type": "local_mock", "topic": "architecture"},
        ),
        SourceDocument(
            title="Research Agent Guardrails",
            url="local://agent-guardrails",
            snippet=(
                "Production agent workflows need max iterations, timeouts, retries, fallback "
                "paths, and validation checks to avoid infinite loops and low-quality outputs."
            ),
            metadata={"source_type": "local_mock", "topic": "guardrails"},
        ),
        SourceDocument(
            title="Benchmarking Single-Agent and Multi-Agent Workflows",
            url="local://agent-benchmarking",
            snippet=(
                "Compare baseline and multi-agent runs with latency, cost, quality, citation "
                "coverage, and failure rate instead of judging only by a polished final answer."
            ),
            metadata={"source_type": "local_mock", "topic": "evaluation"},
        ),
        SourceDocument(
            title="GraphRAG Research Pattern",
            url="local://graphrag-pattern",
            snippet=(
                "GraphRAG combines retrieval with graph-structured relationships so a system "
                "can connect entities, claims, and evidence across a broader context."
            ),
            metadata={"source_type": "local_mock", "topic": "retrieval"},
        ),
        SourceDocument(
            title="Writer Agent Synthesis Checklist",
            url="local://writer-synthesis",
            snippet=(
                "A writer agent should synthesize research and analysis into a clear answer, "
                "preserve caveats, and cite source references for important claims."
            ),
            metadata={"source_type": "local_mock", "topic": "writing"},
        ),
    ]

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query.

        This default implementation ranks a small local corpus by token overlap so the
        lab remains runnable offline and in classrooms without external search keys.
        """

        query_terms = self._tokenize(query)
        ranked = sorted(
            self._LOCAL_CORPUS,
            key=lambda document: self._score(document, query_terms),
            reverse=True,
        )
        return ranked[:max_results]

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) >= 3}

    @classmethod
    def _score(cls, document: SourceDocument, query_terms: set[str]) -> tuple[int, int]:
        haystack = cls._tokenize(f"{document.title} {document.snippet}")
        overlap = len(query_terms & haystack)
        return overlap, len(haystack)
