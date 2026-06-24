"""Researcher agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.search_client import SearchClient


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self, search_client: SearchClient | None = None) -> None:
        self.search_client = search_client or SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`.

        The default search client is local and deterministic. It can be swapped for a
        real web or internal-document search provider without changing the agent.
        """

        sources = self.search_client.search(
            query=state.request.query,
            max_results=state.request.max_sources,
        )
        state.sources = self._deduplicate_sources(sources)
        research_notes = self._render_notes(state)
        state.research_notes = research_notes

        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=research_notes,
                metadata={"source_count": len(state.sources)},
            )
        )
        state.add_trace_event(
            "agent_completed",
            {"agent": self.name, "source_count": len(state.sources)},
        )
        return state

    @staticmethod
    def _deduplicate_sources(sources: list[SourceDocument]) -> list[SourceDocument]:
        seen: set[str] = set()
        unique_sources: list[SourceDocument] = []
        for source in sources:
            key = source.url or source.title.lower()
            if key in seen:
                continue
            seen.add(key)
            unique_sources.append(source)
        return unique_sources

    @staticmethod
    def _render_notes(state: ResearchState) -> str:
        lines = [f"Research notes for: {state.request.query}", ""]
        if not state.sources:
            lines.append(
                "- No sources were found. Continue with a clearly labeled "
                "low-confidence answer."
            )
            return "\n".join(lines)

        for index, source in enumerate(state.sources, start=1):
            lines.append(f"- [{index}] {source.title}: {source.snippet}")
        return "\n".join(lines)
