"""Writer agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm_client = llm_client or LLMClient(settings=self.settings)

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`.

        Uses FireworksAI when configured; otherwise renders a deterministic answer
        from shared state with source references.
        """

        used_llm = False
        if self.settings.fireworks_api_key:
            try:
                final_answer = self._run_llm_writer(state)
                used_llm = True
            except AgentExecutionError as exc:
                state.errors.append(f"Writer LLM fallback used: {exc}")
                final_answer = self._run_local_writer(state)
        else:
            final_answer = self._run_local_writer(state)

        state.final_answer = final_answer

        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=final_answer,
                metadata={
                    "used_llm": used_llm,
                    "source_count": len(state.sources),
                },
            )
        )
        state.add_trace_event(
            "agent_completed",
            {
                "agent": self.name,
                "used_llm": used_llm,
                "has_final_answer": bool(state.final_answer),
            },
        )
        return state

    def _run_llm_writer(self, state: ResearchState) -> str:
        response = self.llm_client.complete(
            system_prompt=(
                "You are the writer in a multi-agent research workflow. "
                "Synthesize a clear final answer for the target audience. "
                "Use bracket citations like [1] when citing provided sources. "
                "State caveats when evidence is weak."
            ),
            user_prompt=(
                f"Audience: {state.request.audience}\n"
                f"User query:\n{state.request.query}\n\n"
                f"Research notes:\n{state.research_notes or 'No research notes.'}\n\n"
                f"Analysis notes:\n{state.analysis_notes or 'No analysis notes.'}"
            ),
        )
        return response.content

    @staticmethod
    def _run_local_writer(state: ResearchState) -> str:
        lines = [
            f"Answer for: {state.request.query}",
            "",
            "Summary",
            (
                "A practical multi-agent research workflow should split responsibilities "
                "between routing, evidence gathering, analysis, and final synthesis. "
                "This keeps each step easier to debug and benchmark."
            ),
            "",
            "Evidence",
        ]

        if state.sources:
            for index, source in enumerate(state.sources, start=1):
                lines.append(f"- [{index}] {source.title}: {source.snippet}")
        else:
            lines.append("- No retrieved sources were available for this run.")

        lines.extend(
            [
                "",
                "Synthesis",
                (
                    "Use the single-agent baseline for simple, low-risk tasks. Use the "
                    "multi-agent workflow when the task benefits from explicit handoffs, "
                    "separate evidence review, traceability, and failure controls."
                ),
                "",
                "Caveats",
                (
                    "The default search client uses a local mock corpus, so fresh facts "
                    "should be verified with a real search provider before production use."
                ),
            ]
        )

        if state.analysis_notes:
            lines.extend(["", "Analyst notes", state.analysis_notes])

        return "\n".join(lines)
