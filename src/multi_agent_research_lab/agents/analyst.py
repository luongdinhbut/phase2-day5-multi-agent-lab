"""Analyst agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm_client = llm_client or LLMClient(settings=self.settings)

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`.

        Uses FireworksAI when configured; otherwise falls back to a deterministic
        analysis so the lab can run offline.
        """

        used_llm = False
        if self.settings.fireworks_api_key:
            try:
                analysis_notes = self._run_llm_analysis(state)
                used_llm = True
            except AgentExecutionError as exc:
                state.errors.append(f"Analyst LLM fallback used: {exc}")
                analysis_notes = self._run_local_analysis(state)
        else:
            analysis_notes = self._run_local_analysis(state)

        state.analysis_notes = analysis_notes

        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=analysis_notes,
                metadata={"used_llm": used_llm},
            )
        )
        state.add_trace_event(
            "agent_completed",
            {"agent": self.name, "used_llm": used_llm},
        )
        return state

    def _run_llm_analysis(self, state: ResearchState) -> str:
        response = self.llm_client.complete(
            system_prompt=(
                "You are the analyst in a multi-agent research workflow. "
                "Extract key claims, evidence strength, gaps, and risks. "
                "Keep the answer structured and concise."
            ),
            user_prompt=(
                f"User query:\n{state.request.query}\n\n"
                f"Research notes:\n{state.research_notes or 'No research notes.'}"
            ),
        )
        return response.content

    @staticmethod
    def _run_local_analysis(state: ResearchState) -> str:
        source_count = len(state.sources)
        evidence_level = "moderate" if source_count >= 3 else "limited"

        lines = [
            "Analysis notes",
            "",
            f"- User need: answer for {state.request.audience}.",
            f"- Evidence level: {evidence_level} ({source_count} source(s) available).",
            "- Key claims:",
        ]

        if state.sources:
            for source in state.sources[:3]:
                lines.append(f"  - {source.snippet}")
        else:
            lines.append("  - No retrieved evidence; final answer must mark uncertainty.")

        lines.extend(
            [
                "- Weak spots:",
                "  - Local mock sources are useful for lab flow, not authoritative research.",
                "  - Fresh external facts should be verified before production use.",
                "- Recommended answer shape:",
                "  - Start with a direct synthesis, then show evidence and caveats.",
            ]
        )
        return "\n".join(lines)
