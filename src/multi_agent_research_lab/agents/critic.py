"""Optional critic agent skeleton for bonus work."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState


class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = "critic"

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append findings.

        This optional agent stays lightweight: it checks whether the final answer exists
        and whether available sources are referenced.
        """

        findings: list[str] = []
        if not state.final_answer:
            findings.append("No final answer is available for critique.")
        elif state.sources and "[1]" not in state.final_answer:
            findings.append("Final answer may be missing bracket citations to available sources.")
        else:
            findings.append("Final answer passed basic citation/existence checks.")

        content = "\n".join(f"- {finding}" for finding in findings)
        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC,
                content=content,
                metadata={"finding_count": len(findings)},
            )
        )
        state.add_trace_event(
            "agent_completed",
            {"agent": self.name, "finding_count": len(findings)},
        )
        return state
