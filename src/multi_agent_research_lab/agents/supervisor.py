"""Supervisor / router skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.state import ResearchState


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route.

        Routing policy:
        - research first when sources or notes are missing
        - analyze once research notes exist
        - write once analysis notes exist
        - stop when a final answer exists or max iterations is reached
        """

        if state.iteration >= self.settings.max_iterations:
            route = "done"
            if not state.final_answer:
                state.errors.append("Stopped by max_iterations before a final answer was produced.")
        elif state.final_answer:
            route = "done"
        elif not state.sources or not state.research_notes:
            route = "researcher"
        elif not state.analysis_notes:
            route = "analyst"
        else:
            route = "writer"

        state.record_route(route)
        state.add_trace_event(
            "supervisor_route",
            {
                "route": route,
                "iteration": state.iteration,
                "has_research": bool(state.research_notes),
                "has_analysis": bool(state.analysis_notes),
                "has_final": bool(state.final_answer),
            },
        )
        return state
