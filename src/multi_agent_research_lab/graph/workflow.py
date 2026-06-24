"""Multi-agent workflow orchestration.

The class deliberately avoids requiring LangGraph at runtime. Learners can replace
`build` with a real LangGraph graph later, while this default implementation keeps
the lab runnable with only core dependencies.
"""

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.state import ResearchState


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        supervisor: SupervisorAgent | None = None,
        agents: dict[str, BaseAgent] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.supervisor = supervisor or SupervisorAgent(settings=self.settings)
        self.agents = agents or {
            "researcher": ResearcherAgent(),
            "analyst": AnalystAgent(settings=self.settings),
            "writer": WriterAgent(settings=self.settings),
        }

    def build(self) -> dict[str, BaseAgent]:
        """Return the runnable node registry.

        A future LangGraph implementation can use this same registry as graph nodes.
        """

        return {"supervisor": self.supervisor, **self.agents}

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return final state.

        The supervisor records a route on each loop. Worker agents then mutate the
        shared state and hand it back to the supervisor until `done` or max iterations.
        """

        self.build()
        while state.iteration < self.settings.max_iterations:
            state = self.supervisor.run(state)
            route = state.route_history[-1]

            if route == "done":
                state.add_trace_event("workflow_completed", {"reason": "supervisor_done"})
                return state

            agent = self.agents.get(route)
            if agent is None:
                raise AgentExecutionError(f"Supervisor selected unknown route: {route}")

            state = agent.run(state)

        if not state.final_answer:
            state.errors.append("Workflow hit max_iterations; writer fallback executed.")
            writer = self.agents.get("writer")
            if writer is None:
                raise AgentExecutionError("Writer fallback is unavailable.")
            state = writer.run(state)

        state.add_trace_event("workflow_completed", {"reason": "max_iterations"})
        return state
