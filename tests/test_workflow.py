from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.services.search_client import SearchClient


def test_search_client_returns_ranked_local_sources() -> None:
    results = SearchClient().search("GraphRAG retrieval benchmark", max_results=2)

    assert len(results) == 2
    assert results[0].title
    assert results[0].snippet


def test_multi_agent_workflow_runs_offline_to_final_answer() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    workflow = MultiAgentWorkflow(settings=Settings(FIREWORKS_API_KEY=None))

    result = workflow.run(state)

    assert result.final_answer
    assert result.research_notes
    assert result.analysis_notes
    assert result.sources
    assert result.route_history[:3] == ["researcher", "analyst", "writer"]
    assert "done" in result.route_history
