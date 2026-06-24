"""Command-line entrypoint for the lab starter."""

from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a minimal single-agent baseline with FireworksAI."""

    _init()
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)
    settings = get_settings()
    client = LLMClient(settings=settings)

    try:
        response = client.complete(
            system_prompt=(
                "You are a concise research assistant for technical learners. "
                "Answer clearly, separate facts from assumptions, and mention "
                "when sources are needed."
            ),
            user_prompt=request.query,
        )
    except AgentExecutionError as exc:
        console.print(Panel.fit(str(exc), title="LLM Error", style="red"))
        raise typer.Exit(code=1) from exc

    state.final_answer = response.content
    state.add_trace_event(
        "baseline_llm",
        {
            "provider": "fireworks",
            "model": settings.fireworks_model,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
        },
    )
    console.print(Panel.fit(state.final_answer, title="Single-Agent Baseline"))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow."""

    _init()
    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow()
    try:
        result = workflow.run(state)
    except AgentExecutionError as exc:
        console.print(Panel.fit(str(exc), title="Workflow Error", style="red"))
        raise typer.Exit(code=1) from exc
    console.print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
