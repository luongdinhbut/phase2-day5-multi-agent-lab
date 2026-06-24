"""Benchmark skeleton for single-agent vs multi-agent."""

import re
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]


def run_benchmark(
    run_name: str,
    query: str,
    runner: Runner,
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency and return simple offline benchmark metrics."""

    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started
    citation_coverage = _citation_coverage(state)
    quality_score = _quality_score(state, citation_coverage)
    notes = (
        f"sources={len(state.sources)}; errors={len(state.errors)}; "
        f"citation_coverage={citation_coverage:.2f}"
    )
    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        quality_score=quality_score,
        notes=notes,
    )
    return state, metrics


def _citation_coverage(state: ResearchState) -> float:
    if not state.sources:
        return 0.0
    if not state.final_answer:
        return 0.0

    cited_indices = {
        int(match)
        for match in re.findall(r"\[(\d+)\]", state.final_answer)
        if int(match) <= len(state.sources)
    }
    return len(cited_indices) / len(state.sources)


def _quality_score(state: ResearchState, citation_coverage: float) -> float:
    score = 4.0
    if state.final_answer:
        score += 2.0
    if state.research_notes:
        score += 1.0
    if state.analysis_notes:
        score += 1.0
    score += min(citation_coverage, 1.0) * 2.0
    score -= min(len(state.errors), 3) * 1.0
    return max(0.0, min(10.0, score))
