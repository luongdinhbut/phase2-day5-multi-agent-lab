# Design Template

## Problem

Build a research assistant that receives a user query, gathers supporting context,
extracts structured insights, and writes a final answer with source references and caveats.

## Why multi-agent?

A single-agent baseline is useful for simple questions, but it hides intermediate reasoning,
source selection, and failure points. A multi-agent workflow makes the handoff explicit:
Supervisor routes the work, Researcher gathers evidence, Analyst checks claims and gaps,
and Writer creates the final response.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| Supervisor | Route the next step and enforce max iterations | Shared state | Route in `route_history` | Stops with error note if max iterations is hit |
| Researcher | Retrieve and summarize sources | Query, max source count | `sources`, `research_notes` | Falls back to local mock corpus |
| Analyst | Extract claims, gaps, risks, and answer shape | Research notes, sources | `analysis_notes` | Uses deterministic analysis if FireworksAI is unavailable |
| Writer | Synthesize final answer with citations and caveats | Query, research notes, analysis notes | `final_answer` | Uses deterministic writer if FireworksAI is unavailable |

## Shared state

- `request`: user query, audience, and source limit.
- `iteration`: guardrail for bounded routing.
- `route_history`: traceable sequence of decisions.
- `sources`: retrieved supporting documents.
- `research_notes`: concise evidence notes.
- `analysis_notes`: structured claims, risks, and gaps.
- `final_answer`: user-facing response.
- `agent_results`: per-agent output for inspection.
- `trace`: lightweight local events for debugging.
- `errors`: fallback and validation notes.

## Routing policy

```text
Supervisor -> Researcher -> Supervisor -> Analyst -> Supervisor -> Writer -> Supervisor -> done
```

Supervisor chooses `done` when `final_answer` exists or `MAX_ITERATIONS` is reached.

## Guardrails

- Max iterations: `MAX_ITERATIONS`, default `6`.
- Timeout: `TIMEOUT_SECONDS`, default `60`, used by FireworksAI requests.
- Retry: LLM requests retry transient HTTP/network failures with exponential backoff.
- Fallback: Analyst and Writer use deterministic local output when FireworksAI is unavailable.
- Validation: workflow rejects unknown routes; critic can check final answer/citation presence.

## Benchmark plan

Queries are listed in `configs/lab_default.yaml`.

Metrics:

- Latency: wall-clock runtime.
- Quality: heuristic based on final answer, research notes, analysis notes, citations and errors.
- Citation coverage: cited source indices divided by available sources.
- Failure signal: count of errors recorded in shared state.

Expected outcome: multi-agent runs should be more traceable and easier to debug than baseline
runs, while baseline may remain faster for simple prompts.
