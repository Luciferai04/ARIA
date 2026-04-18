# aria/nodes/planner_node.py
# Decomposes the user question into 3-5 sub-queries and decides retrieval route.
# This is ARIA's core differentiator — Plan-Execute-Reflect-Synthesise.
# Now includes: comparison mode detection, KB coverage validation.

import json
import re
from aria.state import ARIAState
from aria.config import get_llm
from aria.prompts.planner_prompt import PLANNER_PROMPT


def _detect_comparison(question: str):
    """Detect comparison mode and extract concepts."""
    if question.lower().strip().startswith("compare:"):
        raw = question.split(":", 1)[1].strip()
        # Split on " vs " or ","
        if " vs " in raw.lower():
            concepts = [c.strip() for c in re.split(r'\s+vs\s+', raw, flags=re.IGNORECASE)]
        else:
            concepts = [c.strip() for c in raw.split(",")]
        concepts = [c for c in concepts if c]
        return True, concepts
    return False, []


def _validate_route(route: str, question: str) -> tuple:
    """
    Override LLM route decision based on actual KB coverage score.
    Returns (validated_route, kb_score).
    """
    try:
        from aria.knowledge_base import get_vectorstore
        vs = get_vectorstore()
        results = vs.similarity_search_with_score(question, k=1)
        if results:
            _, score = results[0]
            if score > 1.2:
                # Poor KB coverage — force to both
                return "both", round(score, 2)
            elif score <= 0.5:
                # Excellent KB coverage — allow retrieve
                if route == "both":
                    return "retrieve", round(score, 2)
                return route, round(score, 2)
            return route, round(score, 2)
    except Exception as e:
        print(f"[RouteValidator] Skipped: {e}")
    return route, 0.0


def planner_node(state: ARIAState) -> dict:
    """
    Use the LLM to:
      1. Decompose the user question into 3-5 focused sub-queries.
      2. Decide retrieval route: 'retrieve' | 'tool' | 'both'.

    Output is parsed from JSON; falls back gracefully on malformed output.
    """
    # ── Comparison mode detection ────────────────────────────
    comp_mode, comp_concepts = _detect_comparison(state["question"])

    llm = get_llm()

    # Build last-5-turn history string for context
    # history = state["context_window"][-10:]
    history_str = "\n".join(
        f"{m['role']}: {m['content']}" for m in state.get("context_window", [])[-10:]
    )

    prompt = PLANNER_PROMPT.format(
        history=history_str,
        question=state["question"],
    )

    response = llm.invoke(prompt)
    text = response.content

    # Strip markdown code fences if present
    text = re.sub(r"```json\s*|\s*```", "", text).strip()

    try:
        result = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # Fallback: treat original question as single sub-query, use both routes
        result = {
            "sub_queries": [state["question"]],
            "route": "both",
        }

    sub_queries = result.get("sub_queries", [state["question"]])
    route = result.get("route", "both")

    # Validate route value
    if route not in ("retrieve", "tool", "both"):
        route = "both"

    # ── KB coverage validation ───────────────────────────────
    validated_route, kb_score = _validate_route(route, state["question"])

    return {
        "comparison_mode": comp_mode,
        "comparison_concepts": comp_concepts,
        "kb_coverage_score": kb_score,
        "route": validated_route,
        "sub_queries": sub_queries[:5]  # hard cap at 5
    }
