# aria/state.py
# ARIAState — Single source of truth for all graph data
# Every field that any node reads OR writes MUST be here.

from typing import TypedDict, List, Optional, Annotated
import operator


class ARIAState(TypedDict):
    # ── Input ────────────────────────────────────────────────
    question:         str               # current user question
    thread_id:        str               # session ID for MemorySaver

    # ── Cache Layer ──────────────────────────────────────────
    cache_store:      Annotated[List[dict], operator.add]   # session-level semantic cache
    cache_hit:        bool              # flag if current query bypassed generation
    current_embedding: List[float]      # current question's vector representation

    # ── Planning ─────────────────────────────────────────────
    sub_queries:      List[str]         # planner decomposition (3–5 items)
    route:            str               # "retrieve" | "tool" | "both"
    kb_coverage_score: float            # L2 distance score from KB coverage check
    comparison_mode:  bool              # True when "Compare:" prefix detected
    comparison_concepts: List[str]      # concepts extracted from comparison query

    # ── Retrieval ────────────────────────────────────────────
    retrieval_artifact: dict            # {"chunks": [...], "total_candidates": int, "reranked_to": int}
    retrieved:        str               # ChromaDB context chunks (legacy string)
    sources:          List[str]         # document names for citation
    reranker_scores:  List[float]       # cross-encoder reranking scores

    # ── Tool ─────────────────────────────────────────────────
    tool_artifact:    dict              # {"arxiv_results": [...], "web_results": [...]}
    tool_result:      str               # arXiv + web search results (legacy string)
    arxiv_papers:     List[dict]        # {title, date, abstract_snippet, url}

    # ── Generation ───────────────────────────────────────────
    contract_artifact: dict             # {"required_coverage": [], "required_sources": [], "safety_constraints": ""}
    answer_contract:  str               # sprint contract generated before answer (legacy string)
    answer:           str               # current best answer
    report:           dict              # {summary, key_findings, sources, follow_ups}

    # ── Evaluation (Multi-Axis Harness) ─────────────────────
    faithfulness:     float             # weighted quality score 0.0–1.0
    eval_scores:      dict              # {faithfulness, relevance, completeness, safety, weighted}
    eval_issues:      List[str]         # specific issues identified by evaluator
    eval_retries:     int               # loop guard
    max_retries:      int               # adaptive retry budget (2-3 based on complexity)
    reflection_note:  str               # critique from reflect_node

    # ── Memory ───────────────────────────────────────────────
    messages:         Annotated[List[dict], operator.add]   # full history
    context_window:   List[dict]        # sliding window (last 10 pairs)

    # ── User Profile ─────────────────────────────────────────
    user_context:     str               # injected profile context string

    # ── LLM Failover ─────────────────────────────────────────
    llm_provider_used: str              # "groq" or "gemini"

    # ── GraphRAG ─────────────────────────────────────────────
    graph_context:    str               # entity-relationship context from GraphRAG


def make_initial_state(question: str, thread_id: str) -> ARIAState:
    """Return a clean initial state for a new graph invocation."""
    return {
        "question":          question,
        "thread_id":         thread_id,
        "cache_store":       [],
        "cache_hit":         False,
        "current_embedding": [],
        "sub_queries":       [],
        "route":             "both",
        "kb_coverage_score": 0.0,
        "comparison_mode":   False,
        "comparison_concepts": [],
        "retrieval_artifact": {},
        "retrieved":         "",
        "sources":           [],
        "reranker_scores":   [],
        "tool_artifact":     {},
        "tool_result":       "",
        "arxiv_papers":      [],
        "contract_artifact": {},
        "answer_contract":   "",
        "answer":            "",
        "report":            {},
        "faithfulness":      0.0,
        "eval_scores":       {},
        "eval_issues":       [],
        "eval_retries":      0,
        "max_retries":       2,
        "reflection_note":   "",
        "messages":          [],
        "context_window":    [],
        "user_context":      "",
        "llm_provider_used": "groq",
        "graph_context":     "",
    }
