# aria/graph.py
# Assembles the 7-node ARIA StateGraph with MemorySaver checkpointing.
# Node execution order:
#   memory_node → planner_node → [retrieve_node ‖ tool_node] →
#   answer_node → eval_node → (reflect_node → answer_node)* → save_node

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from aria.state import ARIAState
from aria.nodes import (
    memory_node,
    planner_node,
    retrieve_node,
    tool_node,
    answer_node,
    eval_node,
    reflect_node,
    save_node,
    cache_node,
    contract_node,
)


# ── Routing functions ────────────────────────────────────────────────────────

def route_cache(state: ARIAState) -> str:
    """
    After cache_node: decide if hit or miss.
    """
    if state.get("cache_hit"):
        return "save_node"
    return "planner_node"

def route_retrieval(state: ARIAState) -> str:
    """
    After planner_node: decide which branches run.
    Returns one of: 'retrieve_only' | 'tool_only' | 'both'
    """
    r = state.get("route", "both")
    if r == "retrieve":
        return "retrieve_only"
    elif r == "tool":
        return "tool_only"
    return "both"


def route_from_retrieve(state: ARIAState) -> str:
    """
    After retrieve_node: decide if tool_node is needed.
    """
    if state.get("route", "both") == "both":
        return "tool_node"
    return "contract_node"


def route_after_eval(state: ARIAState) -> str:
    """
    After eval_node: pass, retry, or force-pass.
    Uses adaptive retry budget (max_retries set by eval_node based on complexity).
    Returns one of: 'save_node' | 'reflect_node'
    """
    faithfulness = state.get("faithfulness", 0.0)
    retries      = state.get("eval_retries", 0)
    max_retries  = state.get("max_retries", 2)   # adaptive budget

    if faithfulness >= 0.65:
        return "save_node"          # weighted score passes threshold
    if retries >= max_retries:
        return "save_node"          # force-pass after exhausting retries
    return "reflect_node"           # trigger reflection loop


# ── Graph assembly ───────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Build and compile the ARIA StateGraph.
    Returns a compiled graph with MemorySaver checkpointing.
    """
    builder = StateGraph(ARIAState)

    # ── Register all 8 nodes ─────────────────────────────────────────────────
    builder.add_node("memory_node",   memory_node)
    builder.add_node("planner_node",  planner_node)
    builder.add_node("retrieve_node", retrieve_node)
    builder.add_node("tool_node",     tool_node)
    builder.add_node("answer_node",   answer_node)
    builder.add_node("eval_node",     eval_node)
    builder.add_node("reflect_node",  reflect_node)
    builder.add_node("save_node",     save_node)
    builder.add_node("cache_node",    cache_node)
    builder.add_node("contract_node", contract_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    builder.set_entry_point("memory_node")
    builder.add_edge("memory_node", "cache_node")

    # ── Conditional routing after cache ───────────────────────────────────────
    builder.add_conditional_edges(
        "cache_node",
        route_cache,
        {
            "save_node": "save_node",
            "planner_node": "planner_node",
        },
    )

    # ── Conditional routing after planner ─────────────────────────────────────
    # route="retrieve" → skip tool_node entirely
    # route="tool"     → skip retrieve_node entirely
    # route="both"     → retrieve first, then tool (sequential for simplicity)
    builder.add_conditional_edges(
        "planner_node",
        route_retrieval,
        {
            "retrieve_only": "retrieve_node",
            "tool_only":     "tool_node",
            "both":          "retrieve_node",   # retrieve first, tool second
        },
    )
    
    # ── Conditional routing after retrieve ────────────────────────────────────
    builder.add_conditional_edges(
        "retrieve_node",
        route_from_retrieve,
        {
            "tool_node": "tool_node",
            "contract_node": "contract_node"
        }
    )
    
    # When both run: retrieve_node → tool_node → contract_node
    # If tool only: planner_node → tool_node → contract_node
    builder.add_edge("tool_node",     "contract_node")
    builder.add_edge("contract_node", "answer_node")

    # ── Faithfulness reflection loop ──────────────────────────────────────────
    builder.add_edge("answer_node", "eval_node")
    builder.add_conditional_edges(
        "eval_node",
        route_after_eval,
        {
            "save_node":    "save_node",
            "reflect_node": "reflect_node",
        },
    )
    builder.add_edge("reflect_node", "answer_node")   # retry answer after critique
    builder.add_edge("save_node", END)

    # ── Compile with MemorySaver (thread_id-based persistence) ───────────────
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)
