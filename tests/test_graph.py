# tests/test_graph.py
# Tests for graph structure, routing logic, and retrieval node logic.

import pytest
from aria.graph import build_graph, route_retrieval, route_after_eval
from aria.state import make_initial_state
from aria.nodes.retrieve_node import retrieve_node


def make_state(**overrides):
    s = make_initial_state("test", "t-001")
    s.update(overrides)
    return s


# ── Graph compilation ─────────────────────────────────────────────────────────

def test_graph_compiles():
    """The StateGraph must compile without raising any exceptions."""
    graph = build_graph()
    assert graph is not None


def test_graph_has_correct_nodes():
    """Verify all 8 nodes are present in the compiled graph."""
    graph = build_graph()
    node_names = set(graph.nodes.keys())
    expected = {
        "memory_node", "planner_node", "retrieve_node", "tool_node",
        "answer_node", "eval_node", "reflect_node", "save_node",
    }
    assert expected.issubset(node_names), f"Missing nodes: {expected - node_names}"


# ── route_retrieval ───────────────────────────────────────────────────────────

def test_route_retrieval_retrieve():
    state = make_state(route="retrieve")
    assert route_retrieval(state) == "retrieve_only"


def test_route_retrieval_tool():
    state = make_state(route="tool")
    assert route_retrieval(state) == "tool_only"


def test_route_retrieval_both():
    state = make_state(route="both")
    assert route_retrieval(state) == "both"


def test_route_retrieval_default():
    """Unknown/missing route defaults to 'both'."""
    state = make_state(route="unknown_value")
    assert route_retrieval(state) == "both"


# ── retrieve_node skipping logic ──────────────────────────────────────────────

def test_retrieve_node_skips_on_tool_route():
    """When route='tool', retrieve_node must return empty context."""
    state = make_state(route="tool", sub_queries=["test query"])
    result = retrieve_node(state)
    assert result["retrieved"] == ""
    assert result["sources"]   == []


def test_retrieve_node_runs_on_retrieve_route():
    """When route='retrieve', retrieve_node should attempt KB lookup (may return empty if KB not ingested)."""
    state = make_state(route="retrieve", sub_queries=["LangGraph agents"])
    result = retrieve_node(state)
    # retrieved may be empty if ChromaDB isn't populated — that's OK for unit tests
    assert isinstance(result["retrieved"], str)
    assert isinstance(result["sources"], list)
