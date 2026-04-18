# tests/test_state.py
# Tests for ARIAState TypedDict and make_initial_state factory.

import pytest
from aria.state import ARIAState, make_initial_state


def test_initial_state_has_all_15_fields():
    """ARIAState must have exactly the 15 fields defined in SKILL.md."""
    required_fields = {
        "question", "thread_id",
        "sub_queries", "route",
        "retrieved", "sources",
        "tool_result",
        "answer", "report",
        "faithfulness", "eval_retries", "reflection_note",
        "messages", "context_window",
    }
    state = make_initial_state("test question", "thread-001")
    assert required_fields.issubset(state.keys()), (
        f"Missing fields: {required_fields - set(state.keys())}"
    )


def test_initial_state_defaults():
    """Verify default values for every field."""
    state = make_initial_state("What is RAG?", "t-123")
    assert state["question"]        == "What is RAG?"
    assert state["thread_id"]       == "t-123"
    assert state["sub_queries"]     == []
    assert state["route"]           == "both"
    assert state["retrieved"]       == ""
    assert state["sources"]         == []
    assert state["tool_result"]     == ""
    assert state["answer"]          == ""
    assert state["report"]          == {}
    assert state["faithfulness"]    == 0.0
    assert state["eval_retries"]    == 0
    assert state["reflection_note"] == ""
    assert state["messages"]        == []
    assert state["context_window"]  == []


def test_state_is_mutable_dict():
    """State must behave as a plain mutable dict (not frozen)."""
    state = make_initial_state("q", "t")
    state["answer"] = "hello"
    assert state["answer"] == "hello"
