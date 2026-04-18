# aria/nodes/save_node.py
# Persists the final answer to message history and resets per-turn fields.

from aria.state import ARIAState

WINDOW_SIZE = 10


def save_node(state: ARIAState) -> dict:
    """
    Append the assistant answer (with report metadata) to full history,
    refresh the sliding context window, and reset all per-turn fields
    so the next invocation starts clean.
    """
    answer_msg = {
        "role":         "assistant",
        "content":      state["answer"],
        "report":       state["report"],
        "sources":      state["sources"],
        "faithfulness": state["faithfulness"],
    }
    
    # Update local history for window calculation
    full_history = state.get("messages", []) + [answer_msg]
    context_window = full_history[-(WINDOW_SIZE * 2):]

    # Handle Cache Storage (Pure update)
    cache_update = []
    if not state.get("cache_hit", False) and state.get("current_embedding") and state.get("report"):
        cache_entry = {
            "question_embedding": state["current_embedding"],
            "report": state["report"],
            "answer": state["answer"],
            "sources": state["sources"],
        }
        cache_update = [cache_entry]

    # Handle User Profile (Side effect)
    try:
        from aria.user_profile import load_profile, update_profile, save_profile
        user_id = state.get("thread_id", "default")[:16]
        profile = load_profile(user_id)
        profile = update_profile(
            profile, state["question"], state["report"],
            sub_queries=state.get("sub_queries", [])
        )
        save_profile(user_id, profile)
    except Exception as e:
        print(f"[SaveNode] Profile update skipped: {e}")

    # Return partial update
    return {
        "messages":         [answer_msg],
        "cache_store":      cache_update, # Annotated with operator.add in state? Let's check.
        "context_window":   context_window,
        "reflection_note":  "",
        "eval_retries":     0,
        "cache_hit":        False,
        "current_embedding": [],
        "comparison_mode":  False,
        "comparison_concepts": []
    }
