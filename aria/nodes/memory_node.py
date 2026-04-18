# aria/nodes/memory_node.py
# Maintains conversation history and a sliding context window.

from aria.state import ARIAState

WINDOW_SIZE = 10   # keep last N user+assistant pairs → 2*N messages


def memory_node(state: ARIAState) -> dict:
    """
    Append the current user question to message history and
    refresh the sliding context window sent to LLM prompts.
    """
    new_msg = {"role": "user", "content": state["question"]}
    
    # Calculate window using the history + new message
    full_history = state.get("messages", []) + [new_msg]
    context_window = full_history[-(WINDOW_SIZE * 2):]

    return {
        "messages": [new_msg],
        "context_window": context_window
    }
