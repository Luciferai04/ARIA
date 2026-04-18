# aria/nodes/answer_node.py
# Synthesises a structured JSON research report from KB + tool context.
# Now includes: comparison mode, user context injection, LLM failover.

import json
import re
from aria.state import ARIAState
from aria.prompts.answer_prompt import ANSWER_PROMPT
from aria.prompts.comparison_prompt import COMPARISON_PROMPT


def answer_node(state: ARIAState) -> dict:
    """
    Invoke the LLM with all available context to produce a structured report:
      {summary, key_findings, sources, follow_ups}

    Handles JSON parse failures gracefully with a plain-text fallback.
    Incorporates reflection_note when retrying after low faithfulness.
    Supports comparison mode with a different prompt and output schema.
    """
    history_list = state.get("context_window", [])
    # Serialize JSON artifacts for structured prompt context
    retrieval_json = json.dumps(state.get("retrieval_artifact", {}), indent=2)[:4000]
    tool_json = json.dumps(state.get("tool_artifact", {}), indent=2)[:2000]
    contract_json = json.dumps(state.get("contract_artifact", {}), indent=2)

    # ── Comparison mode ──────────────────────────────────────
    if state.get("comparison_mode") and state.get("comparison_concepts"):
        prompt = COMPARISON_PROMPT.format(
            question=state["question"],
            concepts=", ".join(state["comparison_concepts"]),
            reflection_note=state.get("reflection_note", "") or "",
            retrieval_artifact=retrieval_json,
            tool_artifact=tool_json,
            contract_artifact=contract_json,
            history=history_str,
        )
    else:
        prompt = ANSWER_PROMPT.format(
            question=state["question"],
            reflection_note=state.get("reflection_note", "") or "",
            retrieval_artifact=retrieval_json,
            tool_artifact=tool_json,
            contract_artifact=contract_json,
            history=history_str,
        )

    # ── LLM invocation with failover ─────────────────────────
    try:
        from aria.llm_client import invoke_with_fallback
        response, provider = invoke_with_fallback(prompt, state)
    except Exception:
        # Ultimate fallback: use config.get_llm directly
        from aria.config import get_llm
        llm = get_llm()
        response = llm.invoke(prompt)
        provider = "groq"

    text = response.content
    text = re.sub(r"```json\s*|\s*```", "", text).strip()

    try:
        report = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # Graceful fallback — wrap plain text into expected schema
        report = {
            "summary":      text[:500],
            "key_findings": [],
            "sources":      state.get("sources", []),
            "follow_ups":   [],
        }

    return {
        "report": report,
        "answer": report.get("summary", text),
        "llm_provider_used": provider
    }
