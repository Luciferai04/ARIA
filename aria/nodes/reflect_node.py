# aria/nodes/reflect_node.py
# Multi-axis critique generator for answers that fail the quality threshold.
# Uses per-axis scores and evaluator issues for targeted feedback.

from aria.state import ARIAState
from aria.config import get_llm
from aria.prompts.reflect_prompt import REFLECT_PROMPT
import json

def reflect_node(state: ARIAState) -> dict:
    # ... existing prompt and invocation logic ...
    llm = get_llm()

    retrieval_json = json.dumps(state.get("retrieval_artifact", {}), indent=2)[:3000]
    tool_json = json.dumps(state.get("tool_artifact", {}), indent=2)[:3000]
    contract_json = json.dumps(state.get("contract_artifact", {}), indent=2)[:1000]

    # Extract per-axis scores (with backward compat defaults)
    scores = state.get("eval_scores", {})
    issues = state.get("eval_issues", [])
    issues_str = "\n".join(f"  - {i}" for i in issues) if issues else "  (none specified)"

    prompt = REFLECT_PROMPT.format(
        faith_score=scores.get("faithfulness", state.get("faithfulness", 0.5)),
        relev_score=scores.get("relevance", 0.5),
        compl_score=scores.get("completeness", 0.5),
        safety_score=scores.get("safety", 0.5),
        weighted_score=scores.get("weighted", state.get("faithfulness", 0.5)),
        eval_issues=issues_str,
        answer=state["answer"],
        retrieval_artifact=retrieval_json,
        tool_artifact=tool_json,
        contract_artifact=contract_json,
        question=state["question"],
    )

    response = llm.invoke(prompt)
    return {"reflection_note": response.content}
