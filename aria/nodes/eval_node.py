# aria/nodes/eval_node.py
# Multi-axis evaluation node inspired by Anthropic's GAN harness.
# Scores across 4 axes: faithfulness, relevance, completeness, safety.
# Computes weighted score and provides structured feedback for reflect_node.

import json
import re
from aria.state import ARIAState
from aria.config import get_llm
from aria.prompts.eval_prompt import EVAL_PROMPT


def eval_node(state: ARIAState) -> dict:
    # ... existing prompt and invocation logic ...
    llm = get_llm()
    # Combine JSON artifacts for the evaluator
    retrieval_json = json.dumps(state.get("retrieval_artifact", {}), indent=2)[:3000]
    tool_json = json.dumps(state.get("tool_artifact", {}), indent=2)[:3000]
    contract_json = json.dumps(state.get("contract_artifact", {}), indent=2)[:1000]

    # Include sub-queries so evaluator can check completeness
    sub_queries_str = "\n".join(
        f"  - {sq}" for sq in state.get("sub_queries", [])
    ) or "  (none planned)"

    prompt = EVAL_PROMPT.format(
        answer=state["answer"],
        retrieval_artifact=retrieval_json,
        tool_artifact=tool_json,
        contract_artifact=contract_json,
        question=state["question"],
        sub_queries=sub_queries_str,
    )

    response = llm.invoke(prompt)
    text = re.sub(r"```json\s*|\s*```", "", response.content).strip()

    eval_scores = {}
    eval_issues = []
    try:
        result = json.loads(text)
        faith = float(result.get("faithfulness", 0.5))
        relev = float(result.get("relevance", 0.5))
        compl = float(result.get("completeness", 0.5))
        safety = float(result.get("safety", 0.5))
        weighted = (faith * 0.35) + (relev * 0.30) + (compl * 0.20) + (safety * 0.15)
        eval_scores = {
            "faithfulness": round(faith, 3), "relevance": round(relev, 3),
            "completeness": round(compl, 3), "safety": round(safety, 3), "weighted": round(weighted, 3),
        }
        eval_issues = result.get("issues", [])
    except (json.JSONDecodeError, ValueError):
        weighted = 0.5
        eval_scores = {
            "faithfulness": 0.5, "relevance": 0.5, "completeness": 0.5, "safety": 0.5, "weighted": 0.5,
        }

    return {
        "eval_scores": eval_scores,
        "eval_issues": eval_issues,
        "faithfulness": max(0.0, min(1.0, weighted)),
        "max_retries": _get_max_retries(state),
        "eval_retries": state.get("eval_retries", 0) + 1
    }


def _get_max_retries(state: ARIAState) -> int:
    """
    Adaptive retry budget inspired by Anthropic's finding that iteration
    depth should scale with task complexity.
    """
    if state.get("comparison_mode"):
        return 3  # comparison queries need more refinement
    route = state.get("route", "both")
    if route == "both":
        return 2  # hybrid retrieval = moderate complexity
    return 2  # default
