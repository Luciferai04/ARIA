# aria/nodes/contract_node.py
# Implements Anthropic's 'Sprint Contract' pattern.
# Evaluates context and plans the "Answer Contract" specifying requirements for the final answer.

from aria.state import ARIAState
from aria.config import get_llm
from aria.prompts.contract_prompt import CONTRACT_PROMPT
import json
import re


def contract_node(state: ARIAState) -> dict:
    """
    Produce a pre-answer contract defining success criteria based on context.
    Provides targeted constraints to the Generator (answer_node).
    """
    llm = get_llm()

    context = (state.get("retrieved", "") + "\n" + state.get("tool_result", ""))[:4000]
    sub_queries_str = "\n".join(f"- {sq}" for sq in state.get("sub_queries", []))

    prompt = CONTRACT_PROMPT.format(
        question=state["question"],
        sub_queries=sub_queries_str or "(None)",
        context=context,
    )

    response = llm.invoke(prompt)
    text = re.sub(r"```json\s*|\s*```", "", response.content).strip()

    try:
        contract_dict = json.loads(text)
        # Build legacy string for backward compatibility with older prompts
        cov = "\n".join(f"- {c}" for c in contract_dict.get("required_coverage", []))
        src = "\n".join(f"- {s}" for s in contract_dict.get("required_sources", []))
        safe = contract_dict.get("safety_constraints", "")
        answer_contract = f"REQUIRED COVERAGE:\n{cov}\n\nREQUIRED SOURCES:\n{src}\n\nSAFETY CONSTRAINTS:\n{safe}"
        return {
            "contract_artifact": contract_dict,
            "answer_contract": answer_contract
        }
    except (json.JSONDecodeError, ValueError):
        return {
            "contract_artifact": {},
            "answer_contract": text  # Fallback to raw text
        }
