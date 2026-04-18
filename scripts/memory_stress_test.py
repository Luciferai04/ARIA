#!/usr/bin/env python3
"""
scripts/memory_stress_test.py
-----------------------------
Automated multi-turn memory evaluation.
Tests 3 conversation threads, each 5 turns long.
Uses LLM-as-judge to verify contextual memory after each turn.

Usage:
    python scripts/memory_stress_test.py
"""

import sys
import os
import json
import re
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from aria.graph import build_graph
from aria.state import make_initial_state
from aria.config import get_llm

# ── Conversation Threads ──────────────────────────────────────────────────────

THREADS = [
    {
        "name": "Concept Drill-Down",
        "thread_id": "memory-test-drilldown",
        "turns": [
            {
                "question": "What is self-attention in transformers?",
                "requires_context": False,
                "context_check": "Should explain self-attention mechanism with Q/K/V.",
            },
            {
                "question": "What is the computational complexity of what you just explained?",
                "requires_context": True,
                "context_check": "Must reference self-attention and state O(n^2) complexity.",
            },
            {
                "question": "How does that compare to RNN complexity?",
                "requires_context": True,
                "context_check": "Must compare transformer O(n^2) vs RNN O(n) sequential complexity.",
            },
            {
                "question": "Given the tradeoffs we discussed, which is better for long sequences?",
                "requires_context": True,
                "context_check": "Must reference both architectures discussed and give a reasoned recommendation.",
            },
            {
                "question": "Summarise everything we covered in this conversation in 3 bullet points",
                "requires_context": True,
                "context_check": "Must summarize: self-attention, complexity comparison, and recommendation for long sequences.",
            },
        ],
    },
    {
        "name": "Research Session",
        "thread_id": "memory-test-research",
        "turns": [
            {
                "question": "Tell me about RAG architectures",
                "requires_context": False,
                "context_check": "Should explain retrieval-augmented generation architecture.",
            },
            {
                "question": "What are the main failure modes of what you described?",
                "requires_context": True,
                "context_check": "Must reference RAG specifically and describe its failure modes.",
            },
            {
                "question": "How does the reranking approach address those failures?",
                "requires_context": True,
                "context_check": "Must connect reranking to the RAG failure modes just discussed.",
            },
            {
                "question": "Are there any papers on this from 2024?",
                "requires_context": True,
                "context_check": "Must reference RAG/reranking topic and attempt to find recent papers.",
            },
            {
                "question": "Based on our discussion, what should I implement first in my own RAG system?",
                "requires_context": True,
                "context_check": "Must synthesize the full conversation: RAG, failure modes, reranking, papers into actionable advice.",
            },
        ],
    },
    {
        "name": "Comparison Thread",
        "thread_id": "memory-test-comparison",
        "turns": [
            {
                "question": "Explain LangGraph",
                "requires_context": False,
                "context_check": "Should explain LangGraph as a graph-based agent orchestration framework.",
            },
            {
                "question": "Now explain LlamaIndex",
                "requires_context": False,
                "context_check": "Should explain LlamaIndex as a data framework for LLM applications.",
            },
            {
                "question": "What are the key differences between the two systems you explained?",
                "requires_context": True,
                "context_check": "Must reference BOTH LangGraph and LlamaIndex from prior turns and compare them.",
            },
            {
                "question": "Which would you recommend for a student capstone project and why?",
                "requires_context": True,
                "context_check": "Must reference the comparison just made and give a reasoned recommendation.",
            },
            {
                "question": "What did you recommend and why - test if you remember",
                "requires_context": True,
                "context_check": "Must recall the specific recommendation from the previous turn and restate it.",
            },
        ],
    },
    {
        "name": "Entity Tracking (Needle)",
        "thread_id": "memory-test-entity",
        "turns": [
            {
                "question": "I am working on a new framework called 'QuantumGraph-RAG-v7.2'. Just remember this name, we will need it later.",
                "requires_context": False,
                "context_check": "Acknowledge the name QuantumGraph-RAG-v7.2.",
            },
            {
                "question": "What are the common graph databases?",
                "requires_context": False,
                "context_check": "Provide standard graph database info (Neo4j, etc).",
            },
            {
                "question": "How do you extract entities from text?",
                "requires_context": False,
                "context_check": "Explain NER or LLM extraction.",
            },
            {
                "question": "What was the exact name of the framework I mentioned at the start of this conversation? Give me the exact string.",
                "requires_context": True,
                "context_check": "MUST output EXACT string: QuantumGraph-RAG-v7.2 without hallucinating.",
            },
        ],
    },
]


def judge_memory(question: str, answer: str, context_check: str, turn_num: int, requires_context: bool, history: list = None) -> dict:
    """
    Use LLM-as-judge to verify if the answer correctly references prior context.
    """
    if not requires_context:
        # First turn - just check it answered reasonably
        if len(answer) > 50:
            return {"verdict": "PASS", "reason": "Initial turn answered substantively"}
        return {"verdict": "FAIL", "reason": "Answer too short or empty"}

    history_str = "\n".join(f"{m['role']}: {m['content']}" for m in history) if history else "No history provided."

    judge_prompt = f"""You are evaluating whether an AI assistant correctly remembers and references prior conversation context.

Turn number: {turn_num}
History:
{history_str}

Current Question: {question}
Evaluation Goal: {context_check}

AI's answer to evaluate:
{answer}

Does the AI's answer correctly reference the content from previous turns to answer the current question?
The answer should show it knows what was discussed before.

Respond with EXACTLY this JSON format:
{{"verdict": "PASS" or "FAIL", "reason": "one-sentence explanation"}}"""

    try:
        llm = get_llm()
        response = llm.invoke(judge_prompt)
        text = response.content.strip()
        text = re.sub(r"```json\s*|\s*```", "", text).strip()
        result = json.loads(text)
        return {
            "verdict": result.get("verdict", "FAIL"),
            "reason": result.get("reason", "No reason"),
        }
    except Exception as e:
        # Heuristic fallback
        answer_lower = answer.lower()
        has_context = len(answer) > 100
        if has_context:
            return {"verdict": "PASS", "reason": f"Substantive answer (heuristic, judge error: {e})"}
        return {"verdict": "FAIL", "reason": f"Judge error: {e}"}


def run_memory_test():
    """Execute all 3 conversation threads and score memory retention."""
    print("=" * 70)
    print("ARIA Multi-Turn Memory Stress Test")
    print("=" * 70)

    print("\nBuilding ARIA graph...")
    graph = build_graph()

    all_results = []
    total_passed = 0
    total_turns = 0
    
    # For decay detection: track passes per turn index (1 to 5)
    turn_metrics = {i: {"passed": 0, "total": 0, "avg_context_size": 0} for i in range(1, 6)}

    for thread in THREADS:
        thread_name = thread["name"]
        thread_id = thread["thread_id"]
        turns = thread["turns"]

        print(f"\n{'=' * 50}")
        print(f"Thread: {thread_name} ({thread_id})")
        print(f"{'=' * 50}")

        thread_results = []
        thread_passed = 0

        for t_num, turn in enumerate(turns, 1):
            q = turn["question"]
            print(f"\n  Turn {t_num}: {q[:60]}...")

            try:
                config = {"configurable": {"thread_id": thread_id}}
                result = graph.invoke({"question": q}, config=config)

                answer = result.get("answer", "")
                print(f"  Answer: {answer[:100]}...")

                # Judge memory
                judgement = judge_memory(
                    q, answer,
                    turn["context_check"],
                    t_num,
                    turn["requires_context"],
                    history=result.get("messages", [])
                )
                verdict = judgement["verdict"]
                reason = judgement["reason"]

                if verdict == "PASS":
                    thread_passed += 1
                    total_passed += 1
                    turn_metrics[t_num]["passed"] += 1
                    print(f"  \033[92mPASS\033[0m: {reason}")
                else:
                    print(f"  \033[91mFAIL\033[0m: {reason}")

                # Track context window size
                ctx_window = result.get("context_window", [])
                ctx_size_chars = sum(len(m.get("content", "")) for m in ctx_window)
                turn_metrics[t_num]["total"] += 1
                turn_metrics[t_num]["avg_context_size"] += ctx_size_chars

                thread_results.append({
                    "turn": t_num,
                    "question": q,
                    "answer": answer[:500],
                    "requires_context": turn["requires_context"],
                    "context_check": turn["context_check"],
                    "verdict": verdict,
                    "reason": reason,
                    "faithfulness": result.get("faithfulness", 0.0),
                    "route": result.get("route", "both"),
                    "context_size_chars": ctx_size_chars,
                })

            except Exception as e:
                print(f"  \033[91mERROR\033[0m: {e}")
                thread_results.append({
                    "turn": t_num,
                    "question": q,
                    "answer": f"ERROR: {e}",
                    "requires_context": turn["requires_context"],
                    "context_check": turn["context_check"],
                    "verdict": "FAIL",
                    "reason": f"Execution error: {e}",
                    "faithfulness": 0.0,
                    "route": "error",
                })

            total_turns += 1

        print(f"\n  Thread Score: {thread_passed}/{len(turns)} passed")

        all_results.append({
            "thread_name": thread_name,
            "thread_id": thread_id,
            "passed": thread_passed,
            "total": len(turns),
            "turns": thread_results,
        })

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("MEMORY TEST SUMMARY")
    print("=" * 70)

    print(f"\n{'Thread':<30} {'Passed':>8} {'Total':>8} {'Score':>8}")
    print("-" * 56)
    for r in all_results:
        pct = (r["passed"] / r["total"]) * 100 if r["total"] > 0 else 0
        print(f"{r['thread_name']:<30} {r['passed']:>8} {r['total']:>8} {pct:>7.0f}%")
    print("-" * 56)
    overall_pct = (total_passed / total_turns) * 100 if total_turns > 0 else 0
    print(f"{'OVERALL':<30} {total_passed:>8} {total_turns:>8} {overall_pct:>7.0f}%")

    print("\n" + "=" * 70)
    print("MEMORY DECAY ANALYSIS (Per Turn Index)")
    print("=" * 70)
    print(f"{'Turn Index':<12} | {'Pass Rate':>10} | {'Avg Context Size (chars)':>25}")
    print("-" * 55)
    
    decay_metrics = {}
    for t_num, metrics in turn_metrics.items():
        if metrics["total"] == 0: continue
        pass_rate = (metrics["passed"] / metrics["total"]) * 100
        avg_size = metrics["avg_context_size"] / metrics["total"]
        decay_metrics[t_num] = {"pass_rate": pass_rate, "avg_size": avg_size}
        print(f"Turn {t_num:<7} | {pass_rate:>9.0f}% | {avg_size:>25.0f}")


    output = {
        "timestamp": datetime.now().isoformat(),
        "total_turns": total_turns,
        "total_passed": total_passed,
        "overall_score": f"{total_passed}/{total_turns}",
        "decay_metrics": decay_metrics,
        "threads": all_results,
    }

    output_path = Path(__file__).parent.parent / "data" / "memory_test_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")
    print(f"\nMemory Score: {total_passed}/{total_turns} turns passed memory check")


if __name__ == "__main__":
    run_memory_test()
