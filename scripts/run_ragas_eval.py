#!/usr/bin/env python3
"""
scripts/run_ragas_eval.py
-------------------------
Runs a RAGAS evaluation on the ARIA graph headlessly (no Streamlit).
Evaluates faithfulness, answer_relevancy, context_precision, context_recall
across 10 curated question-ground_truth pairs.

Usage:
    python scripts/run_ragas_eval.py
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from aria.graph import build_graph
from aria.state import make_initial_state

# ── Evaluation Dataset ────────────────────────────────────────────────────────

EVAL_DATASET = [
    # ── KB-only questions (3) ──────────────────────────────
    {
        "category": "KB",
        "question": "What is self-attention in transformer architectures and how does it work?",
        "ground_truth": "Self-attention is a mechanism that computes attention weights between all positions in a sequence simultaneously. Each token generates query, key, and value vectors. Attention scores are computed as the dot product of queries and keys, scaled by the square root of the dimension, then softmaxed and used to weight the values.",
    },
    {
        "category": "KB",
        "question": "Explain the RAG (Retrieval-Augmented Generation) architecture and its key components.",
        "ground_truth": "RAG combines a retriever component (typically a bi-encoder with a vector database like FAISS or ChromaDB) with a generator (LLM). The retriever finds relevant documents for a query, and these documents are inserted into the LLM's context to ground the generation in factual information, reducing hallucination.",
    },
    {
        "category": "KB",
        "question": "What is LangGraph and how does it differ from LangChain?",
        "ground_truth": "LangGraph is a library for building stateful, multi-actor applications with LLMs, built on top of LangChain. Unlike LangChain's sequential chains, LangGraph uses a graph-based architecture with nodes and conditional edges, supporting cycles and complex control flow like reflection loops.",
    },

    # ── Tool-required questions (3) ────────────────────────
    {
        "category": "Tool",
        "question": "What are the most recent arXiv papers on retrieval-augmented generation from 2024?",
        "ground_truth": "Recent 2024 papers on RAG include works on advanced RAG architectures, self-RAG, corrective RAG (CRAG), and RAG evaluation frameworks. These papers address limitations of naive RAG through query decomposition, iterative retrieval, and faithfulness verification.",
    },
    {
        "category": "Tool",
        "question": "What is the latest research on LLM reasoning and chain-of-thought prompting?",
        "ground_truth": "Recent research on LLM reasoning includes chain-of-thought prompting, tree-of-thought, self-consistency decoding, and reasoning via planning. Key advances include improved mathematical reasoning, code generation through structured reasoning, and multi-step problem decomposition.",
    },
    {
        "category": "Tool",
        "question": "What are the current state-of-the-art approaches to reducing LLM hallucination?",
        "ground_truth": "State-of-the-art hallucination reduction approaches include RAG with faithfulness scoring, self-reflection loops, RLHF fine-tuning, constrained decoding, and factual consistency checking. Recent work focuses on detecting and correcting hallucinations post-generation.",
    },

    # ── Multi-turn questions (2) ───────────────────────────
    {
        "category": "Multi-turn",
        "question": "Based on the transformer attention mechanism, how does multi-head attention improve upon single-head attention?",
        "ground_truth": "Multi-head attention runs several attention operations in parallel, each with different learned projections. This allows the model to jointly attend to information from different representation subspaces at different positions, capturing diverse types of relationships that a single head would miss.",
    },
    {
        "category": "Multi-turn",
        "question": "Given what we know about RAG, what are the main failure modes and how can reranking address them?",
        "ground_truth": "RAG failure modes include: retrieving irrelevant chunks (precision failure), missing relevant chunks (recall failure), and the LLM ignoring retrieved context. Cross-encoder reranking addresses precision failures by rescoring bi-encoder results with a more accurate model, promoting truly relevant chunks.",
    },

    # ── Red-team / edge cases (2) ──────────────────────────
    {
        "category": "Edge",
        "question": "What is the current stock price of Apple Inc?",
        "ground_truth": "This question is out of scope for a research assistant focused on AI/ML topics. The system should indicate it cannot provide real-time financial data.",
    },
    {
        "category": "Edge",
        "question": "Since GPT-4 has 1.7 trillion parameters as confirmed by OpenAI, how does this affect inference costs?",
        "ground_truth": "The premise is false - OpenAI has not officially confirmed GPT-4's parameter count. The 1.7 trillion figure is an unconfirmed rumor. The system should flag this uncertainty rather than treating the premise as fact.",
    },
]


def run_evaluation():
    """Run the full ARIA pipeline on each question and collect results."""
    print("=" * 70)
    print("ARIA RAGAS Evaluation")
    print("=" * 70)

    print("\nBuilding ARIA graph...")
    graph = build_graph()

    results = []
    thread_id = "ragas-eval-001"

    for i, item in enumerate(EVAL_DATASET):
        q = item["question"]
        gt = item["ground_truth"]
        print(f"\n[{i+1}/{len(EVAL_DATASET)}] {q[:70]}...")

        try:
            initial = make_initial_state(q, thread_id)
            config = {"configurable": {"thread_id": thread_id}}
            result = graph.invoke(initial, config=config)

            answer = result.get("answer", "")
            retrieved = result.get("retrieved", "")
            contexts = [c.strip() for c in retrieved.split("\n\n---\n\n") if c.strip()]
            faith = result.get("faithfulness", 0.0)
            route = result.get("route", "both")

            entry = {
                "category": item.get("category", "Uncategorized"),
                "question": q,
                "answer": answer,
                "contexts": contexts[:5],
                "ground_truth": gt,
                "faithfulness_internal": faith,
                "route": route,
            }
            results.append(entry)
            print(f"  Route: {route} | Faithfulness: {faith:.2f} | Answer: {answer[:80]}...")

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "category": item.get("category", "Uncategorized"),
                "question": q,
                "answer": f"ERROR: {e}",
                "contexts": [],
                "ground_truth": gt,
                "faithfulness_internal": 0.0,
                "route": "error",
            })

    # ── Compute RAGAS metrics ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Computing RAGAS Metrics...")
    print("=" * 70)

    ragas_scores = compute_ragas_metrics(results)

    # ── Print results table ───────────────────────────────────────────────────
    print(f"\n{'#':<3} {'Question':<50} {'Faith':>7} {'Relev':>7} {'Prec':>7} {'Recall':>7}")
    print("-" * 85)

    for i, (r, s) in enumerate(zip(results, ragas_scores)):
        q_short = r["question"][:48]
        print(f"{i+1:<3} {q_short:<50} {s['faithfulness']:>7.2f} {s['relevancy']:>7.2f} {s['precision']:>7.2f} {s['recall']:>7.2f}")

    # ── Averages ──────────────────────────────────────────────────────────────
    avg_faith = sum(s["faithfulness"] for s in ragas_scores) / len(ragas_scores)
    avg_relev = sum(s["relevancy"] for s in ragas_scores) / len(ragas_scores)
    avg_prec  = sum(s["precision"] for s in ragas_scores) / len(ragas_scores)
    avg_recall = sum(s["recall"] for s in ragas_scores) / len(ragas_scores)

    print("-" * 85)
    print(f"{'AVG':<53} {avg_faith:>7.2f} {avg_relev:>7.2f} {avg_prec:>7.2f} {avg_recall:>7.2f}")

    # ── Per-Category Averages ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Per-Category Averages Matrix")
    print("=" * 70)
    
    categories = list(set([r["category"] for r in results]))
    print(f"{'Category':<15} | {'Faith':>7} | {'Relev':>7} | {'Prec':>7} | {'Recall':>7}")
    print("-" * 60)
    
    cat_averages = {}
    for cat in categories:
        cat_scores = [s for r, s in zip(results, ragas_scores) if r["category"] == cat]
        if not cat_scores: continue
        c_faith = sum(s["faithfulness"] for s in cat_scores) / len(cat_scores)
        c_relev = sum(s["relevancy"] for s in cat_scores) / len(cat_scores)
        c_prec = sum(s["precision"] for s in cat_scores) / len(cat_scores)
        c_recall = sum(s["recall"] for s in cat_scores) / len(cat_scores)
        cat_averages[cat] = {
            "faithfulness": c_faith, "relevancy": c_relev, 
            "precision": c_prec, "recall": c_recall
        }
        print(f"{cat:<15} | {c_faith:>7.2f} | {c_relev:>7.2f} | {c_prec:>7.2f} | {c_recall:>7.2f}")

    # ── Save results and Regression Detection ──────────────────────────────────
    output_path = Path(__file__).parent.parent / "data" / "ragas_baseline.json"
    
    # Bootstrap confidence intervals
    print("\n" + "=" * 70)
    print("95% Confidence Intervals (Bootstrap n=1000)")
    print("=" * 70)
    
    ci_metrics = {}
    for m in ["faithfulness", "relevancy", "precision", "recall"]:
        metric_values = [s[m] for s in ragas_scores if s[m] is not None]
        if len(metric_values) > 1:
            means = [np.mean(np.random.choice(metric_values, size=len(metric_values), replace=True)) for _ in range(1000)]
            lower = np.percentile(means, 2.5)
            upper = np.percentile(means, 97.5)
            ci_metrics[m] = {"mean": np.mean(metric_values), "lower_95": lower, "upper_95": upper}
            print(f"{m:<18}: {ci_metrics[m]['mean']:>6.4f}  (95% CI: [{lower:>6.4f}, {upper:>6.4f}])")
        else:
            ci_metrics[m] = {"mean": 0.0, "lower_95": 0.0, "upper_95": 0.0}
            
    # Check basics for regression
    if output_path.exists():
        try:
            with open(output_path, "r") as f:
                old_data = json.load(f)
            old_avg = old_data.get("averages", {})
            print("\n" + "=" * 70)
            print("Regression Analysis")
            print("=" * 70)
            for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
                o_val = old_avg.get(m, 0.0)
                n_val = {"faithfulness": avg_faith, "answer_relevancy": avg_relev, 
                         "context_precision": avg_prec, "context_recall": avg_recall}[m]
                diff = n_val - o_val
                sign = "+" if diff >= 0 else ""
                print(f"{m:<18}: {o_val:>6.4f} -> {n_val:>6.4f} ({sign}{diff:.4f})")
        except Exception as e:
            print(f"Regression parse error: {e}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output = {
        "timestamp": datetime.now().isoformat(),
        "total_questions": len(EVAL_DATASET),
        "averages": {
            "faithfulness": round(avg_faith, 4),
            "answer_relevancy": round(avg_relev, 4),
            "context_precision": round(avg_prec, 4),
            "context_recall": round(avg_recall, 4),
        },
        "confidence_intervals": {k: {"lower": round(v["lower_95"], 4), "upper": round(v["upper_95"], 4)} for k, v in ci_metrics.items()},
        "category_averages": cat_averages,
        "per_question": [
            {**r, **s} for r, s in zip(results, ragas_scores)
        ],
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    print(f"\nRAGAS Baseline - Faithfulness: {avg_faith:.2f} | Relevancy: {avg_relev:.2f} | Precision: {avg_prec:.2f} | Recall: {avg_recall:.2f}")


def compute_ragas_metrics(results):
    """
    Compute RAGAS-style metrics for each result.
    Uses the LLM to evaluate answer quality against ground truth and contexts.
    Falls back to heuristic scoring if ragas library is not available.
    """
    scores = []

    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset

        # Build HF dataset for RAGAS
        data = {
            "question": [r["question"] for r in results],
            "answer": [r["answer"] for r in results],
            "contexts": [r["contexts"] if r["contexts"] else ["No context retrieved."] for r in results],
            "ground_truth": [r["ground_truth"] for r in results],
        }
        dataset = Dataset.from_dict(data)

        eval_result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )

        df = eval_result.to_pandas()
        for _, row in df.iterrows():
            scores.append({
                "faithfulness": float(row.get("faithfulness", 0.0)),
                "relevancy": float(row.get("answer_relevancy", 0.0)),
                "precision": float(row.get("context_precision", 0.0)),
                "recall": float(row.get("context_recall", 0.0)),
            })
        return scores

    except Exception as e:
        print(f"\n[RAGAS] Library not available or failed ({e}). Using heuristic scoring...")

    # ── Heuristic fallback ────────────────────────────────────────────────────
    for r in results:
        answer = r["answer"].lower()
        gt = r["ground_truth"].lower()
        contexts = " ".join(r["contexts"]).lower()

        # Faithfulness: does the answer stick to context?
        faith = r.get("faithfulness_internal", 0.5)

        # Relevancy: word overlap between answer and question
        q_words = set(r["question"].lower().split())
        a_words = set(answer.split())
        relev = len(q_words & a_words) / max(len(q_words), 1)
        relev = min(relev * 2.5, 1.0)  # scale up

        # Precision: are retrieved contexts relevant?
        if contexts:
            ctx_q_overlap = len(q_words & set(contexts.split())) / max(len(q_words), 1)
            prec = min(ctx_q_overlap * 2.0, 1.0)
        else:
            prec = 0.0

        # Recall: does answer cover ground truth?
        gt_words = set(gt.split())
        recall_overlap = len(gt_words & a_words) / max(len(gt_words), 1)
        recall = min(recall_overlap * 3.0, 1.0)

        scores.append({
            "faithfulness": round(faith, 4),
            "relevancy": round(relev, 4),
            "precision": round(prec, 4),
            "recall": round(recall, 4),
        })

    return scores


if __name__ == "__main__":
    run_evaluation()
