---
name: aria
description: >
  Build, scaffold, extend, fix, test, and deploy ARIA — Agentic Research Intelligence Assistant.
  A 7-node LangGraph StateGraph capstone project for KIIT University B.Tech CS.
  Use this skill whenever the user asks to build any part of ARIA: nodes, graph, KB,
  tools, Streamlit UI, tests, deployment, or any file under the aria/ structure.
  Also trigger for: "build planner_node", "write eval_node", "set up ChromaDB", 
  "implement faithfulness gate", "build the Streamlit app", "write the tool registry",
  "ingest knowledge base", "implement memory", "fix the reflection loop",
  "add a new tool", "deploy to Streamlit Cloud", or any task referencing ARIAState,
  StateGraph, MemorySaver, retrieve_node, answer_node, or the capstone rubric.
  ALWAYS use this skill — do not build ARIA from memory.
author: Soumyajit Ghosh
version: "1.0"
project: KIIT B.Tech CS Capstone · LangGraph Agentic AI · April 2026
---

# ARIA — Agentic Research Intelligence Assistant
## Claude Code SKILL.md — Complete Implementation Reference

---

## 0. NORTH STAR

ARIA is a **Plan-Execute-Reflect-Synthesise (PERS)** agentic system. Its defining
characteristic is a **planner_node** that decomposes user questions into sub-queries
BEFORE any retrieval occurs. This is the single biggest differentiator vs. other
student capstone projects. Never merge or skip the planner_node.

**Capstone 6 Mandatory Capabilities — all must be satisfied:**

| # | Capability | ARIA Implementation |
|---|---|---|
| 1 | LangGraph Workflow | 7-node StateGraph (see §4) |
| 2 | RAG Knowledge Base | ChromaDB, 45+ docs (see §6) |
| 3 | Conversation Memory | MemorySaver + thread_id, k=10 window (see §8) |
| 4 | Self-Reflection | eval_node + reflect_node faithfulness loop (see §7) |
| 5 | Tool Use | arXiv + DuckDuckGo + citation_formatter (see §9) |
| 6 | Deployment | Streamlit Cloud app (see §10) |

---

## 1. FILE STRUCTURE

```
aria/
├── app.py                        # Streamlit entry point
├── requirements.txt
├── config.yaml                   # Domain + model config
├── .env.example                  # GROQ_API_KEY / GOOGLE_API_KEY
│
├── aria/
│   ├── __init__.py
│   ├── state.py                  # ARIAState TypedDict ← DESIGN FIRST
│   ├── graph.py                  # build_graph() — StateGraph assembly
│   ├── knowledge_base.py         # ChromaDB ingest + query
│   │
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── memory_node.py
│   │   ├── planner_node.py
│   │   ├── retrieve_node.py
│   │   ├── tool_node.py
│   │   ├── answer_node.py
│   │   ├── eval_node.py
│   │   ├── reflect_node.py
│   │   └── save_node.py
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── arxiv_tool.py
│   │   ├── web_search_tool.py
│   │   └── citation_formatter.py
│   │
│   └── prompts/
│       ├── planner_prompt.py
│       ├── answer_prompt.py
│       ├── eval_prompt.py
│       └── reflect_prompt.py
│
├── data/
│   ├── raw/                      # Source PDFs, markdown files
│   └── chroma_db/                # Persistent ChromaDB — commit to repo
│
├── scripts/
│   └── ingest_kb.py              # One-time KB ingestion
│
└── tests/
    ├── test_state.py
    ├── test_nodes.py
    ├── test_tools.py
    └── test_graph.py
```

---

## 2. EXACT DEPENDENCIES

```txt
# requirements.txt — pin all versions to avoid Pydantic v1/v2 conflicts
langgraph>=0.2.0
langchain>=0.2.0
langchain-core>=0.2.0
langchain-community>=0.2.0
langchain-google-genai>=1.0.0   # or langchain-groq
chromadb>=0.5.0
sentence-transformers>=2.6.0
pydantic>=2.0.0
streamlit>=1.35.0
python-dotenv>=1.0.0
pyyaml>=6.0
pypdf>=4.0.0
pytest>=8.0.0
pytest-asyncio>=0.23.0
```

**LLM choice (use one):**
- Free option A: `google-generativeai` — Gemini 1.5 Flash (generous free tier)
- Free option B: `groq` — Llama-3.3-70B (fast, free, API key from console.groq.com)

**CRITICAL:** Never use `langchain==0.1.x` — it has breaking changes with LangGraph 0.2+.

---

## 3. STATE — DESIGN THIS FIRST

**Rule: Every field that any node reads OR writes MUST be in ARIAState.**

```python
# aria/state.py
from typing import TypedDict, List, Optional, Annotated
import operator

class ARIAState(TypedDict):
    # ── Input ────────────────────────────────────────────────
    question:        str               # current user question
    thread_id:       str               # session ID for MemorySaver

    # ── Planning ─────────────────────────────────────────────
    sub_queries:     List[str]         # planner decomposition (3–5 items)
    route:           str               # "retrieve" | "tool" | "both"

    # ── Retrieval ────────────────────────────────────────────
    retrieved:       str               # ChromaDB context chunks (concatenated)
    sources:         List[str]         # document names for citation

    # ── Tool ─────────────────────────────────────────────────
    tool_result:     str               # arXiv + web search results

    # ── Generation ───────────────────────────────────────────
    answer:          str               # current best answer
    report:          dict              # {summary, key_findings, sources, follow_ups}

    # ── Evaluation ───────────────────────────────────────────
    faithfulness:    float             # quality score 0.0–1.0
    eval_retries:    int               # loop guard — max 2
    reflection_note: str               # critique from reflect_node

    # ── Memory ───────────────────────────────────────────────
    messages:        Annotated[List[dict], operator.add]   # full history
    context_window:  List[dict]        # sliding window (last 10 pairs)
```

**Initialisation (for graph.invoke calls):**
```python
initial_state = {
    "question": user_input,
    "thread_id": session_id,
    "sub_queries": [],
    "route": "both",
    "retrieved": "",
    "sources": [],
    "tool_result": "",
    "answer": "",
    "report": {},
    "faithfulness": 0.0,
    "eval_retries": 0,
    "reflection_note": "",
    "messages": [],
    "context_window": [],
}
```

---

## 4. GRAPH ASSEMBLY

**Node execution order:**
```
memory_node → planner_node → [retrieve_node || tool_node] →
answer_node → eval_node → (reflect_node → answer_node)* → save_node
```

```python
# aria/graph.py
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from aria.state import ARIAState
from aria.nodes import (
    memory_node, planner_node, retrieve_node, tool_node,
    answer_node, eval_node, reflect_node, save_node
)

def route_retrieval(state: ARIAState) -> str:
    """After planner — determine which branches run."""
    if state["route"] == "retrieve":
        return "retrieve_only"
    elif state["route"] == "tool":
        return "tool_only"
    return "both"

def route_after_eval(state: ARIAState) -> str:
    """After eval — pass, retry, or force-pass."""
    if state["faithfulness"] >= 0.7:
        return "save_node"
    if state["eval_retries"] >= 2:
        return "save_node"   # force-pass with quality flag
    return "reflect_node"

def build_graph() -> StateGraph:
    builder = StateGraph(ARIAState)

    # Add all nodes
    builder.add_node("memory_node",   memory_node)
    builder.add_node("planner_node",  planner_node)
    builder.add_node("retrieve_node", retrieve_node)
    builder.add_node("tool_node",     tool_node)
    builder.add_node("answer_node",   answer_node)
    builder.add_node("eval_node",     eval_node)
    builder.add_node("reflect_node",  reflect_node)
    builder.add_node("save_node",     save_node)

    # Entry
    builder.set_entry_point("memory_node")
    builder.add_edge("memory_node", "planner_node")

    # Conditional routing after planner
    builder.add_conditional_edges("planner_node", route_retrieval, {
        "retrieve_only": "retrieve_node",
        "tool_only":     "tool_node",
        "both":          "retrieve_node",   # retrieve first, then tool
    })
    builder.add_edge("retrieve_node", "tool_node")
    builder.add_edge("tool_node",     "answer_node")

    # Reflection loop
    builder.add_edge("answer_node", "eval_node")
    builder.add_conditional_edges("eval_node", route_after_eval, {
        "save_node":    "save_node",
        "reflect_node": "reflect_node",
    })
    builder.add_edge("reflect_node", "answer_node")
    builder.add_edge("save_node", END)

    # Compile with MemorySaver
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)
```

**CRITICAL RULES:**
- Always use `builder.set_entry_point()` — not `add_edge(START, ...)`
- Always compile with `checkpointer=MemorySaver()` for memory to work
- Never use `llm.invoke()` directly in graph — always wrap in a node function
- The retrieve_node → tool_node edge means BOTH always run when route="both"

---

## 5. NODE IMPLEMENTATIONS

### memory_node
```python
# aria/nodes/memory_node.py
WINDOW_SIZE = 10

def memory_node(state: ARIAState) -> ARIAState:
    """Append turn to history, maintain sliding window."""
    state["messages"].append({"role": "user", "content": state["question"]})
    all_msgs = state["messages"]
    state["context_window"] = all_msgs[-(WINDOW_SIZE * 2):]
    return state
```

### planner_node
```python
# aria/nodes/planner_node.py
from pydantic import BaseModel
from langchain_core.output_parsers import JsonOutputParser

class PlanOutput(BaseModel):
    sub_queries: list[str]   # 3–5 items
    route: str               # "retrieve" | "tool" | "both"

PLANNER_PROMPT = """You are a research planning assistant.
Given the user's question and conversation history, decompose the question
into 3-5 focused sub-queries that together fully answer the original question.
Also decide the retrieval route:
- "retrieve" if the KB likely has this information
- "tool" if live/recent data is needed (papers published after 2023, news)
- "both" if unclear or the question needs both

Conversation history (last 5 turns):
{history}

User question: {question}

Respond ONLY with valid JSON matching this schema:
{{"sub_queries": ["...", "...", "..."], "route": "both"}}
"""

def planner_node(state: ARIAState) -> ARIAState:
    history = state["context_window"][-10:]
    history_str = "\n".join(f"{m['role']}: {m['content']}" for m in history)
    
    response = llm.invoke(PLANNER_PROMPT.format(
        history=history_str, question=state["question"]
    ))
    
    import json, re
    text = response.content
    # Strip markdown code fences if present
    text = re.sub(r'```json\s*|\s*```', '', text).strip()
    result = json.loads(text)
    
    state["sub_queries"] = result.get("sub_queries", [state["question"]])[:5]
    state["route"] = result.get("route", "both")
    return state
```

### retrieve_node
```python
# aria/nodes/retrieve_node.py
from aria.knowledge_base import query_kb

def retrieve_node(state: ARIAState) -> ARIAState:
    """Query ChromaDB for each sub-query, deduplicate, concat."""
    if state["route"] == "tool":
        state["retrieved"] = ""
        state["sources"] = []
        return state
    
    all_chunks = []
    all_sources = []
    seen = set()
    
    for query in state["sub_queries"]:
        results = query_kb(query, top_k=5)
        for doc, metadata in results:
            key = doc[:100]  # dedup by content prefix
            if key not in seen:
                seen.add(key)
                all_chunks.append(doc)
                all_sources.append(metadata.get("source", "Unknown"))
    
    state["retrieved"] = "\n\n---\n\n".join(all_chunks)
    state["sources"] = list(dict.fromkeys(all_sources))  # deduplicated
    return state
```

### tool_node
```python
# aria/nodes/tool_node.py
from aria.tools import arxiv_search, web_search

def tool_node(state: ARIAState) -> ARIAState:
    """Run arXiv + web search on sub-queries."""
    if state["route"] == "retrieve":
        state["tool_result"] = ""
        return state
    
    results = []
    for query in state["sub_queries"][:3]:  # limit API calls
        try:
            arxiv_result = arxiv_search.invoke({"query": query})
            results.append(f"[arXiv] {query}:\n{arxiv_result}")
        except Exception:
            pass
        try:
            web_result = web_search.invoke({"query": query})
            results.append(f"[Web] {query}:\n{web_result}")
        except Exception:
            pass
    
    state["tool_result"] = "\n\n".join(results)
    return state
```

### answer_node
```python
# aria/nodes/answer_node.py
import json, re

ANSWER_PROMPT = """You are a research synthesis assistant.
Answer the user's question using ONLY the provided context.
Do NOT add information not present in the context.
If the context doesn't contain enough information, say so explicitly.

IMPORTANT: If there is a reflection note, address its critique in your answer.

User question: {question}
Reflection note (if any): {reflection_note}

Retrieved KB context:
{retrieved}

Live search results:
{tool_result}

Conversation history:
{history}

Respond ONLY with valid JSON:
{{
  "summary": "2-3 sentence direct answer",
  "key_findings": ["finding 1", "finding 2", "finding 3"],
  "sources": ["source1", "source2"],
  "follow_ups": ["follow-up question 1", "follow-up question 2"]
}}
"""

def answer_node(state: ARIAState) -> ARIAState:
    history_str = "\n".join(
        f"{m['role']}: {m['content']}" for m in state["context_window"][-6:]
    )
    context = (state["retrieved"] or "")[:4000]
    tool_ctx = (state["tool_result"] or "")[:2000]
    
    response = llm.invoke(ANSWER_PROMPT.format(
        question=state["question"],
        reflection_note=state.get("reflection_note", ""),
        retrieved=context,
        tool_result=tool_ctx,
        history=history_str,
    ))
    
    text = response.content
    text = re.sub(r'```json\s*|\s*```', '', text).strip()
    
    try:
        report = json.loads(text)
    except json.JSONDecodeError:
        report = {
            "summary": text[:500],
            "key_findings": [],
            "sources": state["sources"],
            "follow_ups": []
        }
    
    state["report"] = report
    state["answer"] = report.get("summary", text)
    return state
```

### eval_node
```python
# aria/nodes/eval_node.py
import json, re

EVAL_PROMPT = """You are a faithfulness evaluator.
Score how well the ANSWER is grounded in the CONTEXT on a scale of 0.0 to 1.0.

Scoring guide:
- 1.0 = every claim directly supported by context
- 0.7 = most claims supported, minor additions acceptable
- 0.5 = significant unsupported claims
- 0.0 = answer is hallucinated / unrelated to context

ANSWER: {answer}
CONTEXT: {context}

Respond ONLY with JSON: {{"score": <float 0.0-1.0>, "issues": ["issue1", "issue2"]}}
"""

def eval_node(state: ARIAState) -> ARIAState:
    context = (state["retrieved"] + "\n" + state["tool_result"])[:3000]
    
    response = llm.invoke(EVAL_PROMPT.format(
        answer=state["answer"], context=context
    ))
    
    text = re.sub(r'```json\s*|\s*```', '', response.content).strip()
    try:
        result = json.loads(text)
        score = float(result.get("score", 0.5))
    except (json.JSONDecodeError, ValueError):
        score = 0.5
    
    state["faithfulness"] = max(0.0, min(1.0, score))
    state["eval_retries"] = state.get("eval_retries", 0) + 1
    return state
```

### reflect_node
```python
# aria/nodes/reflect_node.py

REFLECT_PROMPT = """The answer below has a faithfulness score of {score:.2f} (threshold: 0.70).
Identify the specific claims that are NOT supported by the context and provide
a concise critique that will guide a better, grounded revision.

Answer: {answer}
Context (excerpt): {context}

Respond with a 2-3 sentence critique starting with "CRITIQUE:"
"""

def reflect_node(state: ARIAState) -> ARIAState:
    context = (state["retrieved"] + "\n" + state["tool_result"])[:2000]
    
    response = llm.invoke(REFLECT_PROMPT.format(
        score=state["faithfulness"],
        answer=state["answer"],
        context=context,
    ))
    
    state["reflection_note"] = response.content
    return state
```

### save_node
```python
# aria/nodes/save_node.py

def save_node(state: ARIAState) -> ARIAState:
    """Persist final answer to message history."""
    answer_msg = {
        "role": "assistant",
        "content": state["answer"],
        "report": state["report"],
        "sources": state["sources"],
        "faithfulness": state["faithfulness"],
    }
    state["messages"].append(answer_msg)
    state["context_window"] = state["messages"][-(10 * 2):]
    # Reset per-turn fields for next invocation
    state["reflection_note"] = ""
    state["eval_retries"] = 0
    return state
```

---

## 6. KNOWLEDGE BASE

```python
# aria/knowledge_base.py
from langchain_community.document_loaders import (
    PyPDFLoader, WebBaseLoader, ArxivLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

CHROMA_PATH = "./data/chroma_db"
COLLECTION  = "aria_kb"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )

def query_kb(query: str, top_k: int = 5) -> list[tuple[str, dict]]:
    vs = get_vectorstore()
    results = vs.similarity_search_with_score(query, k=top_k)
    return [(doc.page_content, doc.metadata) for doc, score in results if score < 1.5]
```

**KB Ingestion script:**
```python
# scripts/ingest_kb.py
SOURCES = [
    # arXiv papers
    {"type": "arxiv", "query": "LangGraph agentic AI 2024", "max_docs": 5},
    {"type": "arxiv", "query": "retrieval augmented generation RAG 2024", "max_docs": 5},
    {"type": "arxiv", "query": "reinforcement learning from human feedback", "max_docs": 5},
    {"type": "arxiv", "query": "large language model reasoning chain of thought", "max_docs": 5},
    # Web pages
    {"type": "web", "url": "https://langchain-ai.github.io/langgraph/concepts/"},
    {"type": "web", "url": "https://python.langchain.com/docs/concepts/rag/"},
    # PDFs in data/raw/
    {"type": "pdf", "path": "./data/raw/"},
]
# Run this ONCE: python scripts/ingest_kb.py
# Commit data/chroma_db/ to repo for Streamlit Cloud
```

**Chunking parameters (do not change):**
- `chunk_size=512` — optimal for all-MiniLM-L6-v2
- `chunk_overlap=64` — preserves boundary context
- `separators=["\n\n", "\n", ". ", " ", ""]`

---

## 7. FAITHFULNESS LOOP

```
eval_node scores answer
      │
      ├── score >= 0.70  ──────────────────────► save_node  ✓
      │
      └── score < 0.70
              │
              ├── eval_retries < 2  ──► reflect_node ──► answer_node ──► eval_node
              │
              └── eval_retries >= 2  ─► save_node (flagged)  ⚠
```

**In Streamlit UI**, display a badge:
- `score >= 0.7` → 🟢 High Confidence
- `0.5 <= score < 0.7` → 🟡 Moderate Confidence
- `score < 0.5` → 🔴 Low Confidence (flagged for review)

---

## 8. MEMORY — INVOCATION PATTERN

```python
# Every graph.invoke call MUST pass config with thread_id
config = {"configurable": {"thread_id": st.session_state.thread_id}}

result = graph.invoke(initial_state, config=config)
```

**MemorySaver auto-persists state between calls with the same thread_id.**
You do NOT manually pass prior messages — LangGraph handles checkpointing.

**Sliding window:** `context_window` keeps last `WINDOW_SIZE * 2 = 20` entries.
This is what's passed to all LLM prompts — NOT the full `messages` list.

---

## 9. TOOLS

```python
# aria/tools/arxiv_tool.py
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper
from langchain_core.tools import tool

@tool
def arxiv_search(query: str) -> str:
    """Search arXiv for recent academic papers. Returns titles, authors, abstracts."""
    wrapper = ArxivAPIWrapper(top_k_results=5, doc_content_chars_max=800)
    return wrapper.run(query)

# aria/tools/web_search_tool.py
from langchain_community.tools import DuckDuckGoSearchRun

@tool
def web_search(query: str) -> str:
    """Search the web for current information about a topic."""
    search = DuckDuckGoSearchRun()
    return search.run(query)

# aria/tools/citation_formatter.py
from langchain_core.tools import tool

@tool
def citation_formatter(sources: list[str]) -> str:
    """Format a list of source strings into a numbered citation list."""
    return "\n".join(f"[{i+1}] {s}" for i, s in enumerate(sources))
```

---

## 10. STREAMLIT APP

```python
# app.py
import streamlit as st
import uuid
from aria.graph import build_graph
from aria.state import ARIAState

st.set_page_config(page_title="ARIA — Research Intelligence", layout="wide")

# ── Session init ─────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()
if "history" not in st.session_state:
    st.session_state.history = []

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 ARIA")
    st.caption("Agentic Research Intelligence Assistant")
    if st.button("New Session"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.history = []
        st.rerun()
    st.divider()
    st.caption(f"Session: {st.session_state.thread_id[:8]}...")

# ── Chat history ─────────────────────────────────────────────
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and "report" in msg:
            render_report(msg["report"], msg.get("faithfulness", 0.0), msg.get("sources", []))
        else:
            st.write(msg["content"])

# ── Input ────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a research question..."):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("ARIA is researching..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            result = st.session_state.graph.invoke(
                {
                    "question": prompt,
                    "thread_id": st.session_state.thread_id,
                    "messages": [], "eval_retries": 0,
                    "sub_queries": [], "route": "both",
                    "retrieved": "", "sources": [],
                    "tool_result": "", "answer": "",
                    "report": {}, "faithfulness": 0.0,
                    "reflection_note": "", "context_window": [],
                },
                config=config
            )
        render_report(result["report"], result["faithfulness"], result["sources"])
        st.session_state.history.append({
            "role": "assistant",
            "content": result["answer"],
            "report": result["report"],
            "faithfulness": result["faithfulness"],
            "sources": result["sources"],
        })

def render_report(report: dict, faith: float, sources: list):
    """Render structured research report in Streamlit."""
    # Faithfulness badge
    if faith >= 0.7:
        st.success(f"🟢 Faithfulness: {faith:.2f}")
    elif faith >= 0.5:
        st.warning(f"🟡 Faithfulness: {faith:.2f}")
    else:
        st.error(f"🔴 Faithfulness: {faith:.2f} — review recommended")
    
    # Summary
    if report.get("summary"):
        st.markdown(f"**Summary:** {report['summary']}")
    
    # Key findings
    if report.get("key_findings"):
        with st.expander("📋 Key Findings", expanded=True):
            for i, finding in enumerate(report["key_findings"], 1):
                st.markdown(f"{i}. {finding}")
    
    # Sources
    if sources:
        with st.expander("📚 Sources"):
            for i, src in enumerate(sources, 1):
                st.caption(f"[{i}] {src}")
    
    # Follow-up questions
    if report.get("follow_ups"):
        with st.expander("💡 Follow-up Questions"):
            for q in report["follow_ups"]:
                st.markdown(f"- {q}")
```

---

## 11. DEPLOYMENT

**Streamlit Cloud deployment:**

1. Push all code + committed `data/chroma_db/` to GitHub
2. Go to share.streamlit.io → New app → select repo → entry: `app.py`
3. Add secrets in Streamlit Cloud dashboard:
   ```toml
   # .streamlit/secrets.toml
   GROQ_API_KEY = "gsk_..."
   # OR
   GOOGLE_API_KEY = "AIza..."
   ```
4. Required files for deployment:
   - `requirements.txt` (all deps pinned)
   - `app.py` at root
   - `data/chroma_db/` committed (skip runtime ingestion in prod)

**config.yaml:**
```yaml
# config.yaml
llm:
  provider: groq              # groq | google
  model: llama-3.3-70b-versatile  # or gemini-1.5-flash
  temperature: 0.1
  max_tokens: 2048

retrieval:
  top_k: 5
  chunk_size: 512
  chunk_overlap: 64
  embed_model: sentence-transformers/all-MiniLM-L6-v2

memory:
  window_size: 10

eval:
  faithfulness_threshold: 0.7
  max_retries: 2

kb:
  domain: ai_ml_research       # ai_ml_research | finance | biotech
  persist_path: ./data/chroma_db
  collection: aria_kb
```

---

## 12. TESTS

```python
# tests/test_nodes.py
import pytest
from aria.state import ARIAState
from aria.nodes.memory_node import memory_node
from aria.nodes.eval_node import eval_node

def make_state(**kwargs) -> ARIAState:
    defaults = {
        "question": "test question", "thread_id": "test-001",
        "sub_queries": [], "route": "both",
        "retrieved": "", "sources": [], "tool_result": "",
        "answer": "", "report": {}, "faithfulness": 0.0,
        "eval_retries": 0, "reflection_note": "",
        "messages": [], "context_window": [],
    }
    defaults.update(kwargs)
    return defaults

def test_memory_node_appends():
    state = make_state(question="What is RAG?")
    result = memory_node(state)
    assert len(result["messages"]) == 1
    assert result["messages"][0]["content"] == "What is RAG?"

def test_eval_routing_pass():
    state = make_state(faithfulness=0.85, eval_retries=0)
    # After eval_node sets faithfulness >= 0.7, route_after_eval should return save_node
    from aria.graph import route_after_eval
    assert route_after_eval(state) == "save_node"

def test_eval_routing_retry():
    state = make_state(faithfulness=0.4, eval_retries=0)
    from aria.graph import route_after_eval
    assert route_after_eval(state) == "reflect_node"

def test_eval_routing_force_pass():
    state = make_state(faithfulness=0.3, eval_retries=2)
    from aria.graph import route_after_eval
    assert route_after_eval(state) == "save_node"
```

---

## 13. COMMON ERRORS & FIXES

| Error | Cause | Fix |
|---|---|---|
| `AttributeError: 'str' is not a valid ParagraphStyle parent` | LangGraph edge not added | Check `builder.add_edge()` calls |
| `KeyError: 'messages'` on first invoke | Missing field in initial_state | Pass full initial state dict |
| `pydantic.ValidationError` in planner_node | LLM returned malformed JSON | Add `re.sub(r'```json...```', ...)` strip |
| ChromaDB `InvalidCollectionException` | Collection not initialised | Run `scripts/ingest_kb.py` first |
| `MemorySaver not persisting` | Missing `config` on invoke | Always pass `config={"configurable":{"thread_id": ...}}` |
| DuckDuckGo rate limit | Too many rapid queries | Add `time.sleep(1)` between tool calls |
| `eval_retries` keeps incrementing | Not reset in save_node | Set `state["eval_retries"] = 0` in save_node |
| Streamlit secrets missing | Not set in cloud dashboard | Use `st.secrets["GROQ_API_KEY"]` in app.py |

---

## 14. CAPSTONE SUBMISSION CHECKLIST

Before submitting, verify every item:

**Mandatory capabilities:**
- [ ] StateGraph has 7 nodes (not fewer than 3)
- [ ] No standalone `llm.invoke()` outside of a node function
- [ ] ARIAState TypedDict has all 15 fields defined
- [ ] ChromaDB has 45+ documents ingested and committed
- [ ] Every answer references at least one ChromaDB source
- [ ] MemorySaver used with thread_id
- [ ] Tested multi-turn: 5-turn conversation works correctly
- [ ] eval_node runs on every answer
- [ ] reflect_node triggers when faithfulness < 0.7
- [ ] Reflection loop has max_retries=2 guard
- [ ] At least 3 tools registered: arxiv_search, web_search, citation_formatter
- [ ] Streamlit app deployed and accessible via public URL
- [ ] Non-technical user can run full query without touching code

**Code quality:**
- [ ] All requirements pinned in requirements.txt
- [ ] .env.example committed (not .env with real keys)
- [ ] README.md with: demo GIF, architecture diagram, usage instructions
- [ ] At least 5 unit tests passing
- [ ] config.yaml for domain/model configuration

**Demo readiness:**
- [ ] App loads in < 10 seconds on cold start
- [ ] 3 demo queries prepared with expected outputs
- [ ] Faithfulness badge visible in UI
- [ ] Sources displayed for every answer
- [ ] Follow-up questions displayed

---

## 15. DEMO QUERIES (for evaluation day)

Prepare these 3 demo queries — they showcase all 6 capabilities:

1. **"What are the latest advances in LangGraph and agentic AI architectures?"**
   - Triggers: both route (KB + arXiv tool)
   - Shows: planner decomposition, RAG retrieval, arXiv live search, structured report

2. **"Based on what we discussed, how does retrieval augmented generation compare to fine-tuning?"** (ask as turn 2)
   - Triggers: multi-turn memory
   - Shows: context_window correctly references prior turn

3. **"Explain transformer attention mechanisms in 5 key points"**
   - Triggers: retrieve-only route (well-covered in KB)
   - Shows: faithfulness gate (should score > 0.8), clean report format

---

*Last updated: April 2026 | KIIT University B.Tech CS Capstone*
*Author: Soumyajit Ghosh | GitHub: Luciferai04*
