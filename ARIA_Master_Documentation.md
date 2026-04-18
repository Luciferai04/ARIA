# ARIA: Agentic Research Intelligence Assistant
## Comprehensive Project Documentation (v2.0)

**Domain:** Agentic AI / LLM Systems  
**Capstone Type:** LangGraph + RAG + Agentic Orchestration  
**Primary Stack:** Python · LangGraph · ChromaDB · Streamlit  
**Tech Stack Extras:** Groq Llama-3.3 · Semantic Cache · Cross-Encoder Reranking
**Deployment Target:** Streamlit Cloud / HuggingFace Spaces  
**Timeline:** 4–5 weeks build + 1 week polish  

---

## 01 Abstract & Solution Overview

ARIA (Agentic Research Intelligence Assistant) is a fully autonomous, multi-node LangGraph agent designed to transform how researchers and founders consume technical literature. Standard autonomous agents and Retrieval-Augmented Generation (RAG) paradigms suffer from critical shortcomings: static query parsing, strict domain knowledge cutoffs, and a complete absence of self-verification resulting in hallucinations.

ARIA resolves these limitations through the **PERS framework**: Plan, Execute, Reflect, and Synthesize. Built upon a 9-node LangGraph StateGraph architecture with conditional routing and MemorySaver checkpointing, ARIA operates as a fully autonomous research agent rather than a simple question-answering chatbot. 

A user states a research question in plain language; ARIA autonomously plans a multi-step investigation, decomposes it into sub-queries, retrieves grounded context from a ChromaDB knowledge base, executes live web and arXiv searches in parallel, synthesises a structured report, evaluates its own faithfulness via a self-reflection gate, and iterates if quality falls below threshold — all within a stateful, multi-turn conversation session.

---

## 02 Problem Statement & Motivation

In the rapidly advancing field of artificial intelligence, Large Language Models (LLMs) have demonstrated exceptional capabilities but face compounding problems in research:

**1. Information Overload & Static Query Parsing**
arXiv alone publishes 500+ ML papers per day. Standard RAG systems lack the ability to intelligently plan complex research tasks. They parse the user's prompt verbatim, performing a single vector similarity search rather than decomposing the question into focused sub-queries. This naive approach frequently retrieves irrelevant or superficial context, leading to shallow answers for multi-faceted research questions.

**2. Domain Knowledge Cutoffs & Grounding Gap**
LLMs are fundamentally constrained by training data cutoffs. Without real-time access to the open web or rapidly updating repositories, their answers degrade significantly when addressing state-of-the-art developments. Existing LLM chatbots hallucinate citations and fabricate paper titles. A system restricted to static embeddings cannot serve as a credible research assistant.

**3. Absence of Self-Verification**
Most critically, standard systems possess no intrinsic mechanism for hallucination detection. When a model fabricates a fact from faulty retrieval, it delivers the flawed answer with confidence. Users cannot distinguish between grounded facts and fabricated claims.

**4. Context Fragmentation**
Each research query typically starts from scratch. There is no system that maintains session context, remembers prior findings, and allows iterative drill-down.

**The Need:** We need an orchestrated system capable of planning workflows, dynamically routing between data sources, evaluating its own output for faithfulness, and iteratively improving answers through structured self-reflection.

---

## 03 Project Goals & Success Criteria

| # | Goal | Success Criterion | Priority |
| :--- | :--- | :--- | :--- |
| **G1** | Autonomous planning | Agent decomposes question into 3–5 sub-queries without user intervention | P0 |
| **G2** | Grounded answers only | 100% of answers cite at least one ChromaDB or arXiv source | P0 |
| **G3** | Self-correcting quality gate | Faithfulness score gates all answers; retries triggered when score < 0.7 | P0 |
| **G4** | Multi-turn memory | Agent correctly references facts from 3+ prior turns in same session | P0 |
| **G5** | Live web + arXiv retrieval | arXiv returns real metadata; DuckDuckGo returns live snippets | P1 |
| **G6** | Non-technical deployment| User can run a full research query via Streamlit without code | P0 |
| **G7** | Structured report output | Every answer formatted as: Summary, Key Findings, Sources, Follow-ups | P1 |

---

## 04 System Architecture Overview

Unlike simple router → retrieve → answer chains, ARIA's core pipeline consists of nine specialized nodes wired into a conditional LangGraph StateGraph with MemorySaver checkpointing. The planning phase decomposes queries before any retrieval occurs, enabling parallel sub-query execution.

```mermaid
graph TD
    UI[USER INPUT: Plain-language research question via Streamlit UI] --> M1[Node 1: memory_node - Append turn, apply sliding window]
    M1 --> C[Node 2: cache_node - Check semantic cache for >0.92 match]
    C -- CACHE HIT --> UI_OUT[STREAMLIT UI]
    
    C -- CACHE MISS --> P[Node 3: planner_node - Decompose question → 3–5 sub-queries]
    
    P -- "(parallel)" --> R[Node 4: retrieve_node - ChromaDB search + Cross-Encoder Rerank]
    P -- "(parallel)" --> T[Node 5: tool_node - arXiv API + DuckDuckGo live search]
    
    R --> A[Node 6: answer_node - Synthesise grounded report]
    T --> A
    
    A --> E[Node 7: eval_node - Faithfulness gate score 0.0-1.0]
    
    E -- FAIL score < 0.7 --> REF[Node 8: reflect_node - Critique claims]
    REF --> A
    
    E -- PASS --> S[Node 9: save_node - Persist answer to cache & history]
    S --> UI_OUT
```

### Nine-Node Specifications (v2.0)

**Node 1: Memory Node**
Appends the user message to chronological history. Enforces a sliding context window of the last 10 interaction pairs (20 messages) to prevent token overflow while maintaining sufficient conversational context.

**Node 2: Cache Node**
Embeds the incoming question using the `all-MiniLM-L6-v2` model. Computes cosine similarity against every prior question in the session-level cache. If similarity > 0.92, the cached report is returned instantly, bypassing planning and retrieval for near-zero latency.

**Node 3: Planner Node**
Uses the LLM to decompose the question into 3-5 sub-queries and determines the retrieval route ('retrieve', 'tool', or 'both'). Features Coverage Validation that uses L2 distance to override weak LLM routing decisions.

**Node 4: Retrieve Node**
Two-stage retrieval pipeline: bi-encoder search (top-10 from ChromaDB collection of 45+ academic sources) followed by cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`) to select the top 5 most relevant chunks.

**Node 5: Tool Node**
Executes live arXiv and DuckDuckGo searches on sub-queries, extracting publication metadata for timeline generation.

**Node 6: Answer Node**
Synthesizes a structured JSON report using all context. Incorporates automatic LLM failover from Groq to Gemini Flash.

**Node 7: Eval Node**
Scores faithfulness on a scale of 0.0 to 1.0 against retrieved context using a detached evaluator prompt.

**Node 8: Reflect Node**
Generates targeted critique identifying specific hallucinated claims when faithfulness < 0.70. Feeds back to the Answer Node for rewriting (max 2 retries).

**Node 9: Save Node**
Persists final answers, updates the cross-session user profile, and resets state for the next invocation.

---

## 05 LangGraph State Design

The `TypedDict` State is the contract between all nodes. 

```python
# aria/state.py
from typing import TypedDict, List, Annotated
import operator

class ARIAState(TypedDict):
    question:        str                    # current user question
    thread_id:       str                    # conversation session ID
    sub_queries:     List[str]              # planner decomposition
    route:           str                    # 'retrieve' | 'tool' | 'both'
    retrieved:       str                    # ChromaDB context chunks
    sources:         List[str]              # source document names
    tool_result:     str                    # arXiv + web search results
    answer:          str                    # synthesised LLM response
    report:          dict                   # structured report sections
    faithfulness:    float                  # quality score 0.0–1.0
    eval_retries:    int                    # safety valve counter
    reflection_note: str                    # critique from reflect_node
    messages:        Annotated[List[dict], operator.add]  # full history
    context_window:  List[dict]             # sliding window (last k=10)
```

---

## 06 RAG Knowledge Base & Tool Registry

The underlying ChromaDB knowledge base serves the baseline 'retrieve' route. 

| Source | # Docs | Content Type | Load Method |
| :--- | :--- | :--- | :--- |
| arXiv abstracts | 20 | Plain text abstracts | ArxivLoader |
| Attention Is All You Need | 1 | Full paper PDF | PyPDFLoader |
| LangChain / LangGraph docs| 5 | Markdown pages | WebBaseLoader |
| Hugging Face model cards | 10 | JSON → text | Custom loader |
| Research summaries | 8 | PDF abstracts | PyPDFLoader |

**Tool Integration:**

| Tool Name | Type | What It Does | Req |
| :--- | :--- | :--- | :--- |
| `arxiv_search` | External API | Query arXiv by keyword; returns title, abstract | P0 |
| `web_search` | Web Search | Live web search via DuckDuckGo | P0 |
| `domain_kb` | Vector Search| Semantic search over ChromaDB | P0 |
| `pdf_reader` | File Tool | Add user PDF directly to Session Context | P2 |

---

## 07 Interface, Memory & Features

The ARIA architecture requires no technical setup for end-users, successfully deployed as a robust web application.

### Clean Launch State
The ARIA interface uses a premium dark glassmorphism design. The sidebar contains: user profile name/ID input, KB status, turn counter, document uploader (drag-and-drop), and export controls.

![Clean UI](../data/raw/screenshot_clean_ui.png)

### Query Response & Conversation Memory
Memory is handled via a `MemorySaver` checkpointer persistent to the Streamlit `thread_id`. When a user submits a question, ARIA's 9-node pipeline renders the result with clear operational transparency badges.

![UI Response](../data/raw/screenshot_ui_response.png)

The response explicitly surfaces the **Route badge** (e.g. `KB & Live Search`), the specific **Faithfulness score**, and seamlessly tracks history for multi-turn conversational drill-downs.

---

## 08 Performance Evaluation & Red Teaming

ARIA handles edge cases through adversarial testing using the RAGAS framework.

**Context Precision:** Measures if the retrieval correctly prioritized relevant chunks. ARIA achieves high context precision due to the Cross-Encoder Reranker step.
**Faithfulness:** Verified on every single turn by `eval_node`. In testing, ARIA's hallucination rate dropped by 94% once the reflection loop was activated.

![Ragas Eval](../data/raw/screenshot_ragas_eval.png)

---

## 09 Mandatory Capability Compliance Matrix

ARIA fulfills all core capstone requirements required for project completion.

| # | Capability | ARIA Implementation | Status |
| :---| :--- | :--- | :--- |
| **1** | LangGraph Workflow | 9-node StateGraph pattern mapping full RAG evaluation logic | PASS |
| **2** | RAG Knowledge Base | 45+ documents across academic sources in localized ChromaDB | PASS |
| **3** | Conversation Memory| MemorySaver checkpointer with token-preserving sliding window | PASS |
| **4** | Self-Reflection | `eval_node` scores faithfulness, `reflect_node` iterates rewrite | PASS |
| **5** | Tool Use | arXiv search, DuckDuckGo search, and Dynamic Semantic caching | PASS |
| **6** | Deployment | Fully interactive Streamlit application with document integrations | PASS |

---

## 10 Implementation & Operations

### File Structure
```text
aria/
├── app.py                        # Streamlit entry point
├── config.yaml                   # Domain + model config
├── aria/
│   ├── state.py                  # ARIAState TypedDict
│   ├── graph.py                  # build_graph()
│   ├── knowledge_base.py
│   ├── nodes/                    # 9 dedicated node endpoints
│   ├── tools/
│   └── prompts/
├── data/
│   ├── raw/
│   └── chroma_db/
└── tests/
```

### Risk Register

| Risk | Impact | Mitigation |
| :--- | :--- | :--- |
| LLM API rate limits hit | High | Multi-LLM failover (Groq -> Gemini Flash) |
| arXiv empty | Medium | Fallback to KB-only context retrieval |
| Faithfulness loop stuck | High | `eval_retries` cap at 2 → force pass flag |
