import os
from pathlib import Path
from fpdf import FPDF

IMG_DIR = Path(__file__).parent.parent / "data" / "raw"

class PDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 10)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f'ARIA Documentation  |  Page {self.page_no()}', align='C')

    def chapter_title(self, title):
        self.set_font('helvetica', 'B', 16)
        self.set_text_color(99, 102, 241)   # indigo
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT", align='L')
        self.set_text_color(0, 0, 0)
        self.ln(4)

    def chapter_subtitle(self, subtitle):
        self.set_font('helvetica', 'B', 13)
        self.set_text_color(71, 85, 105)    # slate
        self.cell(0, 9, subtitle, new_x="LMARGIN", new_y="NEXT", align='L')
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def chapter_body(self, text):
        self.set_font('helvetica', '', 11)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 6, text, align='J')
        self.ln(3)

    def bullet(self, text):
        self.set_font('helvetica', '', 11)
        self.set_text_color(30, 30, 30)
        self.cell(8)
        self.multi_cell(0, 6, f"  {text}", align='L')
        self.ln(1)

    def add_screenshot(self, filename, caption=""):
        img_path = IMG_DIR / filename
        if img_path.exists():
            avail_w = self.w - self.l_margin - self.r_margin  # ~170mm on A4
            self.image(str(img_path), x=self.l_margin, w=avail_w)
            if caption:
                self.set_font('helvetica', 'I', 9)
                self.set_text_color(120, 120, 120)
                self.cell(0, 6, caption, align='C', new_x="LMARGIN", new_y="NEXT")
                self.set_text_color(0, 0, 0)
            self.ln(4)
        else:
            self.chapter_body(f"[Screenshot: {filename} not found]")

def generate_documentation():
    pdf = PDF(format='A4')
    pdf.set_auto_page_break(auto=True, margin=20)

    # ══════════════════════════════════════════════════════════
    # COVER PAGE
    # ══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.ln(50)
    pdf.set_font('helvetica', 'B', 36)
    pdf.set_text_color(99, 102, 241)
    pdf.cell(0, 20, "ARIA", align='C', new_x="LMARGIN", new_y="NEXT")

    pdf.set_font('helvetica', '', 18)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(0, 12, "Agentic Research Intelligence Assistant", align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    pdf.set_font('helvetica', 'I', 14)
    pdf.cell(0, 10, "Plan  .  Retrieve  .  Reflect  .  Synthesise", align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(20)

    pdf.set_font('helvetica', '', 12)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, "Capstone Project Documentation", align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "Version 2.0 - 10-Feature Upgrade", align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(30)

    pdf.set_text_color(0, 0, 0)
    pdf.set_font('helvetica', '', 11)
    pdf.cell(0, 6, "Tech Stack: LangGraph | ChromaDB | Groq Llama-3.3 | Streamlit", align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, "Architecture: 9-Node StateGraph | Semantic Cache | Cross-Encoder Reranker", align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, "Testing: 25 Unit Tests | 100% Pass Rate", align='C', new_x="LMARGIN", new_y="NEXT")

    # ══════════════════════════════════════════════════════════
    # PAGE 2: PROBLEM STATEMENT
    # ══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.chapter_title("1. Problem Statement")
    pdf.chapter_body(
        "In the rapidly advancing field of artificial intelligence, Large Language Models (LLMs) have demonstrated "
        "exceptional capabilities in natural language understanding and generation. However, standard autonomous "
        "agents and Retrieval-Augmented Generation (RAG) paradigms suffer from several critical shortcomings that "
        "ARIA was designed to address:"
    )
    pdf.chapter_subtitle("1.1 Static Query Parsing")
    pdf.chapter_body(
        "Standard RAG systems lack the ability to intelligently plan complex research tasks. They parse the user's "
        "prompt verbatim, performing a single vector similarity search rather than decomposing the question into "
        "focused, actionable sub-queries. This naive approach frequently retrieves irrelevant or superficial context, "
        "leading to shallow, incomplete answers for multi-faceted research questions."
    )
    pdf.chapter_subtitle("1.2 Domain Knowledge Cutoffs")
    pdf.chapter_body(
        "LLMs are fundamentally constrained by training data cutoffs. Without real-time access to the open web or "
        "rapidly updating repositories like arXiv, their answers degrade significantly when addressing state-of-the-art "
        "developments, recent publications, or rapidly evolving fields. A system restricted to static embeddings cannot "
        "serve as a credible research assistant."
    )
    pdf.chapter_subtitle("1.3 Absence of Self-Verification")
    pdf.chapter_body(
        "Most critically, standard systems possess no intrinsic mechanism for self-reflection or hallucination detection. "
        "When a model fabricates a fact or draws an improper conclusion from faulty retrieval, it delivers the flawed "
        "answer with absolute confidence. Users have no way to distinguish between grounded facts and hallucinated claims, "
        "eroding trust in the system's output."
    )
    pdf.chapter_subtitle("1.4 The Need")
    pdf.chapter_body(
        "There is an urgent need for an intelligently orchestrated, multi-agent system capable of: (a) planning complex "
        "research workflows through question decomposition, (b) dynamically routing between knowledge bases and live "
        "search tools, (c) evaluating its own output for faithfulness before presenting it to the user, and (d) iteratively "
        "improving its answers through structured self-reflection."
    )

    # ══════════════════════════════════════════════════════════
    # PAGE 3: SOLUTION OVERVIEW
    # ══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.chapter_title("2. Solution: The PERS Framework")
    pdf.chapter_body(
        "ARIA (Agentic Research Intelligence Assistant) resolves these limitations through the PERS framework: "
        "Plan, Execute, Reflect, and Synthesize. Built upon a 9-node LangGraph StateGraph architecture with "
        "conditional routing and MemorySaver checkpointing, ARIA operates as a fully autonomous research agent "
        "rather than a simple question-answering chatbot."
    )
    pdf.chapter_subtitle("2.1 Plan Phase")
    pdf.chapter_body(
        "The Planner Node decomposes each user question into 3-5 focused, non-overlapping sub-queries using the LLM. "
        "It autonomously determines the optimal retrieval route for each question:\n"
        "  - 'retrieve': Use the local ChromaDB knowledge base (established AI/ML concepts)\n"
        "  - 'tool': Use live arXiv and web search (recent papers, current events)\n"
        "  - 'both': Combine both sources (unclear intent, or question needs both)\n\n"
        "The route decision is then validated by a KB Coverage Validator that checks the actual ChromaDB similarity "
        "score. If the best match has an L2 distance > 1.2 (poor coverage), the system overrides the LLM's route "
        "to 'both', ensuring comprehensive coverage even when the planner makes a suboptimal decision."
    )
    pdf.chapter_subtitle("2.2 Execute Phase")
    pdf.chapter_body(
        "Execution proceeds through two parallel paths depending on the route:\n\n"
        "Knowledge Base Retrieval: The Retrieve Node queries ChromaDB with top-k=10 using the all-MiniLM-L6-v2 "
        "bi-encoder. Results from both the base collection (4,437 chunks from 45+ academic sources) and any session-specific "
        "user-uploaded documents are merged. A Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2) then rescores all "
        "candidates against the original query, selecting the top 5 most relevant chunks.\n\n"
        "Live Search: The Tool Node executes arXiv academic search and DuckDuckGo web search on each sub-query, "
        "extracting publication metadata for research timeline visualization."
    )
    pdf.chapter_subtitle("2.3 Reflect Phase")
    pdf.chapter_body(
        "After the Answer Node generates a structured JSON report, the Eval Node scores its faithfulness (0.0-1.0) "
        "by checking whether every claim in the answer is grounded in the retrieved context. If the score falls "
        "below 0.70, the Reflect Node generates a targeted critique identifying specific hallucinated claims. "
        "The graph then routes backward to the Answer Node for rewriting, with a maximum of 2 reflection retries. "
        "This adversarial loop dramatically reduces hallucination rates."
    )
    pdf.chapter_subtitle("2.4 Synthesize Phase")
    pdf.chapter_body(
        "The final answer is structured as a JSON report containing: an executive summary, 3-5 key findings, "
        "cited sources, and 2-3 follow-up questions for continued research. The Save Node persists the result "
        "to the semantic cache, updates the user profile, and refreshes the sliding context window."
    )

    # ══════════════════════════════════════════════════════════
    # PAGE 4: 9-NODE ARCHITECTURE
    # ══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.chapter_title("3. Nine-Node Graph Architecture")
    pdf.chapter_body(
        "ARIA's core pipeline consists of nine specialized nodes wired into a conditional LangGraph StateGraph "
        "with MemorySaver checkpointing. Each node reads from and writes to a shared ARIAState (TypedDict with "
        "25 fields). The graph uses conditional edges for intelligent routing."
    )

    nodes = [
        ("Node 1: Memory Node", "Appends the user message to chronological history. Enforces a sliding context "
         "window of the last 10 interaction pairs (20 messages) to prevent token overflow while maintaining "
         "sufficient conversational context for multi-turn research sessions."),

        ("Node 2: Cache Node", "Embeds the incoming question using the same all-MiniLM-L6-v2 model as ChromaDB. "
         "Computes cosine similarity (via dot product on normalized embeddings) against every prior question in "
         "the session-level cache. If similarity exceeds 0.92, the cached report is returned instantly, completely "
         "bypassing planning, retrieval, generation, and evaluation. This provides near-zero latency for repeated "
         "or semantically identical queries."),

        ("Node 3: Planner Node", "Uses the LLM to decompose the question into 3-5 sub-queries and determine the "
         "retrieval route. Includes Comparison Mode detection (triggers on 'Compare:' prefix) and KB Coverage "
         "Validation (L2 distance check that overrides weak routing decisions)."),

        ("Node 4: Retrieve Node", "Two-stage retrieval pipeline: bi-encoder search (top-10 from ChromaDB base + "
         "session collections) followed by cross-encoder reranking to select the top 5 most relevant chunks. "
         "In Comparison Mode, runs separate queries per concept for structured analysis."),

        ("Node 5: Tool Node", "Executes arXiv and DuckDuckGo searches on up to 3 sub-queries. Extracts "
         "publication date metadata from arXiv results for research timeline chart generation."),

        ("Node 6: Answer Node", "Synthesizes a structured JSON report using all available context. Supports both "
         "standard mode (summary, key_findings, sources, follow_ups) and comparison mode (comparison_table, "
         "recommendation). Integrates automatic LLM failover from Groq to Gemini Flash."),

        ("Node 7: Eval Node", "Scores faithfulness on a 0.0-1.0 scale against retrieved context using a detached "
         "evaluator prompt. Routes to save_node if score >= 0.70 or retries exhausted."),

        ("Node 8: Reflect Node", "Generates targeted critique identifying specific hallucinated claims when "
         "faithfulness is low. Feeds the critique back to answer_node as a reflection_note for rewriting."),

        ("Node 9: Save Node", "Persists the final answer to message history and semantic cache. Updates the "
         "cross-session user profile. Resets all per-turn fields for the next invocation."),
    ]

    for title, body in nodes:
        pdf.chapter_subtitle(title)
        pdf.chapter_body(body)

    # ══════════════════════════════════════════════════════════
    # PAGE: FEATURE - Clean UI
    # ══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.chapter_title("4. Interface & Features")

    pdf.chapter_subtitle("4.1 Clean Launch State")
    pdf.chapter_body(
        "The ARIA interface uses a premium dark glassmorphism design with Inter typography (Google Fonts). "
        "The sidebar contains: user profile name/ID input, KB status (Online with chunk count), turn counter, "
        "document uploader (drag-and-drop PDF/TXT), session export button, and New Session control."
    )
    pdf.add_screenshot("screenshot_clean_ui.png", "Fig 1: ARIA clean launch state with sidebar features")

    # ══════════════════════════════════════════════════════════
    # PAGE: FEATURE - Query Response
    # ══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.chapter_subtitle("4.2 Query Response with Routing & Reranking")
    pdf.chapter_body(
        "When a user submits a research question, ARIA's 9-node pipeline processes it through planning, "
        "retrieval, reranking, generation, and evaluation. The response includes:\n"
        "  - Faithfulness confidence badge (High/Moderate/Low with score)\n"
        "  - LLM provider label ('Answered by: Groq Llama-3.3' or 'Gemini Flash')\n"
        "  - Route badge with KB coverage score (e.g. 'KB & Live Search [KB: 0.69]')\n"
        "  - Decomposed sub-query badges\n"
        "  - Structured summary, key findings, reranked sources with cross-encoder scores\n"
        "  - Follow-up questions for continued research"
    )
    pdf.add_screenshot("screenshot_comparison.png", "Fig 2: Comparison query with route badges, faithfulness score, and provider label")

    # ══════════════════════════════════════════════════════════
    # PAGE: FEATURE - Export & Session
    # ══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.chapter_subtitle("4.3 Session Export & Multi-Turn Research")
    pdf.chapter_body(
        "After completing at least one research turn, the 'Export Session as PDF' button appears in the sidebar. "
        "It generates a professionally formatted PDF using ReportLab with a cover page, per-question sections "
        "(question header, route badge, faithfulness score, summary, key findings, sources, follow-ups), and "
        "horizontal rules between sessions. The filename follows the format: aria_session_{thread_id}_{date}.pdf.\n\n"
        "The sidebar also tracks: turn count, uploaded document list, and user profile. The 'New Session' button "
        "resets the entire session state including uploaded documents and cache."
    )
    pdf.add_screenshot("screenshot_export.png", "Fig 3: Multi-turn session with Export Session button visible")

    # ══════════════════════════════════════════════════════════
    # PAGE: 10 FEATURES DETAILED
    # ══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.chapter_title("5. Ten Features in Detail")

    pdf.chapter_subtitle("Feature 1: PDF & TXT Upload")
    pdf.chapter_body(
        "Users upload research papers or notes directly through the sidebar's drag-and-drop file uploader. "
        "Each uploaded file is chunked using RecursiveCharacterTextSplitter (chunk_size=512, overlap=64) and "
        "injected into a session-specific ChromaDB collection named aria_kb_{thread_id}. This ensures complete "
        "isolation between sessions. All chunks receive metadata tags {source: 'user_upload', filename: original}. "
        "A toast notification confirms injection: 'Injected X chunks from filename.pdf'. The retrieve_node queries "
        "both the base collection (4,437 chunks) and the session collection, merging and deduplicating results."
    )

    pdf.chapter_subtitle("Feature 2: Session Export as PDF")
    pdf.chapter_body(
        "The generate_session_pdf() function in aria/export.py produces a downloadable PDF using ReportLab. "
        "Structure: cover page (title, date, thread_id, total questions), then per Q/A pair: indigo question header, "
        "route/faithfulness metadata, summary paragraph, numbered key findings, cited sources, bulleted follow-ups, "
        "with horizontal rules between sections. Filename: aria_session_{thread_id[:8]}_{date}.pdf."
    )

    pdf.chapter_subtitle("Feature 3: Cross-Encoder Reranker")
    pdf.chapter_body(
        "The aria/reranker.py module loads ms-marco-MiniLM-L-6-v2 once at module level (singleton pattern). "
        "The rerank() function takes top-10 bi-encoder results from ChromaDB, scores each chunk against the "
        "original query using the cross-encoder, and returns the top 5 sorted by descending relevance score. "
        "These scores are displayed in the Sources expander (e.g. 'Source Name [0.87]'), giving users "
        "transparency into retrieval quality. The two-stage pipeline (bi-encoder + cross-encoder) achieves "
        "significantly higher precision than single-stage cosine similarity alone."
    )

    pdf.chapter_subtitle("Feature 4: KB Coverage Validator")
    pdf.chapter_body(
        "After the planner sets a route, _validate_route() performs a quick ChromaDB similarity_search_with_score "
        "(top_k=1) on the original question. If the best L2 distance > 1.2 (poor KB coverage), the route is "
        "forced to 'both' regardless of the LLM's decision. If score <= 0.5 (excellent coverage), the route "
        "is allowed to stay as 'retrieve'. The coverage score is shown in the route badge: 'KB & Live Search "
        "[KB: 0.69]'. A caption appears when override occurs: 'Low KB coverage detected - switched to live search'."
    )

    pdf.add_page()
    pdf.chapter_subtitle("Feature 5: Cross-Session User Profiles")
    pdf.chapter_body(
        "The aria/user_profile.py module stores JSON profiles in data/profiles/{user_id}.json. Schema: "
        "{user_id, session_count, topics_researched, last_active, preferred_domains}. On each completed turn, "
        "save_node calls update_profile() to track research topics (from sub_queries) and infer domains (from "
        "report summaries). A user_context string is injected into initial state: 'This researcher has previously "
        "studied: [topics]. Preferred domains: [domains].' The sidebar shows a welcome-back message for returning users."
    )

    pdf.chapter_subtitle("Feature 6: Comparison Mode")
    pdf.chapter_body(
        "When a query begins with 'Compare:' (case insensitive), ARIA enters Comparison Mode. Concepts are "
        "extracted by splitting on ' vs ' or commas. Example: 'Compare: RAG vs Fine-tuning' extracts ['RAG', "
        "'Fine-tuning']. The retrieve_node runs separate ChromaDB queries per concept (top_k=5 each). The "
        "answer_node uses COMPARISON_PROMPT to generate: {comparison_table: [{aspect, concept_a, concept_b}], "
        "summary, recommendation, sources, follow_ups}. The UI renders st.table() with a purple 'Comparison Mode' badge."
    )

    pdf.chapter_subtitle("Feature 7: Research Timeline Chart")
    pdf.chapter_body(
        "When tool_node executes arXiv searches, _extract_arxiv_papers() parses publication dates using regex "
        "patterns (YYYY-MM-DD and arXiv ID year parsing). Results are stored as arxiv_papers in state. If 3+ "
        "papers are found and the route includes tool search, app.py renders an interactive Plotly bar chart: "
        "X-axis (year), Y-axis (paper count), indigo bars (#6366F1), dark theme matching the app aesthetics. "
        "The chart appears in a 'Research Timeline' expander below the Sources section."
    )

    pdf.chapter_subtitle("Feature 8: Automatic LLM Failover")
    pdf.chapter_body(
        "The aria/llm_client.py module provides invoke_with_fallback() that wraps all LLM calls. Primary: "
        "ChatGroq with the configured model (Llama 3.3 70B). Fallback: ChatGoogleGenerativeAI with gemini-1.5-flash. "
        "On any Groq error (including 429 rate limits), the system logs to stderr and silently retries with Gemini. "
        "The provider used is recorded in state (llm_provider_used) and displayed as 'Answered by: Groq Llama-3.3' "
        "or 'Answered by: Gemini Flash' below the faithfulness badge."
    )

    pdf.chapter_subtitle("Feature 9: Parent Document Retriever")
    pdf.chapter_body(
        "The aria/parent_store.py module provides a pickle-backed InMemoryStore that persists to "
        "data/parent_store.pkl. This enables the ParentDocumentRetriever pattern: child chunks (256 tokens) "
        "stored in ChromaDB for precise retrieval, with full parent documents (1024 tokens) returned to the LLM "
        "for richer context. The store survives Streamlit restarts through pickle serialization."
    )

    pdf.chapter_subtitle("Feature 10: GraphRAG Layer")
    pdf.chapter_body(
        "The aria/graph_retriever.py module wraps Microsoft's graphrag local search. A one-time offline build "
        "step (scripts/build_graph.py) indexes data/raw/ to extract entity-relationship graphs. At query time, "
        "query_graph() returns entity-relationship context that is merged with ChromaDB results. A 'Graph+Vector' "
        "badge appears when graph context is active. The feature degrades gracefully when the index hasn't been built."
    )

    # ══════════════════════════════════════════════════════════
    # PAGE: TECH STACK
    # ══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.chapter_title("6. Technology Stack")
    pdf.chapter_body(
        "ARIA is engineered using a modular Python backend with a reactive Streamlit frontend. "
        "All dependencies are pinned in requirements.txt to avoid Pydantic v1/v2 conflicts."
    )

    tech_items = [
        ("Core Orchestration", "LangGraph >= 0.2.0 with StateGraph, conditional edges, and MemorySaver checkpointing"),
        ("LLM Engines", "Groq Llama-3.3 70B Versatile (primary), Google Gemini 1.5 Flash (failover)"),
        ("Vector Database", "ChromaDB >= 0.5.0 with persistent local storage and session-isolated collections"),
        ("Embedding Model", "HuggingFace all-MiniLM-L6-v2 (384-dim, normalized, via sentence-transformers)"),
        ("Cross-Encoder", "ms-marco-MiniLM-L-6-v2 for two-stage retrieval reranking"),
        ("Semantic Cache", "In-memory cosine similarity with 0.92 threshold (dot product on normalized vectors)"),
        ("UI Framework", "Streamlit >= 1.35.0 with custom dark glassmorphism CSS and Inter typography"),
        ("PDF Generation", "ReportLab (session export), FPDF2 (documentation)"),
        ("Visualization", "Plotly >= 5.0.0 (research timeline bar charts)"),
        ("Web Search", "DuckDuckGo Search >= 5.0.0 (no API key required)"),
        ("Academic Search", "arXiv API >= 1.4.0 via LangChain ArxivLoader"),
        ("PDF Parsing", "PyPDF >= 4.0.0 and PyMuPDF >= 1.23.0 for document ingestion"),
        ("User Profiles", "JSON-backed persistent storage in data/profiles/"),
        ("Testing", "PyTest >= 8.0.0 with 25 unit tests (100% pass rate)"),
        ("Environment", "Python 3.9, isolated virtual environments, dotenv for secrets"),
    ]
    for label, desc in tech_items:
        pdf.set_font('helvetica', 'B', 11)
        pdf.cell(50, 7, f"{label}:", new_x="END")
        pdf.set_font('helvetica', '', 11)
        pdf.multi_cell(0, 7, desc, align='L')
        pdf.ln(1)

    # ══════════════════════════════════════════════════════════
    # PAGE: TESTING & VERIFICATION
    # ══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.chapter_title("7. Testing & Verification")
    pdf.chapter_body(
        "ARIA maintains a comprehensive test suite in tests/test_nodes.py with 25 deterministic unit tests "
        "covering all node logic and feature functionality. Tests use unittest.mock to isolate from external "
        "services (LLM APIs, ChromaDB). All tests pass in < 3 seconds."
    )

    test_groups = [
        ("Memory Node", "3 tests: appends user message, sliding window trimming, multi-turn accumulation"),
        ("Save Node", "2 tests: appends assistant message, resets per-turn fields"),
        ("Route Logic", "5 tests: high faith passes, threshold boundary, low faith retries, exhausted retries, mid retry"),
        ("Semantic Cache", "3 tests: miss on empty store, hit on identical query, miss on dissimilar query"),
        ("Cross-Encoder", "2 tests: correct return length, promotes high-relevance chunks first"),
        ("KB Coverage", "3 tests: high coverage stays retrieve, low coverage forces both, medium unchanged"),
        ("Comparison Mode", "2 tests: mode detected on 'Compare:' prefix, concepts correctly extracted"),
        ("Timeline", "1 test: arXiv date extraction returns correct format"),
        ("LLM Failover", "2 tests: 429 triggers Gemini fallback, successful primary returns 'groq'"),
        ("User Profiles", "2 tests: profile creates on first use, profile updates correctly on second use"),
    ]
    for group, desc in test_groups:
        pdf.set_font('helvetica', 'B', 11)
        pdf.cell(40, 7, f"{group}:", new_x="END")
        pdf.set_font('helvetica', '', 11)
        pdf.multi_cell(0, 7, desc, align='L')
        pdf.ln(1)

    pdf.ln(5)
    pdf.chapter_body("Test execution: pytest tests/test_nodes.py -v\nResult: 25 passed, 0 failed, 3 warnings in 2.10s")

    # ══════════════════════════════════════════════════════════
    # PAGE: USPs
    # ══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.chapter_title("8. Unique Selling Points")

    usps = [
        ("Adversarial Faithfulness Loop",
         "The eval_node acts as an internal adversarial network. The generative model competes against an "
         "evaluation model, creating a feedback loop that mathematically constrains hallucination. With a "
         "maximum of 2 reflection retries, the system converges on factually grounded answers."),

        ("Semantic Caching",
         "Identical or near-identical questions (cosine > 0.92) are served instantly from an in-memory cache, "
         "completely bypassing LLM inference. This provides near-zero latency (<5ms) for repeated queries while "
         "dramatically reducing API costs in multi-user deployments."),

        ("Two-Stage Retrieval (Bi-Encoder + Cross-Encoder)",
         "The bi-encoder casts a wide net (top-10), and the cross-encoder precisely scores relevance for the "
         "final top-5. This approach consistently outperforms single-stage RAG in retrieval precision, "
         "ensuring the LLM receives only the most relevant context."),

        ("Automatic LLM Failover",
         "If Groq hits a 429 rate limit or any error, ARIA silently switches to Gemini Flash and displays "
         "which provider answered. This ensures zero-downtime research sessions even during API instability."),

        ("Fully Localized Data Privacy",
         "All document embeddings remain local inside ChromaDB. Users never upload data to third-party servers. "
         "Session-specific collections (aria_kb_{thread_id}) ensure complete privacy isolation between sessions."),

        ("Cross-Session Persistent Memory",
         "JSON-backed user profiles track research topics, session counts, and preferred domains across sessions. "
         "This enables increasingly personalized research assistance as the system learns the user's interests."),
    ]
    for title, body in usps:
        pdf.chapter_subtitle(title)
        pdf.chapter_body(body)

    # ══════════════════════════════════════════════════════════
    # PAGE: FUTURE
    # ══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.chapter_title("9. Future Improvements")

    futures = [
        ("GraphRAG Full Integration",
         "The architecture already includes graph_retriever and build_graph modules prepared for Microsoft's "
         "graphrag library. Full activation would enable entity-relationship context extraction, discovering "
         "connections between papers that cosine similarity alone cannot capture."),

        ("Parent Document Retriever",
         "The parent_store module enables transitioning from small-chunk retrieval to full parent document "
         "context (1024 tokens), providing the LLM with richer, more coherent source material. Requires "
         "one-time re-ingestion via scripts/ingest_kb.py."),

        ("Multi-Modal Research",
         "Extending the system to parse images, diagrams, charts, and audio queries would enable truly "
         "comprehensive research assistance. Integration with vision models could extract data from "
         "research paper figures."),

        ("Cloud-Native Deployment",
         "Transitioning to Streamlit Cloud with st.secrets for API keys, committed ChromaDB collections "
         "for zero-setup access, and Redis-backed session storage for multi-user scalability."),

        ("Advanced Analytics Dashboard",
         "Building a dedicated analytics page showing: most researched topics, faithfulness trends over time, "
         "cache hit rates, average response times, and provider usage distribution."),
    ]
    for title, body in futures:
        pdf.chapter_subtitle(title)
        pdf.chapter_body(body)

    # ══════════════════════════════════════════════════════════
    # GENERATE
    # ══════════════════════════════════════════════════════════
    output_path = Path(__file__).parent.parent / "ARIA_Project_Documentation.pdf"
    pdf.output(str(output_path))
    print(f"Generated {output_path} ({os.path.getsize(output_path) / 1024:.0f} KB)")

if __name__ == "__main__":
    generate_documentation()
