import uuid
import os
import tempfile
import streamlit as st
import json
from datetime import datetime
from aria.graph import build_graph
from aria.state import make_initial_state
from aria.knowledge_base import get_collection_count

st.set_page_config(
    page_title="ARIA",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Clean Minimal Dark Theme */
.stApp {
    background: #0f111a;
    color: #e2e8f0;
}
[data-testid="stSidebar"] {
    background: #1a1d2d;
    border-right: 1px solid #2d3748;
}
[data-testid="stChatMessage"] {
    background: #1a1d2d;
    border: 1px solid #2d3748;
    border-radius: 8px;
    margin-bottom: 12px;
    padding: 12px;
}
[data-testid="stChatInput"] textarea {
    background: #1a1d2d !important;
    border: 1px solid #4a5568 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}
.route-badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-bottom: 8px;
    margin-right: 6px;
}
.route-both { background:#4c1d95; color:#ddd6fe; }
.route-retrieve { background:#065f46; color:#a7f3d0; }
.route-tool { background:#1e3a8a; color:#bfdbfe; }
.route-cache { background:#b45309; color:#fde68a; }
.route-upload { background:#7c2d12; color:#fed7aa; }
.route-comparison { background:#6b21a8; color:#e9d5ff; }
.route-graph { background:#0e7490; color:#a5f3fc; }
.subquery-badge {
    display: inline-block;
    background: #374151;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.75rem;
    margin-right: 6px;
    margin-bottom: 6px;
    color: #d1d5db;
}
.provider-label {
    font-size: 0.7rem;
    color: #64748b;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "graph" not in st.session_state:
    with st.spinner("Initialising ARIA graph ..."):
        st.session_state.graph = build_graph()
if "history" not in st.session_state:
    st.session_state.history = []
if "uploaded_files_list" not in st.session_state:
    st.session_state.uploaded_files_list = []
if "user_id" not in st.session_state:
    st.session_state.user_id = "researcher"


# ── PDF/TXT Upload Ingestion ─────────────────────────────────────────────────

def ingest_uploaded_file(uploaded_file, thread_id: str) -> int:
    """Chunk and inject an uploaded file into the session-specific ChromaDB collection."""
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, chunk_overlap=64,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    filename = uploaded_file.name
    texts = []
    metadatas = []

    if filename.lower().endswith(".pdf"):
        # Use PyPDFLoader via temp file
        from langchain_community.document_loaders import PyPDFLoader
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            chunks = splitter.split_documents(docs)
            for chunk in chunks:
                texts.append(chunk.page_content)
                metadatas.append({"source": "user_upload", "filename": filename})
        finally:
            os.unlink(tmp_path)
    elif filename.lower().endswith(".txt"):
        content = uploaded_file.read().decode("utf-8", errors="ignore")
        split_texts = splitter.split_text(content)
        for t in split_texts:
            texts.append(t)
            metadatas.append({"source": "user_upload", "filename": filename})
    else:
        return 0

    if not texts:
        return 0

    # Inject into session-specific collection
    try:
        from aria.knowledge_base import get_vectorstore
        from pathlib import Path

        try:
            from langchain_chroma import Chroma
        except ImportError:
            from langchain_community.vectorstores import Chroma

        chroma_path = str(Path(__file__).parent / "data" / "chroma_db")
        vs = get_vectorstore()
        emb_fn = getattr(vs, "embeddings", None) or getattr(vs, "_embedding_function", None)

        collection_name = f"aria_kb_{thread_id[:8]}"
        session_vs = Chroma(
            collection_name=collection_name,
            embedding_function=emb_fn,
            persist_directory=chroma_path,
        )
        session_vs.add_texts(texts=texts, metadatas=metadatas)
        return len(texts)
    except Exception as e:
        print(f"[Upload] Ingestion error: {e}")
        return 0


# ── Render Report ─────────────────────────────────────────────────────────────

def render_report(report, faith: float, sources: list,
                  sub_queries: list = None, route: str = "both",
                  reranker_scores: list = None, kb_coverage: float = 0.0,
                  comparison_mode: bool = False, arxiv_papers: list = None,
                  llm_provider: str = "groq", graph_context: str = ""):
    import re as _re
    if not isinstance(report, dict):
        try:
            report = json.loads(report)
        except Exception:
            raw_str = str(report)
            raw_str = _re.sub(r'^[\s\{\"\']*(summary|key_findings)[\"\'\\:\s]*', '', raw_str, flags=_re.IGNORECASE)
            report = {"summary": raw_str.replace('}', '').strip()}

    # Clean nested summary if LLM double-jsonifies it
    if report.get("summary") and isinstance(report["summary"], str) and report["summary"].strip().startswith("{"):
        try:
            clean_summary = json.loads(report["summary"])
            if "summary" in clean_summary:
                report["summary"] = clean_summary["summary"]
        except Exception:
            raw_str = report["summary"]
            raw_str = _re.sub(r'^[\s\{\"\']*(summary|key_findings)[\"\'\\:\s]*', '', raw_str, flags=_re.IGNORECASE)
            report["summary"] = raw_str.replace('}', '').strip()

    # ── Multi-Axis Quality Badge ─────────────────────────────
    eval_scores = result.get("eval_scores", {})
    if eval_scores:
        weighted = eval_scores.get("weighted", faith)
        if weighted >= 0.75:
            st.success(f"Quality: {weighted:.2f} -- High Confidence")
        elif weighted >= 0.50:
            st.warning(f"Quality: {weighted:.2f} -- Moderate Confidence")
        else:
            st.error(f"Quality: {weighted:.2f} -- Low Confidence")

        # Show per-axis breakdown in an expander
        with st.expander("Eval Breakdown (4-Axis)", expanded=False):
            cols = st.columns(4)
            axes = [
                ("Faith", eval_scores.get("faithfulness", 0)),
                ("Relev", eval_scores.get("relevance", 0)),
                ("Compl", eval_scores.get("completeness", 0)),
                ("Safety", eval_scores.get("safety", 0)),
            ]
            for col, (name, score) in zip(cols, axes):
                col.metric(name, f"{score:.2f}")

            issues = result.get("eval_issues", [])
            if issues:
                st.caption("Issues: " + " | ".join(issues[:3]))
    else:
        # Fallback for backward compatibility
        if faith >= 0.65:
            st.success(f"Faithfulness: {faith:.2f} -- High Confidence")
        elif faith >= 0.5:
            st.warning(f"Faithfulness: {faith:.2f} -- Moderate Confidence")
        else:
            st.error(f"Faithfulness: {faith:.2f} -- Low Confidence")

    # ── Provider label ───────────────────────────────────────
    provider_label = {"groq": "Groq Llama-3.3", "gemini": "Gemini Flash"}.get(llm_provider, llm_provider)
    st.markdown(f'<div class="provider-label">Answered by: {provider_label}</div>', unsafe_allow_html=True)

    # ── Route badges ─────────────────────────────────────────
    badges_html = ""
    route_labels = {
        "retrieve": "Knowledge Base Only",
        "tool": "Live Search",
        "both": "KB & Live Search",
        "cache": "Cache Hit",
    }

    if comparison_mode:
        badges_html += '<span class="route-badge route-comparison">Comparison Mode</span>'
    elif route == "cache":
        badges_html += '<span class="route-badge route-cache">Cache Hit</span>'
    else:
        label = route_labels.get(route, route)
        if kb_coverage > 0:
            label += f" [KB: {kb_coverage:.2f}]"
        css_class = f"route-{route}"
        badges_html += f'<span class="route-badge {css_class}">{label}</span>'

    if graph_context:
        badges_html += '<span class="route-badge route-graph">Graph+Vector</span>'

    # Check for user uploads in sources
    if sources and any("user_upload" in str(s) for s in sources):
        badges_html += '<span class="route-badge route-upload">User Upload</span>'

    if badges_html:
        st.markdown(badges_html, unsafe_allow_html=True)

    # ── KB coverage caption ──────────────────────────────────
    if kb_coverage > 1.2 and route == "both":
        st.caption("Low KB coverage detected — switched to live search")

    # ── Sub-query badges ─────────────────────────────────────
    if sub_queries:
        sq_badges = "".join(f'<span class="subquery-badge">{q}</span>' for q in sub_queries)
        st.markdown(f"<div>{sq_badges}</div>", unsafe_allow_html=True)
    st.markdown("")

    # ── Comparison table ─────────────────────────────────────
    if comparison_mode and report.get("comparison_table"):
        st.markdown("**Comparison Table**")
        import pandas as pd
        table_data = report["comparison_table"]
        if table_data:
            df = pd.DataFrame(table_data)
            st.table(df)
        if report.get("recommendation"):
            st.info(f"Recommendation: {report['recommendation']}")

    # ── Summary ──────────────────────────────────────────────
    if report.get("summary"):
        st.markdown(f"**Summary**\n\n{report['summary']}")

    # ── Key Findings ─────────────────────────────────────────
    if report.get("key_findings"):
        with st.expander("Key Findings", expanded=True):
            for i, finding in enumerate(report["key_findings"], 1):
                st.markdown(f"{i}. {finding}")

    # ── Sources with reranker scores ─────────────────────────
    if sources:
        with st.expander("Sources (Reranked)"):
            for i, src in enumerate(sources, 1):
                score_str = ""
                if reranker_scores and i - 1 < len(reranker_scores):
                    score_str = f" [{reranker_scores[i-1]:.2f}]"
                st.caption(f"[{i}] {src}{score_str}")

    # ── Follow-up Questions ──────────────────────────────────
    if report.get("follow_ups"):
        with st.expander("Follow-up Questions"):
            for q in report.get("follow_ups", []):
                st.markdown(f"- {q}")

    # ── Research Timeline Chart ──────────────────────────────
    if arxiv_papers and len(arxiv_papers) >= 3 and route != "retrieve":
        try:
            import plotly.express as px
            import pandas as pd

            years = []
            for p in arxiv_papers:
                date_str = p.get("date", "")
                if date_str and len(date_str) >= 4:
                    years.append(date_str[:4])

            if len(years) >= 3:
                df = pd.DataFrame({"year": years})
                counts = df["year"].value_counts().sort_index().reset_index()
                counts.columns = ["Year", "Papers"]

                fig = px.bar(counts, x="Year", y="Papers",
                             title="Research Activity Over Time",
                             color_discrete_sequence=["#6366F1"])
                fig.update_layout(
                    paper_bgcolor="#1a1d2d",
                    plot_bgcolor="#1a1d2d",
                    font_color="#e2e8f0",
                    title_font_size=14
                )
                with st.expander("Research Timeline"):
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            print(f"[Timeline] Chart skipped: {e}")


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ARIA")
    st.caption("Agentic Research Intelligence Assistant")
    st.markdown("---")

    # ── User Profile ─────────────────────────────────────────
    user_id = st.text_input("Your Name / ID", value=st.session_state.user_id, key="user_id_input")
    st.session_state.user_id = user_id

    try:
        from aria.user_profile import load_profile
        profile = load_profile(user_id)
        if profile.get("session_count", 0) > 1:
            st.caption(f"Welcome back! You've done {profile['session_count']} research sessions.")
    except Exception:
        pass

    st.markdown("---")

    # ── KB Status ────────────────────────────────────────────
    try:
        doc_count = get_collection_count()
        kb_status = f"Online [{doc_count} chunks]" if doc_count > 0 else "Offline"
    except Exception:
        kb_status = "Offline"

    st.metric("KB Status", kb_status)
    st.metric("Turns", len(st.session_state.history) // 2)

    st.markdown("---")

    # ── File Upload ──────────────────────────────────────────
    st.markdown("### Upload Documents")
    uploaded_files = st.file_uploader(
        "Add PDFs or text files to your session KB",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="file_uploader",
    )

    if uploaded_files:
        for uf in uploaded_files:
            if uf.name not in st.session_state.uploaded_files_list:
                chunk_count = ingest_uploaded_file(uf, st.session_state.thread_id)
                if chunk_count > 0:
                    st.toast(f"Injected {chunk_count} chunks from {uf.name}")
                    st.session_state.uploaded_files_list.append(uf.name)

    if st.session_state.uploaded_files_list:
        st.markdown("**Your Documents**")
        for fname in st.session_state.uploaded_files_list:
            st.caption(f"- {fname}")

    st.markdown("---")

    # ── Export Session ───────────────────────────────────────
    assistant_msgs = [m for m in st.session_state.history if m.get("role") == "assistant"]
    if assistant_msgs:
        try:
            from aria.export import generate_session_pdf
            date_str = datetime.now().strftime("%Y%m%d")
            filename = f"aria_session_{st.session_state.thread_id[:8]}_{date_str}.pdf"
            pdf_bytes = generate_session_pdf(st.session_state.history, st.session_state.thread_id)
            st.download_button(
                "Export Session as PDF",
                data=pdf_bytes,
                file_name=filename,
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            st.caption(f"Export unavailable: {e}")

    # ── New Session ──────────────────────────────────────────
    if st.button("New Session", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.history = []
        st.session_state.uploaded_files_list = []
        st.rerun()


# ── Main Content ──────────────────────────────────────────────────────────────

st.title("ARIA — Agentic Intelligence")
st.caption("Plan · Retrieve · Reflect · Synthesise")
st.markdown("---")

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and "report" in msg:
            render_report(
                msg["report"],
                msg.get("faithfulness", 0.0),
                msg.get("sources", []),
                msg.get("sub_queries", []),
                msg.get("route", "both"),
                msg.get("reranker_scores", []),
                msg.get("kb_coverage_score", 0.0),
                msg.get("comparison_mode", False),
                msg.get("arxiv_papers", []),
                msg.get("llm_provider_used", "groq"),
                msg.get("graph_context", ""),
            )
        else:
            st.write(msg["content"])

if prompt := st.chat_input("Ask a research question..."):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        status_placeholder.markdown("<span style='color:#a0aec0;'>[System] Initialising...</span>", unsafe_allow_html=True)

        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        initial = make_initial_state(prompt, st.session_state.thread_id)

        # Inject user context from profile
        try:
            from aria.user_profile import load_profile, build_user_context
            profile = load_profile(st.session_state.user_id)
            initial["user_context"] = build_user_context(profile)
        except Exception:
            pass

        try:
            with st.spinner("Researching..."):
                result = st.session_state.graph.invoke(initial, config=config)
            status_placeholder.empty()
        except Exception as e:
            status_placeholder.empty()
            st.error(f"Error: {e}")
            st.stop()

        render_report(
            result.get("report", {}),
            result.get("faithfulness", 0.0),
            result.get("sources", []),
            result.get("sub_queries", []),
            result.get("route", "both"),
            result.get("reranker_scores", []),
            result.get("kb_coverage_score", 0.0),
            result.get("comparison_mode", False),
            result.get("arxiv_papers", []),
            result.get("llm_provider_used", "groq"),
            result.get("graph_context", ""),
        )

        st.session_state.history.append({
            "role": "assistant",
            "content": result.get("answer", ""),
            "report": result.get("report", {}),
            "faithfulness": result.get("faithfulness", 0.0),
            "sources": result.get("sources", []),
            "sub_queries": result.get("sub_queries", []),
            "route": result.get("route", "both"),
            "reranker_scores": result.get("reranker_scores", []),
            "kb_coverage_score": result.get("kb_coverage_score", 0.0),
            "comparison_mode": result.get("comparison_mode", False),
            "arxiv_papers": result.get("arxiv_papers", []),
            "llm_provider_used": result.get("llm_provider_used", "groq"),
            "graph_context": result.get("graph_context", ""),
        })
