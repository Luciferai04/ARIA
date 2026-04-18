#!/usr/bin/env python3
"""
scripts/ingest_kb.py
────────────────────
One-time script to populate ChromaDB with 45+ documents from:
  • arXiv papers  (via LangChain ArxivLoader)
  • Web pages     (via WebBaseLoader)
  • Local PDFs    (via PyPDFLoader, from data/raw/)

Run ONCE before launching the app:
    python scripts/ingest_kb.py

Commit data/chroma_db/ to the repo so Streamlit Cloud skips runtime ingestion.
"""

import sys
import os
from pathlib import Path

# Allow imports from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.document_loaders import ArxivLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma  # type: ignore
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore

CHROMA_PATH = str(Path(__file__).parent.parent / "data" / "chroma_db")
COLLECTION  = "aria_kb"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Chunking parameters (do not change) ──────────────────────────────────────
SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    separators=["\n\n", "\n", ". ", " ", ""],
)

# ── Source definitions ────────────────────────────────────────────────────────
ARXIV_QUERIES = [
    ("LangGraph agentic AI 2024", 5),
    ("retrieval augmented generation RAG 2024", 5),
    ("reinforcement learning from human feedback RLHF", 5),
    ("large language model reasoning chain of thought", 5),
    ("transformer attention mechanism self-attention", 5),
    ("vector database embedding semantic search", 5),
    ("LLM agent planning tool use", 5),
    ("hallucination detection language models faithfulness", 5),
    ("knowledge graph question answering", 5),
]

WEB_URLS = [
    "https://langchain-ai.github.io/langgraph/concepts/",
    "https://python.langchain.com/docs/concepts/rag/",
    "https://python.langchain.com/docs/concepts/agents/",
    "https://python.langchain.com/docs/concepts/tools/",
    "https://huggingface.co/blog/rag-evaluation",
]

RAW_PDF_DIR = Path(__file__).parent.parent / "data" / "raw"


def load_arxiv_docs(query: str, max_docs: int) -> list:
    """Load papers from arXiv using ArxivLoader."""
    try:
        loader = ArxivLoader(query=query, load_max_docs=max_docs)
        docs = loader.load()
        print(f"  ✓ arXiv '{query[:40]}' → {len(docs)} docs")
        return docs
    except Exception as e:
        print(f"  ✗ arXiv '{query[:40]}' failed: {e}")
        return []


def load_web_docs(url: str) -> list:
    """Load a single web page."""
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        print(f"  ✓ Web   '{url[:60]}' → {len(docs)} docs")
        return docs
    except Exception as e:
        print(f"  ✗ Web   '{url[:60]}' failed: {e}")
        return []


def load_pdf_docs(pdf_dir: Path) -> list:
    """Load all PDFs from data/raw/."""
    docs = []
    if not pdf_dir.exists():
        pdf_dir.mkdir(parents=True, exist_ok=True)
        print(f"  ℹ  data/raw/ created — drop PDFs here and re-run to ingest them.")
        return docs
    for pdf_path in pdf_dir.glob("*.pdf"):
        try:
            loader = PyPDFLoader(str(pdf_path))
            loaded = loader.load()
            docs.extend(loaded)
            print(f"  ✓ PDF   '{pdf_path.name}' → {len(loaded)} pages")
        except Exception as e:
            print(f"  ✗ PDF   '{pdf_path.name}' failed: {e}")
    return docs


def main():
    print("=" * 60)
    print("ARIA Knowledge Base Ingestion")
    print("=" * 60)

    all_docs = []

    # 1. arXiv papers
    print("\n[1/3] Loading arXiv papers …")
    for query, max_docs in ARXIV_QUERIES:
        all_docs.extend(load_arxiv_docs(query, max_docs))

    # 2. Web pages
    print("\n[2/3] Loading web pages …")
    for url in WEB_URLS:
        all_docs.extend(load_web_docs(url))

    # 3. Local PDFs
    print("\n[3/3] Loading local PDFs from data/raw/ …")
    all_docs.extend(load_pdf_docs(RAW_PDF_DIR))

    print(f"\nTotal raw documents loaded: {len(all_docs)}")

    if not all_docs:
        print("⚠  No documents loaded. Check your internet connection or add PDFs to data/raw/.")
        sys.exit(1)

    # ── Split into chunks ────────────────────────────────────────────────────
    print("\nSplitting documents into chunks …")
    chunks = SPLITTER.split_documents(all_docs)
    print(f"Total chunks after splitting: {len(chunks)}")

    # ── Embed and persist to ChromaDB ────────────────────────────────────────
    print("\nEmbedding and storing in ChromaDB …")
    print(f"  Model  : {EMBED_MODEL}")
    print(f"  Path   : {CHROMA_PATH}")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=CHROMA_PATH,
    )

    count = vectorstore._collection.count()
    print(f"\n✅ Ingestion complete! {count} chunks stored in ChromaDB.")
    print(f"   Commit data/chroma_db/ to your repo for Streamlit Cloud deployment.")
    print("=" * 60)


if __name__ == "__main__":
    main()
