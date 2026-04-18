# aria/config.py
# Loads config.yaml and builds the shared LLM instance used by all nodes.

import os
import yaml
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load .env from project root (works locally; Streamlit Cloud uses st.secrets)
load_dotenv()

_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

def load_config() -> dict:
    """Load and return the config.yaml as a dict."""
    with open(_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def get_llm():
    """
    Build and return the LangChain chat model based on config.yaml provider.
    Supported providers:
      - groq   → ChatGroq (GROQ_API_KEY)
      - google → ChatGoogleGenerativeAI (GOOGLE_API_KEY)
    """
    cfg = load_config()
    llm_cfg = cfg.get("llm", {})
    provider  = llm_cfg.get("provider", "groq")
    model     = llm_cfg.get("model", "llama-3.3-70b-versatile")
    temp      = llm_cfg.get("temperature", 0.1)
    max_tok   = llm_cfg.get("max_tokens", 2048)

    if provider == "groq":
        try:
            from langchain_groq import ChatGroq
            api_key = os.getenv("GROQ_API_KEY") or _streamlit_secret("GROQ_API_KEY")
            llm = ChatGroq(
                model=model,
                temperature=temp,
                max_tokens=max_tok,
                groq_api_key=api_key,
            )
            # Quick validation: won't detect 429 here, but catches config errors
            return llm
        except Exception:
            pass
        # Auto-fallback to Gemini if Groq construction fails
        google_key = os.getenv("GOOGLE_API_KEY") or _streamlit_secret("GOOGLE_API_KEY")
        if google_key:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=temp,
                max_output_tokens=max_tok,
                google_api_key=google_key,
            )
        # Last resort: return Groq anyway (will fail at invoke time)
        from langchain_groq import ChatGroq
        api_key = os.getenv("GROQ_API_KEY") or _streamlit_secret("GROQ_API_KEY")
        return ChatGroq(
            model=model,
            temperature=temp,
            max_tokens=max_tok,
            groq_api_key=api_key,
        )
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GOOGLE_API_KEY") or _streamlit_secret("GOOGLE_API_KEY")
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temp,
            max_output_tokens=max_tok,
            google_api_key=api_key,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider!r}. Use 'groq' or 'google'.")


def _streamlit_secret(key: str) -> Optional[str]:
    """Safely attempt to read from Streamlit secrets (only in cloud env)."""
    try:
        import streamlit as st
        return st.secrets.get(key)
    except Exception:
        return None
