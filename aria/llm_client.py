# aria/llm_client.py
# LLM failover: Groq primary → Gemini Flash fallback on rate limits or errors.

import os
import sys
from aria.config import load_config, _streamlit_secret


def get_llm_with_fallback():
    """
    Return a ChatGroq instance. If Groq fails at invoke-time,
    use invoke_with_fallback() wrapper instead.
    """
    cfg = load_config()
    llm_cfg = cfg.get("llm", {})
    model = llm_cfg.get("model", "llama-3.3-70b-versatile")
    temp = llm_cfg.get("temperature", 0.1)
    max_tok = llm_cfg.get("max_tokens", 2048)

    from langchain_groq import ChatGroq
    api_key = os.getenv("GROQ_API_KEY") or _streamlit_secret("GROQ_API_KEY")
    return ChatGroq(
        model=model,
        temperature=temp,
        max_tokens=max_tok,
        groq_api_key=api_key,
    )


def invoke_with_fallback(prompt: str, state: dict = None) -> tuple:
    """
    Invoke the LLM with automatic Groq → Gemini failover.

    Returns:
        (response, provider_used) where provider_used is "groq" or "gemini".
    """
    # Try Groq first
    try:
        llm = get_llm_with_fallback()
        response = llm.invoke(prompt)
        return response, "groq"
    except Exception as groq_error:
        print(f"[LLM Failover] Groq unavailable, switching to Gemini: {groq_error}", file=sys.stderr)

    # Fallback to Gemini
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GOOGLE_API_KEY") or _streamlit_secret("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not configured for failover")

        cfg = load_config()
        llm_cfg = cfg.get("llm", {})
        temp = llm_cfg.get("temperature", 0.1)
        max_tok = llm_cfg.get("max_tokens", 2048)

        gemini = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=temp,
            max_output_tokens=max_tok,
            google_api_key=api_key,
        )
        response = gemini.invoke(prompt)
        return response, "gemini"
    except Exception as gemini_error:
        print(f"[LLM Failover] Gemini also failed: {gemini_error}", file=sys.stderr)
        raise Exception(
            f"Both LLM providers failed. Groq: {groq_error}. Gemini: {gemini_error}"
        )
