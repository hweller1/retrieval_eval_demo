"""
Thin OpenAI wrapper used by query_rewriter.py (and future stages 4–5).

Lazy-imports the OpenAI SDK so the basic vector / text / hybrid retrieval
path works without OpenAI installed or OPENAI_API_KEY set. Only when a
caller invokes `complete(...)` do we initialize the client.
"""

from __future__ import annotations

import os
import threading

from dotenv import load_dotenv

load_dotenv()

# Default judge model — gpt-4o-mini is a solid general-purpose choice.
# For domain-specific or nuanced relevance judgements (legal, medical, graded
# relevance) a stronger model may produce more reliable grades. See the
# "Choosing the right judge model" section in README.md for a prompt you can
# use to get a tailored recommendation; then set DEFAULT_MODEL to the result.
#
# To swap providers, also pass base_url="https://<provider>/v1" to OpenAI()
# below. Any OpenAI-compatible API works; set OPENAI_API_KEY accordingly.
DEFAULT_MODEL = "gpt-4o-mini"

_client_lock = threading.Lock()
_client = None


def _get_client():
    """Lazy-init the OpenAI client. Raises a clear error if env is missing."""
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is not None:
            return _client
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit(
                "OPENAI_API_KEY not set. Add it to .env to use LLM-driven "
                "features (query rewriters, LLM-as-judge, agentic retrieval)."
            )
        try:
            from openai import OpenAI
        except ImportError as e:
            raise SystemExit(
                "openai package not installed. "
                "Run: pip3 install openai --break-system-packages"
            ) from e
        _client = OpenAI(api_key=api_key)
        return _client


def complete(
    prompt: str,
    system: str = "",
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> str:
    """One-shot completion. Returns the model's text reply, stripped."""
    client = _get_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()
