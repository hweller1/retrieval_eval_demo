"""
Query interpretation layer.

Given a query, a cheap LLM (gpt-4o-mini) returns a *per-query* fusion
weight `alpha ∈ [0, 1]` for hybrid / comb_sum retrieval:

  alpha = 1.0  →  pure vector / semantic search
  alpha = 0.5  →  balanced
  alpha = 0.0  →  pure text / BM25

The classifier looks at query characteristics — exact-string artifacts
(codes, named entities, abbreviations, technical IDs) push toward text;
abstract concepts and paraphrasable language push toward vector. This is
dataset-agnostic: the same prompt makes per-query decisions on scifact,
nfcorpus, touche2020, fiqa, etc.

Implementation choices:
  - Single-token-ish output to keep cost minimal (~$0.0001/query).
  - JSON-shaped output for parsing robustness.
  - Falls back to alpha=0.7 (mild vector bias) if the LLM output is
    unparseable — empirically vector is the stronger signal on most
    BEIR datasets when in doubt.
"""

from __future__ import annotations

import json
import re
from typing import Iterable

import llm_client


SYSTEM_PROMPT = """\
You route search queries between SEMANTIC (dense vector / voyage-context-3)
and LEXICAL (BM25 / Atlas Search) retrieval. Output a single JSON object:

  {"alpha": 0.85, "reason": "abstract concept, paraphrasable"}

`alpha` is the weight given to SEMANTIC (vector) retrieval, in [0, 1].
The remainder, (1 - alpha), is given to LEXICAL retrieval.

Heuristics:
- Pure-keyword cues (named entities, IDs, codes, technical abbreviations,
  exact strings, file paths, error messages, stock tickers): alpha ≤ 0.3.
- Mixed (some keywords + some natural language): alpha ≈ 0.5.
- Conceptual / paraphrasable / fully natural language with no rare
  vocabulary: alpha ≥ 0.7.
- Very short questions about an abstract topic: alpha ≥ 0.7.

Output ONLY the JSON object. No prose, no code fences, no explanation."""


FALLBACK_ALPHA = 0.7   # mild vector bias when parsing fails — vector is usually stronger
DEFAULT_MODEL  = "gpt-4o-mini"


_alpha_cache: dict[str, float] = {}


def predict_alpha(query: str, model: str = DEFAULT_MODEL) -> float:
    """Predict a per-query alpha. Cached so repeated calls are free."""
    if query in _alpha_cache:
        return _alpha_cache[query]

    raw = llm_client.complete(
        prompt=f"Query: {query}",
        system=SYSTEM_PROMPT,
        model=model,
        temperature=0.0,
        max_tokens=64,
    )
    alpha = _parse_alpha(raw)
    _alpha_cache[query] = alpha
    return alpha


def predict_alphas(queries: Iterable[str], model: str = DEFAULT_MODEL) -> list[float]:
    """Predict alphas for a list of queries (sequential — keep OpenAI happy)."""
    return [predict_alpha(q, model=model) for q in queries]


def _parse_alpha(raw: str) -> float:
    """Robust parser: tries JSON first, then falls back to first float in [0,1]."""
    raw = raw.strip()

    # JSON path
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "alpha" in obj:
            a = float(obj["alpha"])
            return _clamp(a)
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback: first float-looking token
    match = re.search(r"-?\d+(?:\.\d+)?", raw)
    if match:
        try:
            return _clamp(float(match.group(0)))
        except ValueError:
            pass

    return FALLBACK_ALPHA


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def clear_cache() -> None:
    _alpha_cache.clear()
