"""
Query rewriting strategies — turn one user query into one or more
embed-and-retrieve strings.

All rewriters share the signature:

    rewrite(query: str) -> list[str]

The first element is the "primary" rewrite (used directly for single-query
retrieval); when the list has more than one element, the caller fuses
results from each element via RRF (see retrieve.multi_query_retrieve).

Strategies:

  none       — passthrough. Returns [query]. No LLM call.
  hyde       — Hypothetical Document Embeddings (Gao et al. 2022). LLM
               drafts a passage that *would answer* the query; embed that.
               One element returned (the hypothetical doc).
  multi      — Multi-query expansion (Wang et al. 2023). LLM produces
               paraphrases / alternate phrasings; we keep the original
               plus N rewrites and fuse retrieval results.
  decompose  — Question decomposition. LLM breaks a complex query into
               independent sub-questions; each sub-query retrieves
               separately and we fuse.

OPENAI_API_KEY must be set for any rewriter other than `none`.
"""

from __future__ import annotations

from typing import Callable

import llm_client


REWRITERS: tuple[str, ...] = ("none", "hyde", "multi", "decompose")
DEFAULT_REWRITER = "none"


# ── Strategy implementations ─────────────────────────────────────────────────

def _rewrite_none(query: str) -> list[str]:
    return [query]


_HYDE_SYSTEM = (
    "You are a passage-writing assistant. Given a question, write a single "
    "concise factual passage (2–4 sentences) that would directly answer the "
    "question if it appeared in a document. Do not hedge or apologize. Do "
    "not repeat the question. Output only the passage."
)


def _rewrite_hyde(query: str) -> list[str]:
    passage = llm_client.complete(
        prompt=f"Question: {query}\n\nPassage:",
        system=_HYDE_SYSTEM,
        max_tokens=200,
    )
    return [passage.strip() or query]


_MULTI_SYSTEM = (
    "You rewrite search queries. Given one query, produce 3 alternate "
    "phrasings that preserve meaning but vary vocabulary, specificity, "
    "and phrasing style. Output exactly 3 lines, each containing one "
    "rewritten query, with no numbering or commentary."
)


def _rewrite_multi(query: str) -> list[str]:
    raw = llm_client.complete(
        prompt=f"Original query: {query}\n\n3 alternate phrasings:",
        system=_MULTI_SYSTEM,
        max_tokens=200,
    )
    rewrites = [line.strip(" -•\t") for line in raw.splitlines() if line.strip()]
    rewrites = [r for r in rewrites if len(r) > 3][:3]
    return [query, *rewrites] if rewrites else [query]


_DECOMPOSE_SYSTEM = (
    "You decompose complex questions. Given one question, output 2–4 "
    "simpler sub-questions whose answers together address the original. "
    "If the question is already simple, output it unchanged. Output one "
    "sub-question per line with no numbering or commentary."
)


def _rewrite_decompose(query: str) -> list[str]:
    raw = llm_client.complete(
        prompt=f"Question: {query}\n\nSub-questions:",
        system=_DECOMPOSE_SYSTEM,
        max_tokens=200,
    )
    parts = [line.strip(" -•\t") for line in raw.splitlines() if line.strip()]
    parts = [p for p in parts if len(p) > 3][:4]
    return parts or [query]


# ── Dispatch ─────────────────────────────────────────────────────────────────

_DISPATCH: dict[str, Callable[[str], list[str]]] = {
    "none"     : _rewrite_none,
    "hyde"     : _rewrite_hyde,
    "multi"    : _rewrite_multi,
    "decompose": _rewrite_decompose,
}


def rewrite(strategy: str, query: str) -> list[str]:
    fn = _DISPATCH.get(strategy)
    if fn is None:
        raise ValueError(f"unknown rewriter '{strategy}' (expected one of {REWRITERS})")
    return fn(query)
