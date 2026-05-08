"""
Retrieval functions: vector-only, text-only, and hybrid (Reciprocal Rank Fusion).

Each function returns a list of de-duplicated document rows (best chunk per
parent doc) sorted best-first. The shape is identical across modes so callers
can swap them transparently.

Returned row dict:
  {
    "doc_id"   : str,   # parent BEIR doc id
    "chunk_idx": int,   # which chunk of the parent doc
    "title"    : str,
    "text"     : str,
    "score"    : float, # mode-specific score (higher = better)
  }

For hybrid we use RRF (Reciprocal Rank Fusion): for each candidate doc d,
  RRF(d) = Σ (1 / (k_rrf + rank_in_pipeline_i))
where k_rrf is a smoothing constant (60 is the standard from Cormack et al.
2009). Atlas natively supports this via $rankFusion (Atlas 8.1+); we
implement it client-side for maximum cluster-version compatibility and
because the resulting Python is easy to read and tune.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from lib import INDEX_NAME

TEXT_INDEX_NAME = "voyage_text_index"


# ── Vector search ────────────────────────────────────────────────────────────

def vector_only(coll, q_vec: list[float], top_k: int = 10) -> list[dict]:
    """Pure $vectorSearch, deduped to one row per parent doc."""
    pipeline = [
        {"$vectorSearch": {
            "index"        : INDEX_NAME,
            "path"         : "embedding",
            "queryVector"  : q_vec,
            "numCandidates": top_k * 20,
            "limit"        : top_k * 4,
        }},
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
        {"$sort": {"score": -1}},
    ]
    return _dedup_by_doc_id(coll.aggregate(pipeline))[:top_k]


# ── Text search ──────────────────────────────────────────────────────────────

def text_only(coll, query_text: str, top_k: int = 10) -> list[dict]:
    """Pure $search (BM25-style), deduped to one row per parent doc."""
    pipeline = [
        {"$search": {
            "index": TEXT_INDEX_NAME,
            "text": {"path": "text", "query": query_text},
        }},
        {"$addFields": {"score": {"$meta": "searchScore"}}},
        {"$limit": top_k * 4},
        {"$sort": {"score": -1}},
    ]
    return _dedup_by_doc_id(coll.aggregate(pipeline))[:top_k]


# ── Hybrid: native $rankFusion (Atlas 8.0+) ─────────────────────────────────

# Default first-stage candidate depth per mode. Bumped to 100 (was 40) to
# give fusion more material to work with — RRF can't recover relevant docs
# that are below position N in BOTH rankings.
DEFAULT_CANDIDATES = 100


def hybrid(
    coll,
    q_vec: list[float],
    query_text: str,
    top_k: int = 10,
    alpha: float = 0.5,
    candidates_per_mode: int | None = None,
) -> list[dict]:
    """
    Server-side Reciprocal Rank Fusion via $rankFusion (Atlas 8.0+).

    Atlas computes:
        RRFscore(d) = Σ (w_i / (60 + rank_i(d)))
    across each input pipeline, where w_i is the per-pipeline weight.
    We pass weights={vector: alpha, text: 1-alpha} so alpha controls the
    vector/text balance (alpha=1.0 → vector only; alpha=0.0 → text only).

    Because the pipeline restrictions on $rankFusion forbid post-retrieval
    projections inside its selection pipelines, we run the fusion over
    chunks then deduplicate to one chunk per parent doc client-side.
    """
    n = candidates_per_mode or DEFAULT_CANDIDATES

    # $rankFusion requires non-zero weights; nudge edge cases up slightly so
    # we don't trigger validation errors at alpha=0 or 1 while still letting
    # the dominant signal effectively own the result.
    EPS = 1e-3
    w_vec  = max(alpha, EPS)
    w_text = max(1.0 - alpha, EPS)

    pipeline = [
        {"$rankFusion": {
            "input": {
                "pipelines": {
                    "vector": [
                        {"$vectorSearch": {
                            "index"        : INDEX_NAME,
                            "path"         : "embedding",
                            "queryVector"  : q_vec,
                            "numCandidates": n * 4,
                            "limit"        : n,
                        }},
                    ],
                    "text": [
                        {"$search": {
                            "index": TEXT_INDEX_NAME,
                            "text" : {"path": "text", "query": query_text},
                        }},
                        {"$limit": n},
                    ],
                },
            },
            "combination": {"weights": {"vector": w_vec, "text": w_text}},
        }},
        {"$addFields": {"score": {"$meta": "score"}}},
        {"$limit": top_k * 4},
    ]
    return _dedup_by_doc_id(coll.aggregate(pipeline))[:top_k]


# ── Mode dispatch ────────────────────────────────────────────────────────────
#
# Note: an earlier iteration of this file included a comb_sum mode (convex
# combination of min-max normalized scores). It was removed because Atlas
# 8.3+ ships native Relative Score Fusion via $rankFusion that does exactly
# this — duplicating it client-side wasn't worth the maintenance burden,
# especially since the empirical lift over weighted-RRF was marginal.

MODES = ("vector", "text", "hybrid")
DEFAULT_ALPHA = 0.5


def retrieve(
    mode: str,
    coll,
    q_vec: list[float],
    query_text: str,
    top_k: int = 10,
    alpha: float = DEFAULT_ALPHA,
) -> list[dict]:
    """Dispatch to the requested mode. Used by query.py and the test harness."""
    if mode == "vector":
        return vector_only(coll, q_vec, top_k=top_k)
    if mode == "text":
        return text_only(coll, query_text, top_k=top_k)
    if mode == "hybrid":
        return hybrid(coll, q_vec, query_text, top_k=top_k, alpha=alpha)
    raise ValueError(f"unknown retrieval mode '{mode}' (expected one of {MODES})")


def multi_query_retrieve(
    mode: str,
    coll,
    queries: list[tuple[list[float] | None, str]],
    top_k: int = 10,
    alpha: float = DEFAULT_ALPHA,
    k_rrf: int = 60,
    candidates_per_query: int | None = None,
) -> list[dict]:
    """
    Fuse retrieval results from multiple (q_vec, q_text) pairs via RRF.

    Used by query_rewriter outputs that produce more than one rewrite (e.g.
    multi / decompose). For a single-element queries list this returns the
    same result as retrieve(mode, ...).
    """
    if len(queries) == 1:
        q_vec, q_text = queries[0]
        return retrieve(mode, coll, q_vec, q_text, top_k=top_k, alpha=alpha)

    n = candidates_per_query or top_k * 2
    rrf_scores: dict[str, float] = defaultdict(float)
    rows_by_id: dict[str, dict]  = {}

    for q_vec, q_text in queries:
        rows = retrieve(mode, coll, q_vec, q_text, top_k=n, alpha=alpha)
        for rank, row in enumerate(rows, 1):
            rrf_scores[row["doc_id"]] += 1.0 / (k_rrf + rank)
            rows_by_id.setdefault(row["doc_id"], row)

    fused = []
    for doc_id, score in rrf_scores.items():
        row = dict(rows_by_id[doc_id])
        row["score"] = score
        fused.append(row)

    fused.sort(key=lambda r: r["score"], reverse=True)
    return fused[:top_k]


# ── Internal helpers ─────────────────────────────────────────────────────────

def _dedup_by_doc_id(rows: Iterable[dict]) -> list[dict]:
    """Keep the highest-scoring chunk per parent doc."""
    seen: dict[str, dict] = {}
    for row in rows:
        did = row["doc_id"]
        if did not in seen or row["score"] > seen[did]["score"]:
            seen[did] = row
    return sorted(seen.values(), key=lambda r: r["score"], reverse=True)
