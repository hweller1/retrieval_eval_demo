"""
Cross-encoder reranking via Voyage's rerank-2.5 (MongoDB-hosted).

Reranking is a *second stage* on top of any retrieval mode (vector,
text, hybrid). The pattern is:

  1. First-stage retrieval returns top-N candidates (cheap, large N).
  2. Reranker scores the (query, candidate) pairs with a cross-encoder
     and reorders. Returns top-K (expensive, small K).

The cross-encoder sees query and document together so it can judge
relevance directly, unlike bi-encoder retrieval which compares
independent embeddings. Typically lifts NDCG by 5–15 points on BEIR.

Endpoint: POST https://ai.mongodb.com/v1/rerank
Request shape:
  {
    "model"    : "rerank-2.5",
    "query"    : "<query text>",
    "documents": ["doc text 1", "doc text 2", ...],
    "top_k"    : 10
  }
Response:
  {"data": [{"index": int, "relevance_score": float}, ...]}
"""

from __future__ import annotations

import os
import requests

from lib import MONGODB_BASE_URL, VOYAGE_API_KEY

RERANK_MODEL    = "rerank-2.5"
RERANK_ENDPOINT = f"{MONGODB_BASE_URL}/rerank"
DEFAULT_RERANK_K = 10


def rerank(query: str, rows: list[dict], top_k: int = DEFAULT_RERANK_K) -> list[dict]:
    """
    Rerank `rows` against `query` using rerank-2.5.

    Each row must have a `text` field. Returns the same row dicts in the
    new order, with the `score` field replaced by the cross-encoder's
    relevance_score (higher = better, 0–1 range).

    If `rows` is empty or rerank fails, the input is returned unchanged.
    """
    if not rows:
        return rows
    if not VOYAGE_API_KEY:
        raise SystemExit("VOYAGE_API_KEY not set — required for rerank-2.5.")

    documents = [r["text"] for r in rows]
    payload = {
        "model"    : RERANK_MODEL,
        "query"    : query,
        "documents": documents,
        "top_k"    : min(top_k, len(rows)),
    }
    headers = {
        "Authorization": f"Bearer {VOYAGE_API_KEY}",
        "Content-Type" : "application/json",
    }
    response = requests.post(RERANK_ENDPOINT, json=payload, headers=headers, timeout=60)
    response.raise_for_status()

    data = response.json()["data"]
    # Reorder rows by reranker output
    reranked = []
    for entry in data:
        original = dict(rows[entry["index"]])
        original["score"] = float(entry["relevance_score"])
        original["_rerank"] = True
        reranked.append(original)
    return reranked
