"""
Information retrieval metrics computed from a ranked list of doc IDs and
a set/dict of relevant doc IDs (with optional graded relevance scores).

All functions accept:
  ranked    — list[str], doc IDs in retrieval order (best first)
  relevant  — set[str] (binary) or dict[str, int] (graded; integer relevance
              scores ≥ 1 mean "relevant"; 0/missing means "not relevant")

For BEIR qrels, pass the per-query qrels dict directly (e.g.
  {"doc_42": 1, "doc_99": 2, "doc_7": 0, ...}
). Helpers below accept either form transparently.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping


# ── Internal helpers ──────────────────────────────────────────────────────────

def _binary_relevant_set(relevant: set[str] | Mapping[str, int]) -> set[str]:
    if isinstance(relevant, Mapping):
        return {did for did, score in relevant.items() if score and score > 0}
    return set(relevant)


def _graded(relevant: set[str] | Mapping[str, int], doc_id: str) -> int:
    """Graded relevance score for a doc (0 if not relevant)."""
    if isinstance(relevant, Mapping):
        score = relevant.get(doc_id, 0) or 0
        return max(0, int(score))
    return 1 if doc_id in relevant else 0


# ── Metrics ──────────────────────────────────────────────────────────────────

def precision_at_k(ranked: list[str], relevant: set[str] | Mapping[str, int], k: int) -> float:
    if k <= 0:
        return 0.0
    rel_set = _binary_relevant_set(relevant)
    top_k = ranked[:k]
    return sum(1 for d in top_k if d in rel_set) / k


def recall_at_k(ranked: list[str], relevant: set[str] | Mapping[str, int], k: int) -> float:
    rel_set = _binary_relevant_set(relevant)
    if not rel_set:
        return 0.0
    top_k = set(ranked[:k])
    return len(top_k & rel_set) / len(rel_set)


def average_precision(ranked: list[str], relevant: set[str] | Mapping[str, int]) -> float:
    """Standard AP — undefined if no relevant docs exist (returns 0.0)."""
    rel_set = _binary_relevant_set(relevant)
    if not rel_set:
        return 0.0
    n_rel, cum_p = 0, 0.0
    for rank, did in enumerate(ranked, 1):
        if did in rel_set:
            n_rel += 1
            cum_p += n_rel / rank
    return cum_p / len(rel_set)


def reciprocal_rank(ranked: list[str], relevant: set[str] | Mapping[str, int]) -> float:
    """1 / (rank of first relevant doc), or 0 if none retrieved."""
    rel_set = _binary_relevant_set(relevant)
    for rank, did in enumerate(ranked, 1):
        if did in rel_set:
            return 1.0 / rank
    return 0.0


def dcg_at_k(ranked: list[str], relevant: set[str] | Mapping[str, int], k: int) -> float:
    """Discounted cumulative gain using the standard log2(rank+1) discount."""
    return sum(
        _graded(relevant, did) / math.log2(rank + 1)
        for rank, did in enumerate(ranked[:k], 1)
    )


def ndcg_at_k(ranked: list[str], relevant: set[str] | Mapping[str, int], k: int) -> float:
    """Normalized DCG@k. Falls back to binary relevance if input is a set."""
    if isinstance(relevant, Mapping):
        ideal_grades = sorted((s for s in relevant.values() if s and s > 0), reverse=True)
    else:
        ideal_grades = [1] * len(relevant)
    if not ideal_grades:
        return 0.0
    idcg = sum(g / math.log2(rank + 1) for rank, g in enumerate(ideal_grades[:k], 1))
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(ranked, relevant, k) / idcg


# ── Aggregator ────────────────────────────────────────────────────────────────

# The metric configuration the demo reports. Keep this list as the single
# source of truth so query.py and test_harness.py stay in sync.
METRIC_KS = (5, 10)


def compute_query_metrics(
    ranked: list[str],
    relevant: set[str] | Mapping[str, int],
) -> dict[str, float]:
    """Compute all reported metrics for one query."""
    out: dict[str, float] = {}
    for k in METRIC_KS:
        out[f"P@{k}"]    = precision_at_k(ranked, relevant, k)
        out[f"R@{k}"]    = recall_at_k(ranked, relevant, k)
        out[f"NDCG@{k}"] = ndcg_at_k(ranked, relevant, k)
    out["MRR"] = reciprocal_rank(ranked, relevant)
    out["AP"]  = average_precision(ranked, relevant)
    return out


def aggregate_metrics(per_query: list[dict[str, float]]) -> dict[str, float]:
    """Mean each metric across queries. AP-mean is reported as MAP."""
    if not per_query:
        return {}
    keys = per_query[0].keys()
    agg = {k: sum(q[k] for q in per_query) / len(per_query) for k in keys}
    agg["MAP"] = agg.pop("AP")
    return agg


def format_summary(agg: dict[str, float]) -> str:
    """One-line summary: P@5 R@5 NDCG@5 NDCG@10 MRR MAP."""
    order = ["P@5", "R@5", "NDCG@5", "NDCG@10", "MRR", "MAP"]
    parts = [f"{k}={agg[k]:.3f}" for k in order if k in agg]
    return "  ".join(parts)
