"""
End-to-end evaluation of retrieval strategies on the SEC 10-K corpus.

Pipeline:
  1. Load the 300 trader queries from data_loaders/sec_queries.json.
  2. For each strategy, retrieve top-K chunks per query.
  3. Pool: collect every unique (qid, doc_id) seen by any strategy.
  4. Use llm_judge.grade_batch() to grade each unique pair (cached).
  5. Save qrels JSON. Recompute metrics for each strategy using these qrels.
  6. Print summary and per-category breakdown.

Usage:
  python3 experiments/sec_10k_eval.py
  python3 experiments/sec_10k_eval.py --num-queries 50  (smoke test)
  python3 experiments/sec_10k_eval.py --strategies vector hybrid_a08 dynamic
"""

from __future__ import annotations

import os, sys
import time
import json
import argparse
import pathlib

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))  # repo root
sys.path.insert(0, os.path.dirname(_HERE))                    # phase4/

import voyageai
import pymongo
import lib
import llm_judge
import lib_metrics
from lib import (
    DB_NAME, MONGODB_BASE_URL, MONGODB_URI, QUERY_MODEL, VOYAGE_API_KEY,
    collection_name, embed_queries, load_beir_dataset,
)
from retrieve import retrieve, multi_query_retrieve, DEFAULT_ALPHA
from query_rewriter import rewrite
from rerank import rerank as rerank_rows
import query_classifier


DATASET = "sec-10k"
TOP_K_RETRIEVE = 10            # how many top results per query/strategy to pool
TOP_K_REPORT   = 10            # K for NDCG@K reporting
RERANK_CANDIDATES = 50

QRELS_OUT = pathlib.Path(_HERE).parent / "data_loaders" / "sec_qrels.json"


# ── Strategy definitions ─────────────────────────────────────────────────────

STRATEGIES = [
    ("vector",       {"mode": "vector"}),
    ("text",         {"mode": "text"}),
    ("hybrid_a05",   {"mode": "hybrid", "alpha": 0.5}),
    ("hybrid_a08",   {"mode": "hybrid", "alpha": 0.8}),
    ("dynamic",      {"mode": "hybrid", "strategy_mode": "dynamic"}),
]


# ── Retrieval helper (mirrors query.query but returns ranked chunks) ────────

def run_strategy(
    coll, queries: dict[str, str], strategy: dict, voyage_client,
) -> dict[str, list[dict]]:
    """
    Run one strategy over all queries. Returns
      {qid: [top-K chunk rows with doc_id, text, score]}
    """
    mode = strategy.get("mode", "hybrid")
    alpha = strategy.get("alpha", DEFAULT_ALPHA)
    rewriter_strat = strategy.get("rewriter", "none")
    do_rerank = strategy.get("rerank", False)
    strategy_mode = strategy.get("strategy_mode", "static")

    qids = list(queries.keys())
    raw_texts = [queries[q] for q in qids]

    # If dynamic, get per-query strategy from classifier
    if strategy_mode == "dynamic":
        per_query = query_classifier.predict_strategies(raw_texts)
    else:
        per_query = [
            query_classifier.Strategy(alpha=alpha, rerank=do_rerank,
                                      rewriter=rewriter_strat, reasoning="<<static>>")
            for _ in qids
        ]

    # Apply rewriters
    rewrites_per_q = [rewrite(s.rewriter, t) for s, t in zip(per_query, raw_texts)]

    # Embed all flattened rewrites in one batched call (vector / hybrid only)
    needs_vector = mode in ("vector", "hybrid")
    flat_texts = [t for sub in rewrites_per_q for t in sub]
    if needs_vector and flat_texts:
        flat_vecs = embed_queries(voyage_client, flat_texts)
    else:
        flat_vecs = [None] * len(flat_texts)
    vecs_per_q: list[list] = []
    cur = 0
    for sub in rewrites_per_q:
        n = len(sub)
        vecs_per_q.append(flat_vecs[cur:cur + n])
        cur += n

    out: dict[str, list[dict]] = {}
    top_n = max(lib_metrics.METRIC_KS)
    for qid, original_text, sub_texts, sub_vecs, strat in zip(
        qids, raw_texts, rewrites_per_q, vecs_per_q, per_query
    ):
        first_stage_k = RERANK_CANDIDATES if strat.rerank else top_n
        sub_queries = list(zip(sub_vecs, sub_texts))
        rows = multi_query_retrieve(
            mode, coll, sub_queries, top_k=first_stage_k, alpha=strat.alpha,
        )
        if strat.rerank and rows:
            rows = rerank_rows(original_text, rows, top_k=top_n)
        # Dedupe to one chunk per parent doc, take top K
        seen: dict[str, dict] = {}
        for r in rows:
            did = r["doc_id"]
            if did not in seen or r["score"] > seen[did]["score"]:
                seen[did] = r
        out[qid] = sorted(seen.values(), key=lambda r: r["score"], reverse=True)[:top_n]
    return out


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="SEC 10-K LLM-judge evaluation.")
    p.add_argument("--num-queries", type=int, default=300,
                   help="how many queries to evaluate (default: 300, max)")
    p.add_argument("--strategies", nargs="+",
                   choices=[name for name, _ in STRATEGIES],
                   help="subset of strategies to run (default: all)")
    p.add_argument("--max-judge-workers", type=int, default=8,
                   help="parallel grading workers (default: 8)")
    p.add_argument("--skip-judge", action="store_true",
                   help="don't re-grade; reuse data_loaders/sec_qrels.json as-is")
    args = p.parse_args()

    chosen_strategies = [
        (name, kw) for name, kw in STRATEGIES
        if not args.strategies or name in args.strategies
    ]

    # ── Load corpus + queries (corpus needed only to verify, queries from JSON) ──
    print("═" * 88)
    print(f"  SEC 10-K LLM-judge evaluation × {DATASET}")
    print("═" * 88)
    corpus, queries, _, info = load_beir_dataset(DATASET)
    print(f"  Corpus     : {len(corpus)} 10-K filings")
    print(f"  Queries    : {len(queries)} (using {min(args.num_queries, len(queries))})")
    print(f"  Strategies : {', '.join(name for name, _ in chosen_strategies)}")

    # Subset queries
    qids_all = list(queries.keys())[:args.num_queries]
    sub_queries = {q: queries[q] for q in qids_all}

    voyage = voyageai.Client(api_key=VOYAGE_API_KEY, base_url=MONGODB_BASE_URL)
    mongo = pymongo.MongoClient(MONGODB_URI)
    coll  = mongo[DB_NAME][collection_name(DATASET)]

    # ── Phase 1: run all strategies ────────────────────────────────────────
    print()
    print("Phase 1 — retrieval")
    strategy_results: dict[str, dict[str, list[dict]]] = {}
    for name, kw in chosen_strategies:
        print(f"  [{name}] retrieving …", end=" ", flush=True)
        t0 = time.time()
        try:
            res = run_strategy(coll, sub_queries, kw, voyage)
            strategy_results[name] = res
            print(f"{len(res)} qids done  ({time.time() - t0:.1f}s)")
        except Exception as e:
            print(f"FAIL: {type(e).__name__}: {e}")
            strategy_results[name] = {}

    # ── Phase 2: pool & grade ───────────────────────────────────────────────
    print()
    print("Phase 2 — pool + LLM judge")
    if not args.skip_judge:
        # Build the union pool: (qid, doc_id) → best chunk passage seen
        pool: dict[tuple[str, str], str] = {}
        for name, results in strategy_results.items():
            for qid, rows in results.items():
                for r in rows[:TOP_K_RETRIEVE]:
                    key = (qid, r["doc_id"])
                    # Keep the highest-scored chunk text we've seen for this pair
                    pool.setdefault(key, r["text"])

        triples = [(qid, did, passage) for (qid, did), passage in pool.items()]
        print(f"  Pool size: {len(triples)} unique (query, doc) pairs")

        qrels_new = llm_judge.grade_batch(
            triples, sub_queries, max_workers=args.max_judge_workers,
        )

        existing_qrels = llm_judge.load_qrels(QRELS_OUT)
        merged = llm_judge.merge_qrels(existing_qrels, qrels_new)
        llm_judge.save_qrels(merged, QRELS_OUT)
        print(f"  Saved qrels to {QRELS_OUT}")
        qrels = merged
    else:
        qrels = llm_judge.load_qrels(QRELS_OUT)
        print(f"  Loaded {sum(len(v) for v in qrels.values())} existing grades from {QRELS_OUT}")

    # ── Phase 3: compute metrics ────────────────────────────────────────────
    print()
    print("Phase 3 — metrics")
    print()
    cols = ["P@5", "R@5", "NDCG@5", "NDCG@10", "MRR", "MAP"]
    label_w = max(len(name) for name, _ in chosen_strategies) + 2
    print(f"  {'Strategy':<{label_w}} " + " ".join(f"{c:>9}" for c in cols))
    print(f"  {'─' * label_w} " + " ".join(f"{'─' * 9:>9}" for _ in cols))

    summary: dict[str, dict[str, float]] = {}
    for name, kw in chosen_strategies:
        results = strategy_results.get(name, {})
        per_query_metrics = []
        for qid in qids_all:
            qrel = qrels.get(qid, {})
            ranked_ids = [r["doc_id"] for r in results.get(qid, [])]
            m = lib_metrics.compute_query_metrics(ranked_ids, qrel)
            per_query_metrics.append(m)
        agg = lib_metrics.aggregate_metrics(per_query_metrics)
        summary[name] = agg
        cells = " ".join(f"{agg.get(c, 0.0):>9.3f}" for c in cols)
        print(f"  {name:<{label_w}} {cells}")

    # ── Phase 4: per-category breakdown ─────────────────────────────────────
    print()
    print("Phase 4 — per-category NDCG@10 (business_model / key_risks / financial_health)")
    print()
    categories = ["business_model", "key_risks", "financial_health"]
    print(f"  {'Strategy':<{label_w}} " + " ".join(f"{c:>16}" for c in categories))
    print(f"  {'─' * label_w} " + " ".join(f"{'─' * 16:>16}" for _ in categories))
    for name, _ in chosen_strategies:
        results = strategy_results.get(name, {})
        cells = []
        for cat in categories:
            cat_qids = [q for q in qids_all if q.startswith(cat)]
            metrics_list = []
            for qid in cat_qids:
                qrel = qrels.get(qid, {})
                ranked_ids = [r["doc_id"] for r in results.get(qid, [])]
                m = lib_metrics.compute_query_metrics(ranked_ids, qrel)
                metrics_list.append(m)
            if metrics_list:
                agg = lib_metrics.aggregate_metrics(metrics_list)
                cells.append(f"{agg.get('NDCG@10', 0.0):>16.3f}")
            else:
                cells.append(f"{'-':>16}")
        print(f"  {name:<{label_w}} " + " ".join(cells))
    print()

    mongo.close()


if __name__ == "__main__":
    main()
