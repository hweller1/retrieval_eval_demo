"""
Investigate why static-α RRF underperforms pure vector, and whether SOTA
techniques recover the loss.

Compares six retrieval strategies on an already-ingested dataset:

  1. vector                — baseline, pure $vectorSearch
  2. text                  — pure $search (BM25)
  3. hybrid (α=0.5)        — naive RRF, equal weight (the "broken" hybrid)
  4. hybrid (α=0.8)        — manually tuned static, vector-favored
  5. hybrid (α=dynamic)    — LLM classifier picks α per query
  6. comb_sum (α=dynamic)  — same routing but convex combination of
                             min-max normalized scores instead of RRF
  7. hybrid (α=dynamic) + rerank-2.5  — SOTA second-stage cross-encoder

Each strategy runs over the same N queries and reports
NDCG@10 / MAP / Recall@10 plus the delta against pure vector.

Usage:
  python3 experiments/compare_fusion_strategies.py <dataset> [--num-queries N]
"""

from __future__ import annotations

import sys
import time
import argparse

sys.path.insert(0, ".")

import query as query_mod
import query_classifier as qc
from lib import DATASETS


# (label, mode, alpha, alpha_mode, rerank)
STRATEGIES = [
    ("vector",                  "vector",   0.5, "static",  False),
    ("text",                    "text",     0.5, "static",  False),
    ("hybrid α=0.5 (RRF)",      "hybrid",   0.5, "static",  False),
    ("hybrid α=0.8 (RRF)",      "hybrid",   0.8, "static",  False),
    ("hybrid α=dynamic (RRF)",  "hybrid",   0.5, "dynamic", False),
    ("comb_sum α=dynamic",      "comb_sum", 0.5, "dynamic", False),
    ("hybrid α=dynamic + rerank","hybrid",  0.5, "dynamic", True),
]


def bar(value: float, width: int = 22, vmax: float = 1.0) -> str:
    fill = max(0.0, min(value / vmax, 1.0)) * width
    full = int(fill)
    eighths = " ▏▎▍▌▋▊▉█"
    partial = eighths[round((fill - full) * 8)] if full < width else ""
    return ("█" * full + partial).ljust(width)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compare static vs dynamic fusion strategies on one dataset.",
    )
    p.add_argument("dataset", choices=list(DATASETS.keys()))
    p.add_argument("--num-queries", type=int, default=10)
    args = p.parse_args()

    print("═" * 90)
    print(f"  Fusion strategy comparison × {args.dataset}  ({args.num_queries} queries)")
    print("═" * 90)
    print()

    qc.clear_cache()  # so dynamic α calls are recorded fresh

    results = []
    for label, mode, alpha, alpha_mode, rerank in STRATEGIES:
        print(f"  running {label:<32} …", end=" ", flush=True)
        t0 = time.time()
        run = query_mod.query(
            args.dataset,
            num_queries=args.num_queries,
            mode=mode,
            rewriter="none",
            rerank=rerank,
            alpha=alpha,
            alpha_mode=alpha_mode,
            verbose=False,
        )
        elapsed = time.time() - t0
        m = run.aggregate
        results.append((label, m, elapsed))
        print(f"NDCG@10={m['NDCG@10']:.3f}  MAP={m['MAP']:.3f}  R@10={m['R@10']:.3f}  "
              f"({elapsed:.1f}s)")

    # Summary table with deltas
    print()
    print("═" * 90)
    print(f"  Summary  ×  {args.dataset}")
    print("═" * 90)
    print()

    base_ndcg = results[0][1]["NDCG@10"]
    base_map  = results[0][1]["MAP"]
    print(f"  {'Strategy':<32} {'NDCG@10':>9}  Δ vs vec  {'MAP':>9}  Δ vs vec   Time")
    print(f"  {'─' * 32} {'─' * 9:>9}  {'─' * 8:>8}  {'─' * 9:>9}  {'─' * 8:>8}   {'─' * 6}")
    for label, m, elapsed in results:
        d_n = m["NDCG@10"] - base_ndcg
        d_m = m["MAP"]     - base_map
        print(f"  {label:<32} {m['NDCG@10']:>9.3f}  {d_n:>+8.3f}  "
              f"{m['MAP']:>9.3f}  {d_m:>+8.3f}  {elapsed:>5.1f}s")

    # NDCG@10 chart
    print()
    print(f"  NDCG@10 (bars to 1.0)")
    print(f"  {'─' * 60}")
    for label, m, _ in results:
        v = m["NDCG@10"]
        print(f"  {label:<32}  {bar(v)}  {v:.3f}")

    # If dynamic was used, show the per-query alphas chosen
    if any(r[1] for r in results):
        cache = qc._alpha_cache
        if cache:
            print()
            print(f"  Per-query α chosen by classifier:")
            for q, a in cache.items():
                print(f"    α={a:.2f}  |  {q[:80]}")

    print()


if __name__ == "__main__":
    main()
