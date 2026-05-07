"""
Compare static vs dynamic per-query strategy routing on a single dataset.

The "static" strategies hardcode one set of parameters across all queries:
  1. vector                  — pure $vectorSearch baseline
  2. text                    — pure $search (BM25)
  3. hybrid α=0.5            — naive RRF (the "broken" hybrid)
  4. hybrid α=0.8            — manually tuned static, vector-favored

The "dynamic" strategy routes per query via gpt-4o-mini:
  5. dynamic                 — classifier picks alpha + rewriter + rerank
                               from the query alone, no per-dataset tuning

Each strategy runs over the same N queries; we report NDCG@10 / MAP /
Recall@10 plus the delta against pure vector. The dynamic strategy also
prints the routing decision for each query so you can see what the
classifier did and why.

Usage:
  python3 experiments/compare_fusion_strategies.py <dataset> [--num-queries N]

Assumes the dataset's collection has already been ingested.
"""

from __future__ import annotations

import sys
import time
import argparse

sys.path.insert(0, ".")

import query as query_mod
import query_classifier as qc
from lib import DATASETS


# (label, kwargs to pass to query.query)
STRATEGIES = [
    ("vector",                {"mode": "vector",                                                       }),
    ("text",                  {"mode": "text",                                                         }),
    ("hybrid α=0.5 (RRF)",    {"mode": "hybrid",  "alpha": 0.5,                                        }),
    ("hybrid α=0.8 (RRF)",    {"mode": "hybrid",  "alpha": 0.8,                                        }),
    ("dynamic (per-query)",   {"mode": "hybrid",  "strategy_mode": "dynamic",                          }),
]


def bar(value: float, width: int = 22, vmax: float = 1.0) -> str:
    fill = max(0.0, min(value / vmax, 1.0)) * width
    full = int(fill)
    eighths = " ▏▎▍▌▋▊▉█"
    partial = eighths[round((fill - full) * 8)] if full < width else ""
    return ("█" * full + partial).ljust(width)


def main() -> None:
    p = argparse.ArgumentParser(description="Compare static vs dynamic strategies on one dataset.")
    p.add_argument("dataset", choices=list(DATASETS.keys()))
    p.add_argument("--num-queries", type=int, default=10)
    args = p.parse_args()

    print("═" * 90)
    print(f"  Strategy comparison × {args.dataset}  ({args.num_queries} queries)")
    print("═" * 90)
    print()

    qc.clear_cache()  # so dynamic decisions are recorded fresh

    results = []
    last_dynamic_run = None
    for label, extra in STRATEGIES:
        print(f"  running {label:<28} …", end=" ", flush=True)
        t0 = time.time()
        run = query_mod.query(
            args.dataset, num_queries=args.num_queries, verbose=False, **extra,
        )
        elapsed = time.time() - t0
        m = run.aggregate
        results.append((label, m, elapsed, run))
        if run.strategy_mode == "dynamic":
            last_dynamic_run = run
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
    print(f"  {'Strategy':<28} {'NDCG@10':>9}  Δ vs vec  {'MAP':>9}  Δ vs vec   Time")
    print(f"  {'─' * 28} {'─' * 9:>9}  {'─' * 8:>8}  {'─' * 9:>9}  {'─' * 8:>8}   {'─' * 6}")
    for label, m, elapsed, _ in results:
        d_n = m["NDCG@10"] - base_ndcg
        d_m = m["MAP"]     - base_map
        print(f"  {label:<28} {m['NDCG@10']:>9.3f}  {d_n:>+8.3f}  "
              f"{m['MAP']:>9.3f}  {d_m:>+8.3f}  {elapsed:>5.1f}s")

    # NDCG@10 chart
    print()
    print(f"  NDCG@10 (bars to 1.0)")
    print(f"  {'─' * 60}")
    for label, m, _, _ in results:
        v = m["NDCG@10"]
        print(f"  {label:<28}  {bar(v)}  {v:.3f}")

    # Routing trace from the dynamic run
    if last_dynamic_run and last_dynamic_run.per_query_strategies:
        print()
        print(f"  Per-query routing decisions (gpt-4o-mini):")
        print(f"  {'─' * 88}")
        for qres, strat in zip(last_dynamic_run.per_query, last_dynamic_run.per_query_strategies):
            q_short = qres.text[:60] + ("…" if len(qres.text) > 60 else "")
            print(f"    {strat.label():<26}  {q_short}")
            if strat.reasoning:
                print(f"    {' ':<26}  ↳ {strat.reasoning[:90]}")
    print()


if __name__ == "__main__":
    main()
