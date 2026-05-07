"""
Why is RRF-hybrid sometimes worse than vector alone?

Sweep two axes against an already-ingested dataset:
  - fusion method: hybrid (weighted RRF), comb_sum (convex of normalized scores)
  - alpha: 0.0 (text only) → 1.0 (vector only) in 0.1 steps

Plus the two pure-mode baselines (vector, text) and rerank variants of
the best fusion result.

This isolates the question "is RRF the right fusion?" from "is the text
index any good?" and "is the candidate pool deep enough?"

Usage:
  python3 experiments/sweep_fusion.py <dataset> [--num-queries N]

Assumes ingest.py has already populated the dataset's collection.
"""

from __future__ import annotations

import sys
import time
import argparse

# Allow running from project root or from experiments/
sys.path.insert(0, ".")

import query as query_mod
from lib import DATASETS
from retrieve import MODES


ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
FUSIONS = ["hybrid", "comb_sum"]   # weighted-RRF and CombSUM


def fmt_metric(v: float) -> str:
    return f"{v:.3f}"


def bar(value: float, width: int = 18, vmax: float = 1.0) -> str:
    fill = max(0.0, min(value / vmax, 1.0)) * width
    full = int(fill)
    eighths = " ▏▎▍▌▋▊▉█"
    partial = eighths[round((fill - full) * 8)] if full < width else ""
    return ("█" * full + partial).ljust(width)


def run_one(dataset: str, mode: str, alpha: float, num_queries: int) -> dict:
    """Returns aggregate metrics dict from one query run."""
    t0 = time.time()
    result = query_mod.query(
        dataset, num_queries=num_queries, mode=mode,
        rewriter="none", rerank=False, alpha=alpha, verbose=False,
    )
    return {
        "mode"   : mode,
        "alpha"  : alpha,
        "metrics": result.aggregate,
        "elapsed": time.time() - t0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Fusion-method × alpha sweep on a single dataset.")
    p.add_argument("dataset", choices=list(DATASETS.keys()))
    p.add_argument("--num-queries", type=int, default=10)
    args = p.parse_args()

    print("═" * 88)
    print(f"  Fusion sweep × {args.dataset}  ({args.num_queries} queries)")
    print("═" * 88)

    runs = []

    # Baselines: pure modes
    for mode in ("vector", "text"):
        print(f"  [{mode:<10}                          ] …", end=" ", flush=True)
        r = run_one(args.dataset, mode, alpha=0.5, num_queries=args.num_queries)
        runs.append(r)
        m = r["metrics"]
        print(f"NDCG@10={fmt_metric(m['NDCG@10'])}  MAP={fmt_metric(m['MAP'])}  ({r['elapsed']:.1f}s)")

    # Fusion sweep
    for fusion in FUSIONS:
        print()
        print(f"  -- {fusion} --")
        for alpha in ALPHAS:
            print(f"  [{fusion:<10}  alpha={alpha:.1f}              ] …", end=" ", flush=True)
            r = run_one(args.dataset, fusion, alpha=alpha, num_queries=args.num_queries)
            runs.append(r)
            m = r["metrics"]
            print(f"NDCG@10={fmt_metric(m['NDCG@10'])}  MAP={fmt_metric(m['MAP'])}  ({r['elapsed']:.1f}s)")

    # ── Summary visual ───────────────────────────────────────────────────
    print()
    print("═" * 88)
    print(f"  NDCG@10 by fusion × alpha  ({args.dataset})")
    print("═" * 88)

    pure_v = next(r for r in runs if r["mode"] == "vector")
    pure_t = next(r for r in runs if r["mode"] == "text")
    base   = pure_v["metrics"]["NDCG@10"]

    # Show baselines
    for r in (pure_v, pure_t):
        v = r["metrics"]["NDCG@10"]
        d = v - base
        print(f"  {r['mode']:<10}             {bar(v):<18}  {v:.3f}  (Δ vs vector: {d:+.3f})")

    print()
    for fusion in FUSIONS:
        print(f"  {fusion}")
        best = None
        for alpha in ALPHAS:
            r = next(x for x in runs if x["mode"] == fusion and abs(x["alpha"] - alpha) < 1e-6)
            v = r["metrics"]["NDCG@10"]
            d = v - base
            line = f"    α={alpha:.1f}        {bar(v)}  {v:.3f}  Δ {d:+.3f}"
            if best is None or v > best[0]:
                best = (v, alpha, line)
            print(line)
        print(f"    best for {fusion}: α={best[1]:.1f} → NDCG@10={best[0]:.3f}  (Δ {best[0]-base:+.3f})")
        print()


if __name__ == "__main__":
    main()
