"""
Run the static-vs-dynamic strategy comparison across every supported BEIR
dataset and aggregate the results in one table. Auto-ingests datasets
that aren't yet populated at the requested sample size.

Strategies compared (per dataset):
  1. vector              — pure $vectorSearch baseline
  2. text                — pure $search (BM25)
  3. hybrid α=0.5 (RRF)  — naïve uniform-weight RRF (the "broken" baseline)
  4. hybrid α=0.8 (RRF)  — manually tuned static, vector-favored
  5. dynamic             — gpt-4o-mini routes alpha + rewriter + rerank
                           per query

Usage:
  python3 experiments/compare_all_datasets.py
  python3 experiments/compare_all_datasets.py --num-queries 30 --sample 500
  python3 experiments/compare_all_datasets.py --datasets scifact fiqa
"""

from __future__ import annotations

import sys
import time
import argparse

sys.path.insert(0, ".")

import pymongo
import lib
import ingest as ingest_mod
import query as query_mod
import query_classifier as qc
from lib import DATASETS, DB_NAME, collection_name


STRATEGIES = [
    ("vector",                {"mode": "vector"}),
    ("text",                  {"mode": "text"}),
    ("hybrid α=0.5",          {"mode": "hybrid", "alpha": 0.5}),
    ("hybrid α=0.8",          {"mode": "hybrid", "alpha": 0.8}),
    ("dynamic",               {"mode": "hybrid", "strategy_mode": "dynamic"}),
]


def chunks_in_collection(dataset: str) -> int:
    client = pymongo.MongoClient(lib.MONGODB_URI)
    try:
        return client[DB_NAME][collection_name(dataset)].estimated_document_count()
    finally:
        client.close()


def ensure_ingested(dataset: str, sample: int, min_chunks: int) -> None:
    """Re-ingest if the existing collection is too small."""
    have = chunks_in_collection(dataset)
    if have >= min_chunks:
        print(f"    [{dataset}] already has {have:,} chunks — skipping ingest")
        return
    print(f"    [{dataset}] only {have:,} chunks — ingesting at sample={sample} …")
    t0 = time.time()
    ingest_mod.ingest(dataset, corpus_sample=sample)
    print(f"    [{dataset}] ingest took {time.time() - t0:.1f}s")


def bar(value: float, width: int = 18, vmax: float = 1.0) -> str:
    fill = max(0.0, min(value / vmax, 1.0)) * width
    full = int(fill)
    eighths = " ▏▎▍▌▋▊▉█"
    partial = eighths[round((fill - full) * 8)] if full < width else ""
    return ("█" * full + partial).ljust(width)


def main() -> None:
    p = argparse.ArgumentParser(description="Static-vs-dynamic strategy sweep across all BEIR datasets.")
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS.keys()))
    p.add_argument("--sample", type=int, default=500)
    p.add_argument("--num-queries", type=int, default=100)
    p.add_argument("--min-chunks", type=int, default=None,
                   help="re-ingest if existing collection has fewer chunks than this "
                        "(default: estimated as 2 × sample)")
    args = p.parse_args()

    datasets = args.datasets if args.datasets else list(DATASETS.keys())
    # Default: require chunks ≥ sample (i.e. at least 1 chunk per doc on average).
    # Datasets like quora produce ~1 chunk/doc, so requiring 2× would loop forever.
    min_chunks = args.min_chunks if args.min_chunks is not None else args.sample

    print("═" * 96)
    print("  Static-vs-Dynamic strategy sweep across BEIR")
    print("═" * 96)
    print(f"  Datasets    : {', '.join(datasets)}")
    print(f"  Sample/ds   : {args.sample}")
    print(f"  Queries/ds  : {args.num_queries}")
    print(f"  Strategies  : {', '.join(label for label, _ in STRATEGIES)}")
    print()

    # ── Phase 1: ensure all datasets are ingested at the right sample size ──
    print("Phase 1 — ingest check")
    for ds in datasets:
        ensure_ingested(ds, args.sample, min_chunks)
    print()

    # ── Phase 2: run all strategies on all datasets ─────────────────────────
    print("Phase 2 — strategy runs")
    # results[dataset][strategy_label] = aggregate metrics dict
    results: dict[str, dict[str, dict[str, float]]] = {}

    for ds in datasets:
        print(f"\n  ── {ds} ──")
        results[ds] = {}
        qc.clear_cache()
        for label, extra in STRATEGIES:
            print(f"    {label:<16} …", end=" ", flush=True)
            t0 = time.time()
            try:
                run = query_mod.query(
                    ds, num_queries=args.num_queries, verbose=False, **extra,
                )
                m = run.aggregate
                results[ds][label] = m
                print(f"NDCG@10={m['NDCG@10']:.3f}  MAP={m['MAP']:.3f}  "
                      f"({time.time() - t0:.1f}s)")
            except Exception as e:
                print(f"FAIL: {type(e).__name__}: {e}")
                results[ds][label] = {}

    # ── Phase 3: summary tables ─────────────────────────────────────────────
    summary(results, datasets, "NDCG@10")
    summary(results, datasets, "MAP")
    summary(results, datasets, "MRR")

    # Aggregate "wins" — which strategy is best per dataset
    print()
    print("═" * 96)
    print("  Best strategy per dataset (NDCG@10)")
    print("═" * 96)
    print()
    for ds in datasets:
        ds_metrics = results.get(ds, {})
        candidates = [
            (label, m["NDCG@10"]) for label, m in ds_metrics.items() if m
        ]
        if not candidates:
            print(f"  {ds:<14}  no successful runs")
            continue
        best = max(candidates, key=lambda x: x[1])
        vec = ds_metrics.get("vector", {}).get("NDCG@10", 0.0)
        delta = best[1] - vec
        print(f"  {ds:<14}  best: {best[0]:<14} NDCG@10={best[1]:.3f}  "
              f"(Δ vs vector: {delta:+.3f})")

    # ── Phase 4: how often does dynamic beat the best static? ───────────────
    print()
    print("═" * 96)
    print("  Dynamic vs best static")
    print("═" * 96)
    print()
    print(f"  {'Dataset':<14}  {'best static':<22}  {'dynamic':<10}  {'Δ':<10}")
    print(f"  {'─' * 14}  {'─' * 22}  {'─' * 10}  {'─' * 10}")
    static_labels = ["vector", "text", "hybrid α=0.5", "hybrid α=0.8"]
    for ds in datasets:
        ds_metrics = results.get(ds, {})
        statics = [(l, ds_metrics.get(l, {}).get("NDCG@10", 0.0)) for l in static_labels]
        statics = [s for s in statics if s[1] > 0]
        if not statics:
            continue
        best_static = max(statics, key=lambda x: x[1])
        dyn = ds_metrics.get("dynamic", {}).get("NDCG@10", 0.0)
        delta = dyn - best_static[1]
        marker = "✓" if delta >= -0.005 else " "  # within noise = OK
        print(f"  {ds:<14}  {best_static[0]:<14} {best_static[1]:>5.3f}   "
              f"{dyn:>5.3f}      {delta:>+5.3f}  {marker}")
    print()


def summary(results: dict, datasets: list[str], metric: str) -> None:
    """Print a per-dataset × per-strategy table for one metric."""
    print()
    print("═" * 96)
    print(f"  {metric}")
    print("═" * 96)
    print()
    labels = [l for l, _ in STRATEGIES]
    name_w = max(len(d) for d in datasets) + 2
    label_w = max(len(l) for l in labels) + 2

    print(f"  {'Dataset':<{name_w}}", end="")
    for l in labels:
        print(f"  {l:>{label_w}}", end="")
    print()
    print(f"  {'─' * name_w}", end="")
    for l in labels:
        print(f"  {'─' * label_w}", end="")
    print()

    for ds in datasets:
        ds_metrics = results.get(ds, {})
        print(f"  {ds:<{name_w}}", end="")
        for l in labels:
            v = ds_metrics.get(l, {}).get(metric, None)
            print(f"  {(f'{v:.3f}' if v is not None else '   -  '):>{label_w}}", end="")
        print()


if __name__ == "__main__":
    main()
