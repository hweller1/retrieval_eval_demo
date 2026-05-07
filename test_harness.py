"""
End-to-end test harness for ingest.py + query.py
=================================================

For each supported BEIR dataset:
  1. Calls ingest.ingest(dataset, corpus_sample=SAMPLE)
  2. Verifies the MongoDB collection is populated and the vector index exists
  3. Calls query.query(dataset, num_queries=NUM_QUERIES, verbose=False) and
     reads the returned RunResult — no stdout scraping
  4. Asserts the run produced reasonable retrieval (≥1 hit somewhere)

Defaults are tuned for fast iteration. Use --quick for the smallest datasets
or --datasets to limit the run.

Usage:
  python3 test_harness.py                    # all 8 datasets, sample=200
  python3 test_harness.py --quick            # 3 small datasets
  python3 test_harness.py --datasets scifact nfcorpus
  python3 test_harness.py --sample 500 --num-queries 10
"""

from __future__ import annotations

import io
import os
import sys
import time
import argparse
import contextlib
import traceback
from dataclasses import dataclass, field

# Quiet down BEIR / tqdm noise so the harness output stays readable
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import lib
import ingest as ingest_mod
import query  as query_mod
from lib import DATASETS, DB_NAME, INDEX_NAME, collection_name


QUICK_DATASETS = ["scifact", "nfcorpus", "arguana"]


@dataclass
class StageResult:
    name: str
    passed: bool
    duration_s: float
    detail: str = ""


@dataclass
class DatasetResult:
    dataset: str
    ingest: StageResult
    query : StageResult | None
    chunks_in_collection: int = 0
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def overall_passed(self) -> bool:
        return self.ingest.passed and (self.query is not None and self.query.passed)


def run_stage(label: str, fn) -> tuple[StageResult, str, object]:
    """
    Run fn() while capturing stdout. Returns (StageResult, captured_output, return_value).
    """
    buf = io.StringIO()
    t0 = time.time()
    passed, detail = True, ""
    return_val = None
    try:
        with contextlib.redirect_stdout(buf):
            return_val = fn()
    except SystemExit as e:
        passed = (e.code in (None, 0))
        detail = f"SystemExit({e.code})"
    except Exception as e:
        passed = False
        detail = f"{type(e).__name__}: {e}"
        traceback.print_exc(file=buf)
    return StageResult(label, passed, time.time() - t0, detail), buf.getvalue(), return_val


def verify_collection(dataset: str) -> tuple[int, bool, str]:
    import pymongo
    if not lib.MONGODB_URI:
        return 0, False, "MONGODB_URI not set"

    client = pymongo.MongoClient(lib.MONGODB_URI)
    try:
        coll = client[DB_NAME][collection_name(dataset)]
        count = coll.estimated_document_count()
        if count == 0:
            return 0, False, "collection empty"

        sample = coll.find_one({"embedding": {"$exists": True}})
        if not sample:
            return count, False, "no embeddings stored"

        indexes = list(coll.list_search_indexes())
        if not any(idx["name"] == INDEX_NAME for idx in indexes):
            return count, False, f"index '{INDEX_NAME}' missing"

        return count, True, ""
    finally:
        client.close()


def run_dataset(dataset: str, sample: int, num_queries: int, verbose: bool) -> DatasetResult:
    print(f"\n  ── {dataset} ────────────────────────────────────────────")

    # --- ingest ---
    print(f"    [ingest] sample={sample} …", end=" ", flush=True)
    ingest_result, ingest_output, _ = run_stage(
        "ingest", lambda: ingest_mod.ingest(dataset, corpus_sample=sample)
    )
    print(f"{ingest_result.duration_s:.1f}s "
          f"{'PASS' if ingest_result.passed else 'FAIL'}")
    if not ingest_result.passed:
        print(f"      ↳ {ingest_result.detail}")
        if verbose:
            print(ingest_output)
        return DatasetResult(dataset=dataset, ingest=ingest_result, query=None)

    # --- verify mongo state ---
    chunks, ok, why = verify_collection(dataset)
    if not ok:
        ingest_result.passed = False
        ingest_result.detail = f"verify failed: {why}"
        print(f"      ↳ verify FAIL: {why}")
        return DatasetResult(dataset=dataset, ingest=ingest_result, query=None,
                             chunks_in_collection=chunks)
    print(f"      ↳ {chunks:,} chunks, index queryable")

    # --- query ---
    print(f"    [query]  num_queries={num_queries} …", end=" ", flush=True)
    query_result, query_output, run_result = run_stage(
        "query",
        lambda: query_mod.query(dataset, num_queries=num_queries, verbose=False),
    )

    metrics = run_result.aggregate if run_result is not None else {}
    map_score = metrics.get("MAP", 0.0)
    ndcg10    = metrics.get("NDCG@10", 0.0)
    p_at_5    = metrics.get("P@5", 0.0)

    print(f"{query_result.duration_s:.1f}s "
          f"{'PASS' if query_result.passed else 'FAIL'}  "
          f"MAP={map_score:.3f}  NDCG@10={ndcg10:.3f}  P@5={p_at_5:.3f}")

    if not query_result.passed:
        print(f"      ↳ {query_result.detail}")
        if verbose:
            print(query_output)
    elif metrics and map_score == 0.0 and p_at_5 == 0.0:
        query_result.passed = False
        query_result.detail = "all metrics zero — no relevant docs retrieved"
        print(f"      ↳ {query_result.detail}")

    return DatasetResult(
        dataset=dataset, ingest=ingest_result, query=query_result,
        chunks_in_collection=chunks, metrics=metrics,
    )


def _bar(value: float, width: int = 24, vmax: float = 1.0) -> str:
    """Render a Unicode bar of `width` cells; value clamped to [0, vmax]."""
    if vmax <= 0:
        return " " * width
    fill = max(0.0, min(value / vmax, 1.0)) * width
    full_blocks = int(fill)
    remainder   = fill - full_blocks
    # Fractional block: 1/8 → 7/8 of █
    eighths = " ▏▎▍▌▋▊▉█"
    partial = eighths[round(remainder * 8)] if full_blocks < width else ""
    return ("█" * full_blocks + partial).ljust(width)


def print_charts(results: list[DatasetResult]) -> None:
    """Per-metric bar charts comparing every dataset side-by-side."""
    passed = [r for r in results if r.overall_passed and r.metrics]
    if not passed:
        return

    metrics = ["NDCG@10", "MAP", "MRR", "R@5", "NDCG@5", "P@5"]
    name_w  = max(len(r.dataset) for r in passed)

    print("═" * 96)
    print("  Comparison")
    print("═" * 96)
    for metric in metrics:
        print()
        print(f"  {metric}")
        print(f"  {'─' * (name_w + 32)}")
        for r in passed:
            v = r.metrics.get(metric, 0.0)
            bar = _bar(v, width=24, vmax=1.0)
            print(f"  {r.dataset:<{name_w}}  {bar}  {v:.3f}")
    print()


def print_summary(results: list[DatasetResult]) -> None:
    print()
    print("═" * 96)
    print("  Test Harness Summary")
    print("═" * 96)
    print()
    cols = ["P@5", "R@5", "NDCG@5", "NDCG@10", "MRR", "MAP"]
    print(f"  {'Dataset':<14} {'Ingest':>7} {'Query':>7} {'Chunks':>8} "
          + " ".join(f"{c:>8}" for c in cols)
          + "  Status")
    print(f"  {'─' * 14} {'─' * 7:>7} {'─' * 7:>7} {'─' * 8:>8} "
          + " ".join(f"{'─' * 8:>8}" for _ in cols)
          + "  ──────")

    pass_count = 0
    for r in results:
        status = "PASS" if r.overall_passed else "FAIL"
        if r.overall_passed:
            pass_count += 1
        ing = f"{r.ingest.duration_s:.1f}s" if r.ingest else "-"
        qry = f"{r.query.duration_s:.1f}s" if r.query else "-"
        metric_strs = [f"{r.metrics.get(c, 0.0):>8.3f}" for c in cols]
        print(f"  {r.dataset:<14} {ing:>7} {qry:>7} {r.chunks_in_collection:>8,} "
              + " ".join(metric_strs)
              + f"  {status}")

    print()
    print(f"  {pass_count}/{len(results)} datasets passed")
    print()
    print_charts(results)


def write_markdown_report(results: list[DatasetResult], path: str) -> None:
    cols = ["P@5", "R@5", "NDCG@5", "NDCG@10", "MRR", "MAP"]
    lines = [
        "# voyage-context-3 BEIR comparison",
        "",
        "Per-dataset retrieval metrics from `test_harness.py`. "
        "Bars are scaled 0 → 1.0 across all metrics.",
        "",
        "| Dataset | Chunks | " + " | ".join(cols) + " |",
        "|---|---|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for r in results:
        if not r.overall_passed:
            continue
        cells = [f"{r.metrics.get(c, 0.0):.3f}" for c in cols]
        lines.append(f"| {r.dataset} | {r.chunks_in_collection:,} | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("## Per-metric comparison")
    lines.append("")
    for metric in ["NDCG@10", "MAP", "MRR", "R@5", "NDCG@5", "P@5"]:
        lines.append(f"### {metric}")
        lines.append("")
        lines.append("```")
        passed = [r for r in results if r.overall_passed]
        name_w = max((len(r.dataset) for r in passed), default=10)
        for r in passed:
            v = r.metrics.get(metric, 0.0)
            lines.append(f"  {r.dataset:<{name_w}}  {_bar(v, width=30, vmax=1.0)}  {v:.3f}")
        lines.append("```")
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Wrote markdown report to {path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="End-to-end test harness for ingest.py + query.py.")
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS.keys()),
                   help="subset of datasets to test (default: all)")
    p.add_argument("--quick", action="store_true",
                   help=f"test only small datasets: {', '.join(QUICK_DATASETS)}")
    p.add_argument("--sample", type=int, default=200,
                   help="docs to ingest per dataset (default: 200)")
    p.add_argument("--num-queries", type=int, default=3,
                   help="queries to run per dataset (default: 3)")
    p.add_argument("--verbose", action="store_true",
                   help="print full output on failure")
    p.add_argument("--report", metavar="PATH",
                   help="also write a Markdown comparison report to PATH")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.datasets:
        datasets = args.datasets
    elif args.quick:
        datasets = QUICK_DATASETS
    else:
        datasets = list(DATASETS.keys())

    print("═" * 96)
    print("  voyage-context-3 Test Harness")
    print("═" * 96)
    print(f"  Datasets   : {', '.join(datasets)}")
    print(f"  Sample/ds  : {args.sample}")
    print(f"  Queries/ds : {args.num_queries}")

    overall_t0 = time.time()
    results: list[DatasetResult] = []
    for dataset in datasets:
        results.append(run_dataset(
            dataset, args.sample, args.num_queries, verbose=args.verbose,
        ))

    print(f"\n  Total elapsed: {time.time() - overall_t0:.1f}s")
    print_summary(results)

    if args.report:
        write_markdown_report(results, args.report)

    failures = sum(1 for r in results if not r.overall_passed)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
