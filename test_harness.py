"""
End-to-end test harness for demo.py
====================================

Validates that --ingest and --query work for every supported BEIR dataset:

  1. Calls demo.cmd_ingest(dataset, corpus_sample=SAMPLE) for each dataset.
  2. Calls demo.cmd_query(dataset, num_queries=NUM_QUERIES).
  3. Verifies each step produced the expected MongoDB state and returned
     non-trivial retrieval metrics (at least one hit, MAP > 0).
  4. Prints a PASS/FAIL summary table.

Defaults are tuned for fast iteration (small sample, few queries) so the
full suite finishes in a few minutes. Use --quick to test only the smallest
datasets, or --datasets to limit the run.

Usage:
  python3 test_harness.py                    # all 8 datasets, sample=200
  python3 test_harness.py --quick            # only the small datasets
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
from dataclasses import dataclass

# Quiet down BEIR / tqdm noise so the harness output stays readable
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import demo
from demo import DATASETS, DB_NAME, INDEX_NAME, collection_name


# Datasets considered "quick" — small corpora that download/ingest in seconds
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
    map_score: float = 0.0
    mean_p_at_k: float = 0.0

    @property
    def overall_passed(self) -> bool:
        return self.ingest.passed and (self.query is not None and self.query.passed)


def parse_metrics(captured: str) -> tuple[float, float]:
    """Pull MAP and Mean P@K from the demo's stdout."""
    map_score, p_at_k = 0.0, 0.0
    for line in captured.splitlines():
        line = line.strip()
        if line.startswith("MAP"):
            try:
                map_score = float(line.split(":")[1].strip())
            except (IndexError, ValueError):
                pass
        elif line.startswith("Mean P@"):
            try:
                p_at_k = float(line.split(":")[1].strip())
            except (IndexError, ValueError):
                pass
    return map_score, p_at_k


def run_stage(label: str, fn) -> tuple[StageResult, str]:
    """Run fn() while capturing stdout. Returns (StageResult, captured_output)."""
    buf = io.StringIO()
    t0 = time.time()
    passed, detail = True, ""
    try:
        with contextlib.redirect_stdout(buf):
            fn()
    except SystemExit as e:
        passed = (e.code in (None, 0))
        detail = f"SystemExit({e.code})"
    except Exception as e:
        passed = False
        detail = f"{type(e).__name__}: {e}"
        traceback.print_exc(file=buf)
    return StageResult(label, passed, time.time() - t0, detail), buf.getvalue()


def verify_collection(dataset: str) -> tuple[int, bool, str]:
    """Confirm the collection exists, has chunks, and the vector index is queryable."""
    import pymongo
    if not demo.MONGODB_URI:
        return 0, False, "MONGODB_URI not set"

    client = pymongo.MongoClient(demo.MONGODB_URI)
    try:
        coll = client[DB_NAME][collection_name(dataset)]
        count = coll.estimated_document_count()
        if count == 0:
            return 0, False, "collection empty"

        # Check at least one document has an embedding field
        sample = coll.find_one({"embedding": {"$exists": True}})
        if not sample:
            return count, False, "no embeddings stored"

        # Check the index exists
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
    ingest_result, ingest_output = run_stage(
        "ingest", lambda: demo.cmd_ingest(dataset, corpus_sample=sample)
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
    query_result, query_output = run_stage(
        "query", lambda: demo.cmd_query(dataset, num_queries=num_queries)
    )
    map_score, p_at_k = parse_metrics(query_output)
    print(f"{query_result.duration_s:.1f}s "
          f"{'PASS' if query_result.passed else 'FAIL'}  "
          f"MAP={map_score:.3f}  P@5={p_at_k:.3f}")

    if not query_result.passed:
        print(f"      ↳ {query_result.detail}")
        if verbose:
            print(query_output)
    elif map_score == 0.0 and p_at_k == 0.0:
        # Sanity check: no relevant hits at all is suspicious
        query_result.passed = False
        query_result.detail = "MAP=0 and P@K=0 — no relevant docs retrieved"
        print(f"      ↳ {query_result.detail}")

    return DatasetResult(
        dataset=dataset, ingest=ingest_result, query=query_result,
        chunks_in_collection=chunks, map_score=map_score, mean_p_at_k=p_at_k,
    )


def print_summary(results: list[DatasetResult]) -> None:
    print()
    print("═" * 78)
    print("  Test Harness Summary")
    print("═" * 78)
    print()
    print(f"  {'Dataset':<14} {'Ingest':>10} {'Query':>10} {'Chunks':>10} "
          f"{'MAP':>7} {'P@5':>7}  Status")
    print(f"  {'─' * 14} {'─' * 10:>10} {'─' * 10:>10} {'─' * 10:>10} "
          f"{'─' * 7:>7} {'─' * 7:>7}  ──────")

    pass_count = 0
    for r in results:
        status = "PASS" if r.overall_passed else "FAIL"
        if r.overall_passed:
            pass_count += 1
        ing = f"{r.ingest.duration_s:.1f}s" if r.ingest else "-"
        qry = f"{r.query.duration_s:.1f}s" if r.query else "-"
        print(f"  {r.dataset:<14} {ing:>10} {qry:>10} "
              f"{r.chunks_in_collection:>10,} "
              f"{r.map_score:>7.3f} {r.mean_p_at_k:>7.3f}  {status}")

    print()
    print(f"  {pass_count}/{len(results)} datasets passed")
    print()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="End-to-end test harness for demo.py.")
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS.keys()),
                   help="subset of datasets to test (default: all)")
    p.add_argument("--quick", action="store_true",
                   help=f"test only small datasets: {', '.join(QUICK_DATASETS)}")
    p.add_argument("--sample", type=int, default=200,
                   help="docs to ingest per dataset (default: 200)")
    p.add_argument("--num-queries", type=int, default=3,
                   help="queries to run per dataset (default: 3)")
    p.add_argument("--verbose", action="store_true",
                   help="print full demo output on failure")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.datasets:
        datasets = args.datasets
    elif args.quick:
        datasets = QUICK_DATASETS
    else:
        datasets = list(DATASETS.keys())

    print("═" * 78)
    print("  voyage-context-3 Test Harness")
    print("═" * 78)
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

    failures = sum(1 for r in results if not r.overall_passed)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
