"""
End-to-end test harness for ingest.py + query.py
=================================================

For each supported BEIR dataset:
  1. Calls ingest.ingest(dataset, corpus_sample=SAMPLE)
  2. Verifies the MongoDB collection is populated and both indexes exist
  3. Calls query.query(dataset, mode=m) for each retrieval mode (vector,
     text, hybrid) and reads the returned RunResult
  4. Asserts each mode produced reasonable retrieval

Output sections:
  * Summary table — one row per (dataset × mode), key metrics
  * Comparison charts — bar chart per metric, grouped by dataset, with
    a bar per mode side-by-side. This is the visual showing whether
    hybrid beats vector and by how much.
  * Optional Markdown report (--report PATH)

Defaults are tuned for fast iteration. Use --quick for the smallest
datasets or --datasets to limit the run.

Usage:
  python3 test_harness.py                              # all datasets, all modes
  python3 test_harness.py --quick                       # 3 small datasets
  python3 test_harness.py --datasets scifact nfcorpus
  python3 test_harness.py --modes vector hybrid         # subset of modes
  python3 test_harness.py --quick --report report.md
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
from retrieve import MODES, TEXT_INDEX_NAME


QUICK_DATASETS = ["scifact", "nfcorpus", "arguana"]


@dataclass
class StageResult:
    name: str
    passed: bool
    duration_s: float
    detail: str = ""


@dataclass
class ModeRun:
    mode: str
    stage: StageResult
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class DatasetResult:
    dataset: str
    ingest: StageResult
    chunks_in_collection: int = 0
    by_mode: dict[str, ModeRun] = field(default_factory=dict)

    @property
    def overall_passed(self) -> bool:
        return self.ingest.passed and bool(self.by_mode) and all(
            mr.stage.passed for mr in self.by_mode.values()
        )


# ── Stage runner ─────────────────────────────────────────────────────────────

def run_stage(label: str, fn) -> tuple[StageResult, str, object]:
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
        if not coll.find_one({"embedding": {"$exists": True}}):
            return count, False, "no embeddings stored"
        names = {idx["name"] for idx in coll.list_search_indexes()}
        for needed in (INDEX_NAME, TEXT_INDEX_NAME):
            if needed not in names:
                return count, False, f"index '{needed}' missing"
        return count, True, ""
    finally:
        client.close()


def run_dataset(dataset: str, sample: int, num_queries: int,
                modes: list[str], verbose: bool) -> DatasetResult:
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
        return DatasetResult(dataset=dataset, ingest=ingest_result)

    chunks, ok, why = verify_collection(dataset)
    if not ok:
        ingest_result.passed = False
        ingest_result.detail = f"verify failed: {why}"
        print(f"      ↳ verify FAIL: {why}")
        return DatasetResult(dataset=dataset, ingest=ingest_result,
                             chunks_in_collection=chunks)
    print(f"      ↳ {chunks:,} chunks; vector + text indexes ready")

    # --- query for each mode ---
    by_mode: dict[str, ModeRun] = {}
    for mode in modes:
        print(f"    [query/{mode:<6}] num_queries={num_queries} …", end=" ", flush=True)
        stage, qout, run = run_stage(
            f"query-{mode}",
            lambda m=mode: query_mod.query(
                dataset, num_queries=num_queries, mode=m, verbose=False,
            ),
        )
        metrics = run.aggregate if run is not None else {}
        ndcg10  = metrics.get("NDCG@10", 0.0)
        map_s   = metrics.get("MAP", 0.0)
        print(f"{stage.duration_s:.1f}s "
              f"{'PASS' if stage.passed else 'FAIL'}  "
              f"NDCG@10={ndcg10:.3f}  MAP={map_s:.3f}")
        if not stage.passed:
            print(f"      ↳ {stage.detail}")
            if verbose:
                print(qout)
        elif metrics and ndcg10 == 0.0 and map_s == 0.0:
            stage.passed = False
            stage.detail = "all metrics zero — no relevant docs retrieved"
            print(f"      ↳ {stage.detail}")
        by_mode[mode] = ModeRun(mode=mode, stage=stage, metrics=metrics)

    return DatasetResult(
        dataset=dataset, ingest=ingest_result,
        chunks_in_collection=chunks, by_mode=by_mode,
    )


# ── Visual rendering ─────────────────────────────────────────────────────────

def _bar(value: float, width: int = 24, vmax: float = 1.0) -> str:
    if vmax <= 0:
        return " " * width
    fill = max(0.0, min(value / vmax, 1.0)) * width
    full_blocks = int(fill)
    remainder   = fill - full_blocks
    eighths = " ▏▎▍▌▋▊▉█"
    partial = eighths[round(remainder * 8)] if full_blocks < width else ""
    return ("█" * full_blocks + partial).ljust(width)


# Per-mode glyphs/colors for the comparison chart.
MODE_GLYPH = {"vector": "█", "text": "▒", "hybrid": "▓"}


def _bar_glyph(value: float, glyph: str, width: int = 24, vmax: float = 1.0) -> str:
    fill = max(0.0, min(value / vmax, 1.0)) * width
    return (glyph * int(fill)).ljust(width)


def print_summary(results: list[DatasetResult], modes: list[str]) -> None:
    print()
    print("═" * 104)
    print("  Test Harness Summary")
    print("═" * 104)
    print()
    cols = ["P@5", "R@5", "NDCG@5", "NDCG@10", "MRR", "MAP"]
    print(f"  {'Dataset':<14} {'Mode':<8} {'Query':>7} "
          + " ".join(f"{c:>8}" for c in cols)
          + "  Status")
    print(f"  {'─' * 14} {'─' * 8:<8} {'─' * 7:>7} "
          + " ".join(f"{'─' * 8:>8}" for _ in cols)
          + "  ──────")

    pass_count = 0
    total      = 0
    for r in results:
        for mode in modes:
            mr = r.by_mode.get(mode)
            total += 1
            if not mr:
                print(f"  {r.dataset:<14} {mode:<8} {'-':>7} "
                      + " ".join(f"{'-':>8}" for _ in cols)
                      + "  SKIP")
                continue
            status = "PASS" if mr.stage.passed else "FAIL"
            if mr.stage.passed:
                pass_count += 1
            qry = f"{mr.stage.duration_s:.1f}s"
            metric_strs = [f"{mr.metrics.get(c, 0.0):>8.3f}" for c in cols]
            print(f"  {r.dataset:<14} {mode:<8} {qry:>7} "
                  + " ".join(metric_strs)
                  + f"  {status}")
        print()

    print(f"  {pass_count}/{total} (dataset × mode) runs passed")
    print()


def print_comparison_charts(results: list[DatasetResult], modes: list[str]) -> None:
    """Per-metric bar chart with one bar per mode, grouped by dataset."""
    passed = [r for r in results if r.by_mode]
    if not passed:
        return

    print("═" * 104)
    print("  Mode Comparison")
    print("═" * 104)
    legend = "  ".join(f"{MODE_GLYPH[m]} {m}" for m in modes if m in MODE_GLYPH)
    print(f"  legend: {legend}")
    name_w = max(len(r.dataset) for r in passed)
    mode_w = max(len(m) for m in modes)

    for metric in ["NDCG@10", "MAP", "MRR", "R@5", "NDCG@5", "P@5"]:
        print()
        print(f"  {metric}")
        print(f"  {'─' * (name_w + mode_w + 36)}")
        for r in passed:
            for i, mode in enumerate(modes):
                mr = r.by_mode.get(mode)
                v = mr.metrics.get(metric, 0.0) if mr and mr.stage.passed else 0.0
                glyph = MODE_GLYPH.get(mode, "█")
                ds_label = r.dataset if i == 0 else ""
                print(f"  {ds_label:<{name_w}}  {mode:<{mode_w}}  "
                      f"{_bar_glyph(v, glyph, width=24, vmax=1.0)}  {v:.3f}")
            print(f"  {' ' * name_w}")  # spacer between datasets
    print()


def print_deltas(results: list[DatasetResult], baseline: str = "vector") -> None:
    """Per-dataset table: NDCG@10 / MAP delta vs baseline mode."""
    if baseline not in MODES:
        return
    relevant = [r for r in results if r.by_mode.get(baseline)]
    if not relevant:
        return
    other_modes = [m for m in MODES if m != baseline]
    if not other_modes:
        return

    print("═" * 96)
    print(f"  Δ vs {baseline}-only (NDCG@10 and MAP)")
    print("═" * 96)
    name_w = max(len(r.dataset) for r in relevant)
    header_cells = []
    for m in other_modes:
        header_cells.extend([f"{m}-NDCG10", f"Δ", f"{m}-MAP", f"Δ"])
    print(f"  {'Dataset':<{name_w}}  {f'{baseline}-NDCG10':>14}  {f'{baseline}-MAP':>11}  "
          + "  ".join(f"{c:>10}" for c in header_cells))
    for r in relevant:
        base = r.by_mode[baseline].metrics
        row = [f"  {r.dataset:<{name_w}}",
               f"{base.get('NDCG@10', 0.0):>14.3f}",
               f"{base.get('MAP', 0.0):>11.3f}"]
        for m in other_modes:
            mm = r.by_mode.get(m)
            if mm and mm.metrics:
                d_n = mm.metrics.get("NDCG@10", 0.0) - base.get("NDCG@10", 0.0)
                d_m = mm.metrics.get("MAP", 0.0)     - base.get("MAP", 0.0)
                row += [f"{mm.metrics.get('NDCG@10', 0.0):>10.3f}",
                        f"{d_n:>+10.3f}",
                        f"{mm.metrics.get('MAP', 0.0):>10.3f}",
                        f"{d_m:>+10.3f}"]
            else:
                row += [f"{'-':>10}"] * 4
        print("  ".join(row))
    print()


# ── Markdown report ──────────────────────────────────────────────────────────

def write_markdown_report(results: list[DatasetResult], modes: list[str], path: str) -> None:
    cols = ["P@5", "R@5", "NDCG@5", "NDCG@10", "MRR", "MAP"]
    lines = [
        "# voyage-context-3 BEIR comparison",
        "",
        "Per-dataset retrieval metrics from `test_harness.py`. Modes compared:",
        "**" + ", ".join(modes) + "**.",
        "",
    ]

    for metric in ["NDCG@10", "MAP", "MRR"]:
        lines.append(f"## {metric}")
        lines.append("")
        lines.append("| Dataset | " + " | ".join(modes) + " |")
        lines.append("|---|" + "|".join(["---"] * len(modes)) + "|")
        for r in results:
            cells = [
                f"{r.by_mode[m].metrics.get(metric, 0.0):.3f}" if r.by_mode.get(m) else "-"
                for m in modes
            ]
            lines.append(f"| {r.dataset} | " + " | ".join(cells) + " |")
        lines.append("")

    lines.append("## Full metric table")
    lines.append("")
    lines.append("| Dataset | Mode | " + " | ".join(cols) + " |")
    lines.append("|---|---|" + "|".join(["---"] * len(cols)) + "|")
    for r in results:
        for m in modes:
            mr = r.by_mode.get(m)
            if not mr:
                continue
            cells = [f"{mr.metrics.get(c, 0.0):.3f}" for c in cols]
            lines.append(f"| {r.dataset} | {m} | " + " | ".join(cells) + " |")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Wrote markdown report to {path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="End-to-end test harness for ingest.py + query.py.")
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS.keys()),
                   help="subset of datasets to test (default: all)")
    p.add_argument("--quick", action="store_true",
                   help=f"test only small datasets: {', '.join(QUICK_DATASETS)}")
    p.add_argument("--modes", nargs="+", choices=list(MODES), default=list(MODES),
                   help=f"retrieval modes to compare (default: {' '.join(MODES)})")
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

    print("═" * 104)
    print("  voyage-context-3 Test Harness")
    print("═" * 104)
    print(f"  Datasets   : {', '.join(datasets)}")
    print(f"  Modes      : {', '.join(args.modes)}")
    print(f"  Sample/ds  : {args.sample}")
    print(f"  Queries/ds : {args.num_queries}")

    overall_t0 = time.time()
    results: list[DatasetResult] = []
    for dataset in datasets:
        results.append(run_dataset(
            dataset, args.sample, args.num_queries, args.modes, verbose=args.verbose,
        ))

    print(f"\n  Total elapsed: {time.time() - overall_t0:.1f}s")
    print_summary(results, args.modes)
    print_deltas(results, baseline="vector")
    print_comparison_charts(results, args.modes)

    if args.report:
        write_markdown_report(results, args.modes, args.report)

    failures = sum(1 for r in results if not r.overall_passed)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
