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
from query_rewriter import REWRITERS


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
    rewriter: str
    stage: StageResult
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def label(self) -> str:
        """Display label like 'hybrid' or 'hybrid+hyde'."""
        return self.mode if self.rewriter == "none" else f"{self.mode}+{self.rewriter}"


@dataclass
class DatasetResult:
    dataset: str
    ingest: StageResult
    chunks_in_collection: int = 0
    runs: list[ModeRun] = field(default_factory=list)

    def find(self, mode: str, rewriter: str) -> ModeRun | None:
        for r in self.runs:
            if r.mode == mode and r.rewriter == rewriter:
                return r
        return None

    @property
    def overall_passed(self) -> bool:
        return self.ingest.passed and bool(self.runs) and all(
            r.stage.passed for r in self.runs
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
                modes: list[str], rewriters: list[str], verbose: bool) -> DatasetResult:
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

    # --- query for each (mode, rewriter) ---
    runs: list[ModeRun] = []
    label_w = max(
        len(m if r == "none" else f"{m}+{r}")
        for m in modes for r in rewriters
    )
    for mode in modes:
        for rewriter in rewriters:
            label = mode if rewriter == "none" else f"{mode}+{rewriter}"
            print(f"    [query/{label:<{label_w}}] num_queries={num_queries} …", end=" ", flush=True)
            stage, qout, run = run_stage(
                f"query-{label}",
                lambda m=mode, r=rewriter: query_mod.query(
                    dataset, num_queries=num_queries, mode=m, rewriter=r, verbose=False,
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
            runs.append(ModeRun(mode=mode, rewriter=rewriter, stage=stage, metrics=metrics))

    return DatasetResult(
        dataset=dataset, ingest=ingest_result,
        chunks_in_collection=chunks, runs=runs,
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


# Glyph palette for comparison bars — cycles through these in order.
GLYPH_PALETTE = ("█", "▓", "▒", "░", "▌", "▍", "▎", "▏")


def _bar_glyph(value: float, glyph: str, width: int = 24, vmax: float = 1.0) -> str:
    fill = max(0.0, min(value / vmax, 1.0)) * width
    return (glyph * int(fill)).ljust(width)


def _strategies(modes: list[str], rewriters: list[str]) -> list[tuple[str, str]]:
    """Cartesian product (mode, rewriter), preserving caller order."""
    return [(m, r) for m in modes for r in rewriters]


def _strategy_label(mode: str, rewriter: str) -> str:
    return mode if rewriter == "none" else f"{mode}+{rewriter}"


def print_summary(results: list[DatasetResult], modes: list[str], rewriters: list[str]) -> None:
    print()
    print("═" * 110)
    print("  Test Harness Summary")
    print("═" * 110)
    print()
    cols = ["P@5", "R@5", "NDCG@5", "NDCG@10", "MRR", "MAP"]
    strats = _strategies(modes, rewriters)
    label_w = max(len(_strategy_label(*s)) for s in strats)

    print(f"  {'Dataset':<14} {'Strategy':<{label_w}} {'Query':>7} "
          + " ".join(f"{c:>8}" for c in cols)
          + "  Status")
    print(f"  {'─' * 14} {'─' * label_w:<{label_w}} {'─' * 7:>7} "
          + " ".join(f"{'─' * 8:>8}" for _ in cols)
          + "  ──────")

    pass_count = 0
    total      = 0
    for r in results:
        for mode, rewriter in strats:
            mr = r.find(mode, rewriter)
            total += 1
            label = _strategy_label(mode, rewriter)
            if not mr:
                print(f"  {r.dataset:<14} {label:<{label_w}} {'-':>7} "
                      + " ".join(f"{'-':>8}" for _ in cols)
                      + "  SKIP")
                continue
            status = "PASS" if mr.stage.passed else "FAIL"
            if mr.stage.passed:
                pass_count += 1
            qry = f"{mr.stage.duration_s:.1f}s"
            metric_strs = [f"{mr.metrics.get(c, 0.0):>8.3f}" for c in cols]
            print(f"  {r.dataset:<14} {label:<{label_w}} {qry:>7} "
                  + " ".join(metric_strs)
                  + f"  {status}")
        print()

    print(f"  {pass_count}/{total} (dataset × strategy) runs passed")
    print()


def print_comparison_charts(results: list[DatasetResult],
                            modes: list[str], rewriters: list[str]) -> None:
    """Per-metric bar chart with one bar per (mode × rewriter), grouped by dataset."""
    passed = [r for r in results if r.runs]
    if not passed:
        return

    strats   = _strategies(modes, rewriters)
    glyphs   = {s: GLYPH_PALETTE[i % len(GLYPH_PALETTE)] for i, s in enumerate(strats)}
    label_w  = max(len(_strategy_label(*s)) for s in strats)
    name_w   = max(len(r.dataset) for r in passed)

    print("═" * 110)
    print("  Strategy Comparison")
    print("═" * 110)
    legend = "  ".join(f"{glyphs[s]} {_strategy_label(*s)}" for s in strats)
    print(f"  legend: {legend}")

    for metric in ["NDCG@10", "MAP", "MRR", "R@5", "NDCG@5", "P@5"]:
        print()
        print(f"  {metric}")
        print(f"  {'─' * (name_w + label_w + 36)}")
        for r in passed:
            for i, (mode, rewriter) in enumerate(strats):
                mr = r.find(mode, rewriter)
                v = mr.metrics.get(metric, 0.0) if mr and mr.stage.passed else 0.0
                glyph = glyphs[(mode, rewriter)]
                label = _strategy_label(mode, rewriter)
                ds_label = r.dataset if i == 0 else ""
                print(f"  {ds_label:<{name_w}}  {label:<{label_w}}  "
                      f"{_bar_glyph(v, glyph, width=24, vmax=1.0)}  {v:.3f}")
            print(f"  {' ' * name_w}")
    print()


def print_deltas(results: list[DatasetResult], baseline: tuple[str, str] = ("vector", "none")) -> None:
    """Per-dataset table: NDCG@10 / MAP delta vs baseline (mode, rewriter)."""
    base_mode, base_rewriter = baseline
    relevant = [r for r in results if r.find(base_mode, base_rewriter)]
    if not relevant:
        return

    # All strategies actually present in the runs, minus the baseline
    seen_strats: list[tuple[str, str]] = []
    for r in relevant:
        for run in r.runs:
            s = (run.mode, run.rewriter)
            if s != baseline and s not in seen_strats:
                seen_strats.append(s)
    if not seen_strats:
        return

    base_label = _strategy_label(*baseline)
    print("═" * 110)
    print(f"  Δ vs {base_label} (NDCG@10 and MAP)")
    print("═" * 110)
    name_w = max(len(r.dataset) for r in relevant)
    label_w = max(len(_strategy_label(*s)) for s in seen_strats)

    header_cells = []
    for s in seen_strats:
        lab = _strategy_label(*s)
        header_cells.extend([f"{lab}-NDCG10", "Δ", f"{lab}-MAP", "Δ"])
    cell_w = max(11, label_w + 7)
    print(f"  {'Dataset':<{name_w}}  {f'{base_label}-NDCG10':>{cell_w}}  "
          f"{f'{base_label}-MAP':>{cell_w}}  "
          + "  ".join(f"{c:>{cell_w}}" for c in header_cells))
    for r in relevant:
        base = r.find(*baseline).metrics
        row = [f"  {r.dataset:<{name_w}}",
               f"{base.get('NDCG@10', 0.0):>{cell_w}.3f}",
               f"{base.get('MAP', 0.0):>{cell_w}.3f}"]
        for s in seen_strats:
            mm = r.find(*s)
            if mm and mm.metrics and mm.stage.passed:
                d_n = mm.metrics.get("NDCG@10", 0.0) - base.get("NDCG@10", 0.0)
                d_m = mm.metrics.get("MAP", 0.0)     - base.get("MAP", 0.0)
                row += [f"{mm.metrics.get('NDCG@10', 0.0):>{cell_w}.3f}",
                        f"{d_n:>+{cell_w}.3f}",
                        f"{mm.metrics.get('MAP', 0.0):>{cell_w}.3f}",
                        f"{d_m:>+{cell_w}.3f}"]
            else:
                row += [f"{'-':>{cell_w}}"] * 4
        print("  ".join(row))
    print()


# ── Markdown report ──────────────────────────────────────────────────────────

def write_markdown_report(results: list[DatasetResult],
                          modes: list[str], rewriters: list[str], path: str) -> None:
    cols = ["P@5", "R@5", "NDCG@5", "NDCG@10", "MRR", "MAP"]
    strats = _strategies(modes, rewriters)
    strat_labels = [_strategy_label(*s) for s in strats]

    lines = [
        "# voyage-context-3 retrieval comparison",
        "",
        "Per-dataset metrics from `test_harness.py`.",
        "",
        f"- Modes: **{', '.join(modes)}**",
        f"- Rewriters: **{', '.join(rewriters)}**",
        "",
    ]

    for metric in ["NDCG@10", "MAP", "MRR"]:
        lines.append(f"## {metric}")
        lines.append("")
        lines.append("| Dataset | " + " | ".join(strat_labels) + " |")
        lines.append("|---|" + "|".join(["---"] * len(strats)) + "|")
        for r in results:
            cells = []
            for mode, rewriter in strats:
                mr = r.find(mode, rewriter)
                cells.append(f"{mr.metrics.get(metric, 0.0):.3f}" if mr else "-")
            lines.append(f"| {r.dataset} | " + " | ".join(cells) + " |")
        lines.append("")

    lines.append("## Full metric table")
    lines.append("")
    lines.append("| Dataset | Strategy | " + " | ".join(cols) + " |")
    lines.append("|---|---|" + "|".join(["---"] * len(cols)) + "|")
    for r in results:
        for mode, rewriter in strats:
            mr = r.find(mode, rewriter)
            if not mr:
                continue
            cells = [f"{mr.metrics.get(c, 0.0):.3f}" for c in cols]
            lines.append(f"| {r.dataset} | {_strategy_label(mode, rewriter)} | "
                         + " | ".join(cells) + " |")
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
    p.add_argument("--rewriters", nargs="+", choices=list(REWRITERS), default=["none"],
                   help="query rewriters to compare (default: none). "
                        "Anything other than 'none' requires OPENAI_API_KEY.")
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

    n_combos = len(datasets) * len(args.modes) * len(args.rewriters)
    print("═" * 110)
    print("  voyage-context-3 Test Harness")
    print("═" * 110)
    print(f"  Datasets   : {', '.join(datasets)}")
    print(f"  Modes      : {', '.join(args.modes)}")
    print(f"  Rewriters  : {', '.join(args.rewriters)}")
    print(f"  Sample/ds  : {args.sample}")
    print(f"  Queries/ds : {args.num_queries}")
    print(f"  → {n_combos} dataset×mode×rewriter combinations")

    overall_t0 = time.time()
    results: list[DatasetResult] = []
    for dataset in datasets:
        results.append(run_dataset(
            dataset, args.sample, args.num_queries,
            args.modes, args.rewriters, verbose=args.verbose,
        ))

    print(f"\n  Total elapsed: {time.time() - overall_t0:.1f}s")
    print_summary(results, args.modes, args.rewriters)
    print_deltas(results, baseline=("vector", "none"))
    print_comparison_charts(results, args.modes, args.rewriters)

    if args.report:
        write_markdown_report(results, args.modes, args.rewriters, args.report)

    failures = sum(1 for r in results if not r.overall_passed)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
