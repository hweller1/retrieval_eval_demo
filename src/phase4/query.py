"""
query.py — embed BEIR queries with voyage-3-large, retrieve from the
ingested MongoDB collection, and report a full IR metric suite (P@K,
R@K, NDCG@K, MRR, MAP) against official relevance judgments.

Three retrieval modes are available:

  vector  : pure $vectorSearch over voyage-context-3 embeddings
  text    : pure $search (BM25) over the chunk text
  hybrid  : weighted Reciprocal Rank Fusion of vector + text

Two strategy modes:

  static   — use the explicit --alpha / --rewriter / --rerank flags
  dynamic  — for each query, a cheap LLM (gpt-4o-mini) decides alpha,
             rerank on/off, and which rewriter (if any) to use. Same
             classifier across all datasets. Requires OPENAI_API_KEY.

Examples:

  python3 query.py --list
  python3 query.py <dataset>                                # default
  python3 query.py <dataset> --mode vector
  python3 query.py <dataset> --strategy dynamic             # per-query routing
  python3 query.py <dataset> --rewriter hyde --rerank       # static + extras

Requires `python3 ingest.py <dataset>` to have been run first.
"""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import textwrap
from dataclasses import dataclass, field

from lib import (
    DEFAULT_QUERIES, DB_NAME, INDEX_NAME, MODEL, MONGODB_BASE_URL,
    MONGODB_URI, QUERY_MODEL, TOP_K,
    DATASETS,
    collection_name, embed_queries, header, load_beir_dataset,
    print_dataset_list, require_credentials, VOYAGE_API_KEY,
)
from lib_metrics import (
    METRIC_KS, aggregate_metrics, compute_query_metrics, format_summary,
)
from retrieve import MODES, DEFAULT_ALPHA, retrieve, multi_query_retrieve
from query_rewriter import REWRITERS, DEFAULT_REWRITER, rewrite
from rerank import rerank as rerank_rows
import query_classifier

DEFAULT_MODE          = "hybrid"
STRATEGY_MODES        = ("static", "dynamic")
DEFAULT_STRATEGY_MODE = "static"
RERANK_CANDIDATES     = 50   # candidates to fetch from first-stage when reranking


@dataclass
class QueryResult:
    qid: str
    text: str
    relevant_count: int
    metrics: dict[str, float]
    top_docs: list[dict] = field(default_factory=list)


@dataclass
class RunResult:
    dataset: str
    mode: str
    rewriter: str            # baseline (when strategy=static) — overridden per-query when dynamic
    rerank: bool
    strategy_mode: str
    num_queries: int
    chunks_in_collection: int
    per_query: list[QueryResult]
    aggregate: dict[str, float]
    # Populated only when strategy_mode == "dynamic": the per-query strategy
    # the classifier picked. Useful for demoing what routing happened.
    per_query_strategies: list = field(default_factory=list)


def _print_query_block(qr: QueryResult) -> None:
    m = qr.metrics
    print(f"\n  Query: \"{qr.text}\"")
    print(f"  {qr.relevant_count} relevant in corpus  |  "
          f"P@5={m['P@5']:.2f}  R@5={m['R@5']:.2f}  "
          f"NDCG@5={m['NDCG@5']:.3f}  MRR={m['MRR']:.3f}  AP={m['AP']:.3f}")
    print(f"  {'─' * 66}")
    for rank, row in enumerate(qr.top_docs[:TOP_K], 1):
        marker = "✓" if row.get("_relevant") else " "
        title  = row["title"] or "(untitled)"
        title  = (title[:55] + "…") if len(title) > 56 else title
        snippet = textwrap.shorten(row["text"].strip(), width=90, placeholder="…")
        print(f"  {rank}. [{marker}] {row['score']:.4f}  {title}")
        print(f"        {snippet}")


def query(
    dataset: str,
    num_queries: int = DEFAULT_QUERIES,
    mode: str = DEFAULT_MODE,
    rewriter: str = DEFAULT_REWRITER,
    rerank: bool = False,
    alpha: float = DEFAULT_ALPHA,
    strategy_mode: str = DEFAULT_STRATEGY_MODE,
    verbose: bool = True,
) -> RunResult:
    """
    Run `num_queries` queries against the ingested collection.

    When `strategy_mode == "static"`: uses the explicit `mode`, `alpha`,
    `rewriter`, `rerank` arguments uniformly across all queries.

    When `strategy_mode == "dynamic"`: a cheap LLM picks per-query alpha,
    rewriter, and rerank flag. The user-supplied `mode` (vector/text/hybrid)
    is preserved as the retrieval backend, but alpha/rewriter/rerank are
    overridden per-query by the classifier.

    Returns a RunResult with per-query and aggregate metrics.
    """
    require_credentials()
    if mode not in MODES:
        raise SystemExit(f"unknown --mode '{mode}' (expected one of {', '.join(MODES)})")
    if rewriter not in REWRITERS:
        raise SystemExit(f"unknown --rewriter '{rewriter}' (expected one of {', '.join(REWRITERS)})")
    if strategy_mode not in STRATEGY_MODES:
        raise SystemExit(f"unknown --strategy '{strategy_mode}' "
                         f"(expected one of {', '.join(STRATEGY_MODES)})")

    import voyageai
    import pymongo

    coll_name = collection_name(dataset)

    if verbose:
        header(f"Query  ×  {dataset}  ×  mode={mode}  ×  strategy={strategy_mode}")
        print(f"  Doc embedding   : {MODEL}")
        print(f"  Query embedding : {QUERY_MODEL}")
        if strategy_mode == "static":
            print(f"  Rewriter        : {rewriter}")
            print(f"  Reranker        : {'rerank-2.5' if rerank else 'off'}")
            print(f"  α               : {alpha:.2f}")
        else:
            print(f"  Strategy        : per-query via gpt-4o-mini")
        print(f"  Collection      : {DB_NAME}.{coll_name}")

    mongo = pymongo.MongoClient(MONGODB_URI)
    coll  = mongo[DB_NAME][coll_name]
    count = coll.estimated_document_count()
    if count == 0:
        raise SystemExit(
            f"Collection '{coll_name}' is empty. "
            f"Run `python3 ingest.py {dataset}` first."
        )

    if verbose:
        print(f"  Chunks          : {count:,}")

    _, queries, qrels, info = load_beir_dataset(dataset)

    eligible = [qid for qid in queries if any(s > 0 for s in qrels.get(qid, {}).values())]
    chosen_qids = eligible[:num_queries]
    if not chosen_qids:
        raise SystemExit(f"No queries with relevance judgments found for {dataset}.")

    if verbose:
        print(f"  Running {len(chosen_qids)} queries (of {len(queries):,} total) …")

    raw_query_texts = [queries[qid] for qid in chosen_qids]

    # ── Decide per-query strategy ────────────────────────────────────────────
    # In static mode, every query uses the same alpha / rewriter / rerank.
    # In dynamic mode, gpt-4o-mini picks each per-query.
    per_query_strategies: list[query_classifier.Strategy] = []
    if strategy_mode == "dynamic":
        per_query_strategies = query_classifier.predict_strategies(raw_query_texts)
    else:
        for _ in raw_query_texts:
            per_query_strategies.append(query_classifier.Strategy(
                alpha=alpha, rerank=rerank, rewriter=rewriter, reasoning="<<static>>",
            ))

    # ── Apply each query's rewriter (may produce multiple rewritten texts) ──
    rewrites_per_query: list[list[str]] = [
        rewrite(s.rewriter, t) for s, t in zip(per_query_strategies, raw_query_texts)
    ]

    # ── Embed every rewritten text in one batched Voyage call ────────────────
    needs_vector = mode in ("vector", "hybrid")
    flat_texts = [t for sub in rewrites_per_query for t in sub]
    if needs_vector and flat_texts:
        voyage = voyageai.Client(api_key=VOYAGE_API_KEY, base_url=MONGODB_BASE_URL)
        flat_vecs = embed_queries(voyage, flat_texts)
    else:
        flat_vecs = [None] * len(flat_texts)

    vecs_per_query: list[list[list[float] | None]] = []
    cursor = 0
    for sub in rewrites_per_query:
        n = len(sub)
        vecs_per_query.append(flat_vecs[cursor : cursor + n])
        cursor += n

    if verbose:
        header("Results")

    per_query: list[QueryResult] = []
    top_n = max(METRIC_KS)

    for qid, original_text, sub_texts, sub_vecs, strat in zip(
        chosen_qids, raw_query_texts, rewrites_per_query, vecs_per_query,
        per_query_strategies,
    ):
        qrel = qrels.get(qid, {})
        relevant_set = {did for did, s in qrel.items() if s > 0}

        # Reranking needs a deeper candidate pool from first-stage retrieval
        first_stage_k = RERANK_CANDIDATES if strat.rerank else top_n

        sub_queries = list(zip(sub_vecs, sub_texts))
        ranked_rows = multi_query_retrieve(
            mode, coll, sub_queries, top_k=first_stage_k, alpha=strat.alpha,
        )

        if strat.rerank and ranked_rows:
            ranked_rows = rerank_rows(original_text, ranked_rows, top_k=top_n)

        ranked_ids = [r["doc_id"] for r in ranked_rows]

        metrics = compute_query_metrics(ranked_ids, qrel)

        for row in ranked_rows:
            row["_relevant"] = row["doc_id"] in relevant_set

        qr = QueryResult(
            qid=qid, text=original_text,
            relevant_count=len(relevant_set),
            metrics=metrics,
            top_docs=ranked_rows[:TOP_K],
        )
        per_query.append(qr)

        if verbose:
            _print_query_block(qr)
            if strategy_mode == "dynamic":
                print(f"        ↳ routed: {strat.label()}  ({strat.reasoning})")
            if strat.rewriter != "none" and len(sub_texts) > 0:
                print(f"        ↳ rewriter produced {len(sub_texts)} text(s); "
                      f"first: \"{sub_texts[0][:80]}\"")

    aggregate = aggregate_metrics([qr.metrics for qr in per_query])

    if verbose:
        header("Summary")
        print(f"  Dataset           : {dataset}")
        print(f"  Mode              : {mode}")
        print(f"  Strategy          : {strategy_mode}")
        print(f"  Queries evaluated : {len(chosen_qids)}")
        print()
        print(f"  {format_summary(aggregate)}")
        print()

    mongo.close()

    return RunResult(
        dataset=dataset,
        mode=mode,
        rewriter=rewriter,
        rerank=rerank,
        strategy_mode=strategy_mode,
        num_queries=len(chosen_qids),
        chunks_in_collection=count,
        per_query=per_query,
        aggregate=aggregate,
        per_query_strategies=per_query_strategies if strategy_mode == "dynamic" else [],
    )


def main() -> None:
    p = argparse.ArgumentParser(
        prog="query.py",
        description="Query an ingested BEIR dataset via MongoDB Atlas Vector / Text / Hybrid search.",
    )
    p.add_argument("dataset", nargs="?", choices=list(DATASETS.keys()),
                   help="dataset to query (omit and pass --list to see options)")
    p.add_argument("--list", action="store_true",
                   help="print supported BEIR datasets and exit")
    p.add_argument("--mode", choices=MODES, default=DEFAULT_MODE,
                   help=f"retrieval mode (default: {DEFAULT_MODE})")
    p.add_argument("--rewriter", choices=REWRITERS, default=DEFAULT_REWRITER,
                   help=f"query rewriter (default: {DEFAULT_REWRITER}; "
                        "anything other than 'none' requires OPENAI_API_KEY)")
    p.add_argument("--rerank", action="store_true",
                   help="apply Voyage rerank-2.5 cross-encoder as a second stage")
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                   help="vector weight for hybrid fusion (0=text only, 1=vector only, "
                        f"0.5=balanced; default: {DEFAULT_ALPHA}). Ignored under "
                        "--strategy dynamic.")
    p.add_argument("--strategy", choices=STRATEGY_MODES, default=DEFAULT_STRATEGY_MODE,
                   help="how each query's full strategy (alpha + rewriter + rerank) "
                        "is chosen. 'static' uses the explicit flags; 'dynamic' has "
                        "gpt-4o-mini decide per query based on the query's "
                        "characteristics. Same classifier across all datasets. "
                        f"(default: {DEFAULT_STRATEGY_MODE}; dynamic requires OPENAI_API_KEY)")
    p.add_argument("--num-queries", type=int, default=DEFAULT_QUERIES,
                   help=f"how many queries to run (default: {DEFAULT_QUERIES})")
    args = p.parse_args()

    if args.list:
        print_dataset_list()
        return
    if not args.dataset:
        p.error("dataset is required (or pass --list to see supported datasets)")

    query(args.dataset, num_queries=args.num_queries,
          mode=args.mode, rewriter=args.rewriter, rerank=args.rerank,
          alpha=args.alpha, strategy_mode=args.strategy)


if __name__ == "__main__":
    main()
