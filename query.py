"""
query.py — embed BEIR queries with voyage-3-large, retrieve from the
ingested MongoDB collection, and report a full IR metric suite (P@K,
R@K, NDCG@K, MRR, MAP) against official relevance judgments.

Three retrieval modes are available:

  vector  : pure $vectorSearch over voyage-context-3 embeddings
  text    : pure $search (BM25) over the chunk text
  hybrid  : Reciprocal Rank Fusion of vector + text results

Examples:

  python3 query.py --list
  python3 query.py <dataset>                 # default mode: hybrid
  python3 query.py <dataset> --mode vector
  python3 query.py <dataset> --num-queries 10

Requires `python3 ingest.py <dataset>` to have been run first.
"""

from __future__ import annotations

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
from retrieve import MODES, retrieve, multi_query_retrieve
from query_rewriter import REWRITERS, DEFAULT_REWRITER, rewrite
from rerank import rerank as rerank_rows

DEFAULT_MODE       = "hybrid"
RERANK_CANDIDATES  = 50   # how many candidates to fetch from first-stage when reranking


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
    rewriter: str
    rerank: bool
    num_queries: int
    chunks_in_collection: int
    per_query: list[QueryResult]
    aggregate: dict[str, float]


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
    verbose: bool = True,
) -> RunResult:
    """
    Run `num_queries` queries against the ingested collection in `mode`,
    optionally applying a query `rewriter` first. Returns a RunResult with
    per-query and aggregate metrics. When verbose, also prints results.
    """
    require_credentials()
    if mode not in MODES:
        raise SystemExit(f"unknown --mode '{mode}' (expected one of {', '.join(MODES)})")
    if rewriter not in REWRITERS:
        raise SystemExit(f"unknown --rewriter '{rewriter}' (expected one of {', '.join(REWRITERS)})")

    import voyageai
    import pymongo

    coll_name = collection_name(dataset)

    if verbose:
        header(f"Query  ×  {dataset}  ×  mode={mode}  ×  rewriter={rewriter}"
               + ("  ×  +rerank" if rerank else ""))
        print(f"  Doc embedding   : {MODEL}")
        print(f"  Query embedding : {QUERY_MODEL}")
        print(f"  Reranker        : {'rerank-2.5' if rerank else 'off'}")
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

    # ── Apply rewriter to each query (yields list[str] per original query) ──
    raw_query_texts = [queries[qid] for qid in chosen_qids]
    rewrites_per_query: list[list[str]] = [rewrite(rewriter, t) for t in raw_query_texts]

    # ── Embed all rewrites in one batched call (vector / hybrid only) ───────
    needs_vector = mode in ("vector", "hybrid")
    flat_texts = [t for sub in rewrites_per_query for t in sub]
    if needs_vector and flat_texts:
        voyage = voyageai.Client(api_key=VOYAGE_API_KEY, base_url=MONGODB_BASE_URL)
        flat_vecs = embed_queries(voyage, flat_texts)
    else:
        flat_vecs = [None] * len(flat_texts)

    # Re-group vectors back per original query
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
    # When reranking, fetch a deeper candidate pool from first-stage retrieval
    # so the cross-encoder has room to reorder useful misses up.
    first_stage_k = RERANK_CANDIDATES if rerank else top_n

    for qid, original_text, sub_texts, sub_vecs in zip(
        chosen_qids, raw_query_texts, rewrites_per_query, vecs_per_query
    ):
        qrel = qrels.get(qid, {})
        relevant_set = {did for did, s in qrel.items() if s > 0}

        sub_queries = list(zip(sub_vecs, sub_texts))
        ranked_rows = multi_query_retrieve(mode, coll, sub_queries, top_k=first_stage_k)

        if rerank and ranked_rows:
            # Cross-encode against the original (un-rewritten) query for fairness
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
            if rewriter != "none" and len(sub_texts) > 0:
                print(f"        ↳ rewriter produced {len(sub_texts)} text(s); "
                      f"first: \"{sub_texts[0][:80]}\"")

    aggregate = aggregate_metrics([qr.metrics for qr in per_query])

    if verbose:
        header("Summary")
        print(f"  Dataset           : {dataset}")
        print(f"  Mode              : {mode}")
        print(f"  Rewriter          : {rewriter}")
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
        num_queries=len(chosen_qids),
        chunks_in_collection=count,
        per_query=per_query,
        aggregate=aggregate,
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
    p.add_argument("--num-queries", type=int, default=DEFAULT_QUERIES,
                   help=f"how many queries to run (default: {DEFAULT_QUERIES})")
    args = p.parse_args()

    if args.list:
        print_dataset_list()
        return
    if not args.dataset:
        p.error("dataset is required (or pass --list to see supported datasets)")

    query(args.dataset, num_queries=args.num_queries,
          mode=args.mode, rewriter=args.rewriter, rerank=args.rerank)


if __name__ == "__main__":
    main()
