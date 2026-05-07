"""
query.py — embed BEIR queries with voyage-3-large, retrieve from the
ingested MongoDB collection via $vectorSearch, and report a full IR
metric suite (P@K, R@K, NDCG@K, MRR, MAP) against official relevance
judgments.

  python3 query.py --list
  python3 query.py <dataset>
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


@dataclass
class QueryResult:
    qid: str
    text: str
    relevant_count: int
    metrics: dict[str, float]
    top_docs: list[dict] = field(default_factory=list)  # raw $vectorSearch rows for printing


@dataclass
class RunResult:
    dataset: str
    num_queries: int
    chunks_in_collection: int
    per_query: list[QueryResult]
    aggregate: dict[str, float]


def _retrieve(coll, q_vec: list[float], top_k: int) -> list[dict]:
    """Run $vectorSearch and dedupe to best chunk per parent doc."""
    pipeline = [
        {"$vectorSearch": {
            "index"        : INDEX_NAME,
            "path"         : "embedding",
            "queryVector"  : q_vec,
            "numCandidates": top_k * 20,
            "limit"        : top_k * 4,
        }},
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
        {"$sort": {"score": -1}},
    ]
    raw = list(coll.aggregate(pipeline))
    seen: dict[str, dict] = {}
    for row in raw:
        did = row["doc_id"]
        if did not in seen or row["score"] > seen[did]["score"]:
            seen[did] = row
    return sorted(seen.values(), key=lambda r: r["score"], reverse=True)


def _print_query_block(qr: QueryResult) -> None:
    m = qr.metrics
    print(f"\n  Query: \"{qr.text}\"")
    print(f"  {qr.relevant_count} relevant in corpus  |  "
          f"P@5={m['P@5']:.2f}  R@5={m['R@5']:.2f}  "
          f"NDCG@5={m['NDCG@5']:.3f}  MRR={m['MRR']:.3f}  AP={m['AP']:.3f}")
    print(f"  {'─' * 66}")
    for rank, row in enumerate(qr.top_docs[:TOP_K], 1):
        # Not ideal — we don't know the relevant set here, so we re-mark
        # at print time using the position in top_docs that matched.
        marker = "✓" if row.get("_relevant") else " "
        title  = row["title"] or "(untitled)"
        title  = (title[:55] + "…") if len(title) > 56 else title
        snippet = textwrap.shorten(row["text"].strip(), width=90, placeholder="…")
        print(f"  {rank}. [{marker}] {row['score']:.4f}  {title}")
        print(f"        {snippet}")


def query(dataset: str, num_queries: int = DEFAULT_QUERIES, verbose: bool = True) -> RunResult:
    """
    Run `num_queries` against the ingested collection and return RunResult
    with per-query and aggregate metrics. When `verbose`, also prints results.
    """
    require_credentials()

    import voyageai
    import pymongo

    coll_name = collection_name(dataset)

    if verbose:
        header(f"Query  ×  {dataset}  ×  {MODEL} / {QUERY_MODEL}")
        print(f"  Collection : {DB_NAME}.{coll_name}")

    mongo = pymongo.MongoClient(MONGODB_URI)
    coll  = mongo[DB_NAME][coll_name]
    count = coll.estimated_document_count()
    if count == 0:
        raise SystemExit(
            f"Collection '{coll_name}' is empty. "
            f"Run `python3 ingest.py {dataset}` first."
        )

    if verbose:
        print(f"  Chunks in collection: {count:,}")

    _, queries, qrels, info = load_beir_dataset(dataset)

    eligible = [qid for qid in queries if any(s > 0 for s in qrels.get(qid, {}).values())]
    chosen_qids = eligible[:num_queries]
    if not chosen_qids:
        raise SystemExit(f"No queries with relevance judgments found for {dataset}.")

    if verbose:
        print(f"  Running {len(chosen_qids)} queries (of {len(queries):,} total) …")

    voyage = voyageai.Client(api_key=VOYAGE_API_KEY, base_url=MONGODB_BASE_URL)
    query_texts = [queries[qid] for qid in chosen_qids]
    query_vecs  = embed_queries(voyage, query_texts)

    if verbose:
        header("Results")

    per_query: list[QueryResult] = []

    for qid, q_text, q_vec in zip(chosen_qids, query_texts, query_vecs):
        qrel = qrels.get(qid, {})
        relevant_set = {did for did, s in qrel.items() if s > 0}

        ranked_rows = _retrieve(coll, q_vec, top_k=max(METRIC_KS))
        ranked_ids  = [r["doc_id"] for r in ranked_rows]

        metrics = compute_query_metrics(ranked_ids, qrel)

        # Tag rows with relevance for printing
        for row in ranked_rows:
            row["_relevant"] = row["doc_id"] in relevant_set

        qr = QueryResult(
            qid=qid, text=q_text,
            relevant_count=len(relevant_set),
            metrics=metrics,
            top_docs=ranked_rows[:TOP_K],
        )
        per_query.append(qr)

        if verbose:
            _print_query_block(qr)

    aggregate = aggregate_metrics([qr.metrics for qr in per_query])

    if verbose:
        header("Summary")
        print(f"  Dataset             : {dataset}")
        print(f"  Queries evaluated   : {len(chosen_qids)}")
        print(f"  Doc embedding model : {MODEL}")
        print(f"  Query embedding     : {QUERY_MODEL}")
        print()
        print(f"  {format_summary(aggregate)}")
        print()

    mongo.close()

    return RunResult(
        dataset=dataset,
        num_queries=len(chosen_qids),
        chunks_in_collection=count,
        per_query=per_query,
        aggregate=aggregate,
    )


def main() -> None:
    p = argparse.ArgumentParser(
        prog="query.py",
        description="Query an ingested BEIR dataset via MongoDB Atlas Vector Search.",
    )
    p.add_argument("dataset", nargs="?", choices=list(DATASETS.keys()),
                   help="dataset to query (omit and pass --list to see options)")
    p.add_argument("--list", action="store_true",
                   help="print supported BEIR datasets and exit")
    p.add_argument("--num-queries", type=int, default=DEFAULT_QUERIES,
                   help=f"how many queries to run (default: {DEFAULT_QUERIES})")
    args = p.parse_args()

    if args.list:
        print_dataset_list()
        return
    if not args.dataset:
        p.error("dataset is required (or pass --list to see supported datasets)")

    query(args.dataset, num_queries=args.num_queries)


if __name__ == "__main__":
    main()
