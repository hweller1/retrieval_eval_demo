"""
query.py — embed BEIR queries with voyage-3-large, retrieve from the
ingested MongoDB collection via $vectorSearch, and report Precision@K
and MAP against official relevance judgments.

  python3 query.py --list
  python3 query.py <dataset>
  python3 query.py <dataset> --num-queries 10

Requires `python3 ingest.py <dataset>` to have been run first.
"""

from __future__ import annotations

import argparse
import textwrap

from lib import (
    DEFAULT_QUERIES, DB_NAME, INDEX_NAME, MODEL, MONGODB_BASE_URL,
    MONGODB_URI, QUERY_MODEL, TOP_K,
    DATASETS,
    collection_name, embed_queries, header, load_beir_dataset,
    print_dataset_list, require_credentials, VOYAGE_API_KEY,
)


def query(dataset: str, num_queries: int = DEFAULT_QUERIES) -> None:
    require_credentials()

    import voyageai
    import pymongo

    coll_name = collection_name(dataset)

    header(f"Query  ×  {dataset}  ×  {MODEL} / {QUERY_MODEL}")
    print(f"  Collection : {DB_NAME}.{coll_name}")

    # --- Verify collection is populated ------------------------------------
    mongo = pymongo.MongoClient(MONGODB_URI)
    coll  = mongo[DB_NAME][coll_name]
    count = coll.estimated_document_count()
    if count == 0:
        raise SystemExit(
            f"Collection '{coll_name}' is empty. "
            f"Run `python3 ingest.py {dataset}` first."
        )
    print(f"  Chunks in collection: {count:,}")

    # --- Load queries + qrels (corpus not needed for retrieval) -----------
    _, queries, qrels, info = load_beir_dataset(dataset)

    eligible = [qid for qid in queries if any(s > 0 for s in qrels.get(qid, {}).values())]
    chosen_qids = eligible[:num_queries]
    if not chosen_qids:
        raise SystemExit(f"No queries with relevance judgments found for {dataset}.")

    print(f"  Running {len(chosen_qids)} queries (of {len(queries):,} total) …")

    voyage = voyageai.Client(api_key=VOYAGE_API_KEY, base_url=MONGODB_BASE_URL)
    query_texts = [queries[qid] for qid in chosen_qids]
    query_vecs  = embed_queries(voyage, query_texts)

    header("Results")
    ap_scores: list[float]    = []
    p_at_k_scores: list[float] = []

    for qid, q_text, q_vec in zip(chosen_qids, query_texts, query_vecs):
        relevant_set = {did for did, s in qrels.get(qid, {}).items() if s > 0}

        pipeline = [
            {"$vectorSearch": {
                "index"        : INDEX_NAME,
                "path"         : "embedding",
                "queryVector"  : q_vec,
                "numCandidates": TOP_K * 20,
                "limit"        : TOP_K * 4,
            }},
            {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
            {"$sort": {"score": -1}},
        ]
        raw = list(coll.aggregate(pipeline))

        # Dedupe to best chunk per parent doc
        seen: dict[str, dict] = {}
        for row in raw:
            did = row["doc_id"]
            if did not in seen or row["score"] > seen[did]["score"]:
                seen[did] = row

        ranked = sorted(seen.values(), key=lambda r: r["score"], reverse=True)
        top_docs = ranked[:TOP_K]
        p_at_k = sum(1 for r in top_docs if r["doc_id"] in relevant_set) / TOP_K
        p_at_k_scores.append(p_at_k)

        n_rel, cum_p = 0, 0.0
        for rank, row in enumerate(ranked, 1):
            if row["doc_id"] in relevant_set:
                n_rel += 1
                cum_p += n_rel / rank
        ap = cum_p / len(relevant_set) if relevant_set else 0.0
        ap_scores.append(ap)

        print(f"\n  Query: \"{q_text}\"")
        print(f"  {len(relevant_set)} relevant in corpus  |  P@{TOP_K}={p_at_k:.2f}  |  AP={ap:.3f}")
        print(f"  {'─' * 66}")

        for rank, row in enumerate(top_docs, 1):
            marker  = "✓" if row["doc_id"] in relevant_set else " "
            title   = row["title"] or "(untitled)"
            title   = (title[:55] + "…") if len(title) > 56 else title
            snippet = textwrap.shorten(row["text"].strip(), width=90, placeholder="…")
            print(f"  {rank}. [{marker}] {row['score']:.4f}  {title}")
            print(f"        {snippet}")

    header("Summary")
    map_score   = sum(ap_scores)     / len(ap_scores)
    mean_p_at_k = sum(p_at_k_scores) / len(p_at_k_scores)
    print(f"  Dataset             : {dataset}")
    print(f"  Queries evaluated   : {len(chosen_qids)}")
    print(f"  Doc embedding model : {MODEL}")
    print(f"  Query embedding     : {QUERY_MODEL}")
    print(f"  Mean P@{TOP_K}           : {mean_p_at_k:.3f}")
    print(f"  MAP                 : {map_score:.3f}")
    print()
    mongo.close()


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
