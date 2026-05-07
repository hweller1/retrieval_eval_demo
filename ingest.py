"""
ingest.py — load a BEIR dataset, split each document with a recursive
character text splitter, embed every chunk via voyage-context-3's
contextualized endpoint, write chunks + vectors into MongoDB Atlas, and
create the vector search index.

  python3 ingest.py --list
  python3 ingest.py <dataset>
  python3 ingest.py <dataset> --sample 500

Drops the dataset's collection before re-ingesting.
"""

from __future__ import annotations

import time
import random
import argparse
from collections import defaultdict

from lib import (
    CHUNK_SIZE, CHUNK_OVERLAP, CORPUS_SAMPLE,
    DB_NAME, INDEX_NAME, INGEST_BATCH,
    MODEL, MONGODB_BASE_URL, MONGODB_URI,
    DATASETS,
    collection_name, embed_contextualized, header, load_beir_dataset,
    print_dataset_list, require_credentials, split_text,
)
from retrieve import TEXT_INDEX_NAME


def ingest(dataset: str, corpus_sample: int | None = CORPUS_SAMPLE) -> None:
    require_credentials()

    import pymongo
    from pymongo.operations import SearchIndexModel

    coll_name = collection_name(dataset)

    header(f"Ingest  ×  {dataset}  ×  voyage-context-3")
    print(f"  Model      : {MODEL}")
    print(f"  Endpoint   : {MONGODB_BASE_URL}")
    print(f"  Collection : {DB_NAME}.{coll_name}")

    # --- 0. Load dataset ----------------------------------------------------
    corpus, queries, qrels, info = load_beir_dataset(dataset)
    all_ids = list(corpus.keys())

    # --- 1. Sample ---------------------------------------------------------
    random.seed(42)
    if corpus_sample and corpus_sample < len(all_ids):
        # Include relevant docs first so query.py has hits to retrieve, but
        # only those that actually exist in the corpus (some BEIR datasets
        # have qrels referencing missing IDs). Cap at corpus_sample.
        corpus_set = set(all_ids)
        must_include = [
            did for q_qrels in qrels.values()
            for did, s in q_qrels.items() if s > 0 and did in corpus_set
        ]
        must_include = list(dict.fromkeys(must_include))      # dedupe, keep order
        if len(must_include) > corpus_sample:
            must_include = must_include[:corpus_sample]
        remaining = [did for did in all_ids if did not in set(must_include)]
        sample_ids = must_include + random.sample(
            remaining, max(0, corpus_sample - len(must_include))
        )
    else:
        sample_ids = all_ids

    print(f"  Sample     : {len(sample_ids):,} / {len(all_ids):,} documents")
    print(f"  Chunk size : {CHUNK_SIZE} chars  |  overlap: {CHUNK_OVERLAP} chars")

    # --- 2. Split into chunks ---------------------------------------------
    header("Step 1 — Recursive character text splitting")
    print("  Separators tried in order: \\n\\n → \\n → sentence → word → char")

    records: list[dict] = []
    for did in sample_ids:
        doc  = corpus[did]
        full = f"{doc['title']}\n\n{doc['text']}" if doc["title"] else doc["text"]
        for i, chunk in enumerate(split_text(full)):
            records.append({
                "doc_id"   : did,
                "chunk_idx": i,
                "title"    : doc["title"],
                "text"     : chunk,
            })

    chunk_lens = sorted(len(r["text"]) for r in records)
    print(f"  {len(sample_ids):,} documents → {len(records):,} chunks")
    print(f"  Chunk length: median {chunk_lens[len(chunk_lens)//2]} chars, "
          f"max {max(chunk_lens)} chars, min {min(chunk_lens)} chars")

    # --- 3. Embed chunks (contextualized) ---------------------------------
    header(f"Step 2 — Contextualizing {len(records):,} chunks with {MODEL}")
    print(f"  Endpoint: POST {MONGODB_BASE_URL}/contextualizedembeddings")

    doc_to_records: dict[str, list[int]] = defaultdict(list)
    for idx, rec in enumerate(records):
        doc_to_records[rec["doc_id"]].append(idx)

    doc_order: list[str] = list(doc_to_records.keys())
    doc_chunk_pairs: list[tuple[str, list[str]]] = []
    for did in doc_order:
        full_text = (
            f"{corpus[did]['title']}\n\n{corpus[did]['text']}"
            if corpus[did]["title"] else corpus[did]["text"]
        )
        chunks = [records[i]["text"] for i in doc_to_records[did]]
        doc_chunk_pairs.append((full_text, chunks))

    chunk_vecs_flat = embed_contextualized(doc_chunk_pairs)
    flat_idx = 0
    for did in doc_order:
        for rec_idx in doc_to_records[did]:
            records[rec_idx]["embedding"] = chunk_vecs_flat[flat_idx]
            flat_idx += 1
    dims = len(chunk_vecs_flat[0])
    print(f"  ✓ {len(chunk_vecs_flat):,} contextualized embeddings, {dims} dimensions each")

    # --- 4. Insert into MongoDB ------------------------------------------
    header(f"Step 3 — Ingesting into MongoDB ({DB_NAME}.{coll_name})")
    mongo = pymongo.MongoClient(MONGODB_URI)
    coll  = mongo[DB_NAME][coll_name]
    coll.drop()

    print(f"  Inserting {len(records):,} chunks …")
    t0 = time.time()
    for i in range(0, len(records), INGEST_BATCH):
        coll.insert_many(records[i : i + INGEST_BATCH])
    print(f"  ✓ Done in {time.time() - t0:.1f}s")

    # --- 5. Create vector + text search indexes -------------------------
    header("Step 4 — Creating search indexes (vector + text)")
    vector_index = SearchIndexModel(
        definition={"fields": [{
            "type": "vector", "path": "embedding",
            "numDimensions": dims, "similarity": "cosine",
        }]},
        name=INDEX_NAME, type="vectorSearch",
    )
    text_index = SearchIndexModel(
        definition={
            "mappings": {
                "dynamic": False,
                "fields": {
                    "text" : {"type": "string", "analyzer": "lucene.standard"},
                    "title": {"type": "string", "analyzer": "lucene.standard"},
                },
            }
        },
        name=TEXT_INDEX_NAME, type="search",
    )

    existing = {idx["name"] for idx in coll.list_search_indexes()}
    to_create: list[str] = []
    for idx_name, idx_model in [(INDEX_NAME, vector_index),
                                (TEXT_INDEX_NAME, text_index)]:
        if idx_name in existing:
            print(f"  Index '{idx_name}' already exists.")
        else:
            coll.create_search_index(idx_model)
            to_create.append(idx_name)
            print(f"  Index '{idx_name}' created.")

    if to_create:
        print(f"  Waiting for indexes to become queryable …")
        for _ in range(60):  # up to 5 minutes — text indexes can be slow
            time.sleep(5)
            statuses = {idx["name"]: idx.get("queryable", False)
                        for idx in coll.list_search_indexes()}
            if all(statuses.get(n) for n in to_create):
                print(f"  ✓ All indexes queryable.")
                break
        else:
            print(f"  Some indexes still building — first query may be slow.")

    print()
    print(f"  Ingest complete. Run queries with:")
    print(f"      python3 query.py {dataset}                    # default mode: hybrid")
    print(f"      python3 query.py {dataset} --mode vector")
    print(f"      python3 query.py {dataset} --mode text")
    print()
    mongo.close()


def main() -> None:
    p = argparse.ArgumentParser(
        prog="ingest.py",
        description="Ingest a BEIR dataset into MongoDB Atlas using voyage-context-3.",
    )
    p.add_argument("dataset", nargs="?", choices=list(DATASETS.keys()),
                   help="dataset to ingest (omit and pass --list to see options)")
    p.add_argument("--list", action="store_true",
                   help="print supported BEIR datasets and exit")
    p.add_argument("--sample", type=int, default=CORPUS_SAMPLE,
                   help=f"how many documents to ingest (default: {CORPUS_SAMPLE})")
    args = p.parse_args()

    if args.list:
        print_dataset_list()
        return
    if not args.dataset:
        p.error("dataset is required (or pass --list to see supported datasets)")

    ingest(args.dataset, corpus_sample=args.sample)


if __name__ == "__main__":
    main()
