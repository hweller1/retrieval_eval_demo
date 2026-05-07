"""
voyage-context-3 demo: BEIR retrieval on MongoDB Atlas
======================================================
Two-phase CLI:

  python3 demo.py --list
      Show supported BEIR datasets.

  python3 demo.py --ingest <dataset>
      Download the dataset, split each document with a recursive character
      text splitter, embed every chunk via voyage-context-3's contextualized
      endpoint, store chunks + vectors in MongoDB, and create a vector
      search index. Drops any prior collection for the same dataset.

  python3 demo.py --query <dataset> [--num-queries N]
      Run the dataset's queries against the previously ingested collection
      and report Precision@K and MAP against official relevance judgments.

voyage-context-3 specifics:
  - Documents go through POST /v1/contextualizedembeddings — each chunk is
    embedded with awareness of the full parent document.
  - Queries are embedded with voyage-3-large via POST /v1/embeddings (same
    generation, compatible embedding space).

Environment variables (.env):
  VOYAGE_API_KEY   — MongoDB-issued Voyage AI key
  MONGODB_URI      — Atlas connection string (mongodb+srv://...)
"""

from __future__ import annotations

import os
import re
import sys
import time
import random
import textwrap
import argparse
import requests
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
VOYAGE_API_KEY   = os.environ.get("VOYAGE_API_KEY")
MONGODB_URI      = os.environ.get("MONGODB_URI")
MONGODB_BASE_URL = "https://ai.mongodb.com/v1"
MODEL            = "voyage-context-3"
QUERY_MODEL      = "voyage-3-large"

DB_NAME          = "voyage_context_demo"
INDEX_NAME       = "voyage_vector_index"
DATA_DIR         = "/tmp/beir_datasets"

CORPUS_SAMPLE    = 2_000   # docs per dataset (set to None to ingest the full corpus)
CHUNK_SIZE       = 1_000   # chars (~250 tokens)
CHUNK_OVERLAP    = 150
TOP_K            = 5
DEFAULT_QUERIES  = 5       # number of queries to run by default
INGEST_BATCH     = 500
EMBED_BATCH_DOCS = 50      # docs per contextualized API call


# ── Supported BEIR datasets ──────────────────────────────────────────────────
# Each entry maps a short CLI name → BEIR archive details.

DATASETS: dict[str, dict] = {
    "touche2020": {
        "url"        : "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/webis-touche2020.zip",
        "folder"     : "webis-touche2020",
        "split"      : "test",
        "description": "Argument retrieval — controversial debate topics (49 queries / 382k docs)",
    },
    "scifact": {
        "url"        : "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
        "folder"     : "scifact",
        "split"      : "test",
        "description": "Scientific claim verification (300 queries / 5.2k abstracts)",
    },
    "fiqa": {
        "url"        : "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip",
        "folder"     : "fiqa",
        "split"      : "test",
        "description": "Financial Q&A — long-form opinionated answers (648 queries / 57k docs)",
    },
    "nfcorpus": {
        "url"        : "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip",
        "folder"     : "nfcorpus",
        "split"      : "test",
        "description": "Medical literature retrieval (323 queries / 3.6k docs)",
    },
    "arguana": {
        "url"        : "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/arguana.zip",
        "folder"     : "arguana",
        "split"      : "test",
        "description": "Counter-argument retrieval (1.4k queries / 8.7k arguments)",
    },
    "trec-covid": {
        "url"        : "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip",
        "folder"     : "trec-covid",
        "split"      : "test",
        "description": "COVID-19 research retrieval (50 queries / 171k docs)",
    },
    "scidocs": {
        "url"        : "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scidocs.zip",
        "folder"     : "scidocs",
        "split"      : "test",
        "description": "Scientific paper retrieval (1k queries / 25k docs)",
    },
    "quora": {
        "url"        : "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/quora.zip",
        "folder"     : "quora",
        "split"      : "test",
        "description": "Duplicate question retrieval (10k queries / 523k docs)",
    },
}


# ── Recursive character text splitter ────────────────────────────────────────
SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""]


def _merge_splits(splits: list[str], separator: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for s in splits:
        sep_len = len(separator) if current else 0
        if current_len + sep_len + len(s) > chunk_size and current:
            chunk = separator.join(current).strip()
            if chunk:
                chunks.append(chunk)
            while current and current_len > chunk_overlap:
                current_len -= len(current[0]) + (len(separator) if len(current) > 1 else 0)
                current.pop(0)
        current.append(s)
        current_len = sum(len(p) for p in current) + len(separator) * (len(current) - 1)

    if current:
        chunk = separator.join(current).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _split_text(text: str, separators: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    final: list[str] = []
    sep = separators[-1]

    for candidate in separators:
        if candidate == "":
            sep = candidate
            break
        if candidate in text:
            sep = candidate
            break

    parts = re.split(re.escape(sep), text) if sep else list(text)

    good: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(part) <= chunk_size:
            good.append(part)
        else:
            if good:
                final.extend(_merge_splits(good, sep, chunk_size, chunk_overlap))
                good = []
            remaining = separators[separators.index(sep) + 1:] if sep in separators else []
            if remaining:
                final.extend(_split_text(part, remaining, chunk_size, chunk_overlap))
            else:
                final.append(part)

    if good:
        final.extend(_merge_splits(good, sep, chunk_size, chunk_overlap))

    return [c for c in final if c.strip()]


def split_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[str]:
    return _split_text(text, SEPARATORS, chunk_size, chunk_overlap)


# ── Embedding helpers ─────────────────────────────────────────────────────────

def embed_contextualized(
    doc_chunks: list[tuple[str, list[str]]],
    batch_docs: int = EMBED_BATCH_DOCS,
) -> list[list[float]]:
    """
    POST /v1/contextualizedembeddings.
    Prepends the full document text to each inner list so chunks are embedded
    with full-document context. Skips the document anchor in the response.
    """
    url     = f"{MONGODB_BASE_URL}/contextualizedembeddings"
    headers = {"Authorization": f"Bearer {VOYAGE_API_KEY}", "Content-Type": "application/json"}
    total   = len(doc_chunks)
    all_vecs: list[list[float]] = []

    for batch_start in range(0, total, batch_docs):
        batch  = doc_chunks[batch_start : batch_start + batch_docs]
        inputs = [[full_doc] + chunks for full_doc, chunks in batch]
        payload = {"model": MODEL, "inputs": inputs}
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        result = response.json()

        for doc_group in result["data"]:
            chunk_items = sorted(doc_group["data"], key=lambda x: x["index"])[1:]
            all_vecs.extend(item["embedding"] for item in chunk_items)

        done = min(batch_start + batch_docs, total)
        print(f"    {done:>{len(str(total))}}/{total} documents contextualized …", end="\r")

    print()
    return all_vecs


def embed_queries(client, texts: list[str]) -> list[list[float]]:
    return client.embed(texts, model=QUERY_MODEL, input_type="query").embeddings


# ── Pretty print ─────────────────────────────────────────────────────────────

def rule(char: str = "═", w: int = 72) -> None:
    print(char * w)


def header(title: str) -> None:
    print()
    rule()
    print(f"  {title}")
    rule()


# ── Dataset loader ───────────────────────────────────────────────────────────

def load_beir_dataset(name: str):
    """Download (if needed) and load a BEIR dataset. Returns (corpus, queries, qrels, info)."""
    if name not in DATASETS:
        raise SystemExit(
            f"Unknown dataset '{name}'. Run with --list to see supported datasets."
        )

    info     = DATASETS[name]
    data_path = os.path.join(DATA_DIR, info["folder"])

    from beir.datasets.data_loader import GenericDataLoader
    from beir import util

    if not os.path.isdir(data_path):
        print(f"  Downloading {name} from BEIR …")
        data_path = util.download_and_unzip(info["url"], DATA_DIR)

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=info["split"])
    return corpus, queries, qrels, info


def collection_name(dataset: str) -> str:
    return f"chunks_{dataset.replace('-', '_')}"


# ── Commands ─────────────────────────────────────────────────────────────────

def cmd_list() -> None:
    header("Supported BEIR datasets")
    print()
    width = max(len(k) for k in DATASETS)
    for name, info in DATASETS.items():
        print(f"  {name:<{width + 2}} {info['description']}")
    print()
    print("  Usage:")
    print("    python3 demo.py --ingest <dataset>")
    print("    python3 demo.py --query  <dataset> [--num-queries N]")
    print()


def cmd_ingest(dataset: str, corpus_sample: int | None = CORPUS_SAMPLE) -> None:
    if not VOYAGE_API_KEY:
        raise SystemExit("VOYAGE_API_KEY not set in .env.")
    if not MONGODB_URI:
        raise SystemExit("MONGODB_URI not set in .env.")

    import pymongo
    from pymongo.operations import SearchIndexModel

    coll_name = collection_name(dataset)

    header(f"Ingest  ×  {dataset}  ×  voyage-context-3")
    print(f"  Model      : {MODEL}")
    print(f"  Endpoint   : {MONGODB_BASE_URL}")
    print(f"  Collection : {DB_NAME}.{coll_name}")

    # Load
    corpus, queries, qrels, info = load_beir_dataset(dataset)
    all_ids = list(corpus.keys())

    # Sample
    random.seed(42)
    if corpus_sample and corpus_sample < len(all_ids):
        # Include docs marked relevant (so --query has hits), but only those that
        # actually exist in the corpus. Cap at corpus_sample to respect the limit.
        corpus_set = set(all_ids)
        must_include = [
            did for q_qrels in qrels.values()
            for did, s in q_qrels.items() if s > 0 and did in corpus_set
        ]
        # Dedupe while preserving order
        must_include = list(dict.fromkeys(must_include))
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

    # Split
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

    # Embed
    header(f"Step 2 — Contextualizing {len(records):,} chunks with {MODEL}")
    print(f"  Endpoint: POST {MONGODB_BASE_URL}/contextualizedembeddings")

    doc_to_records: dict[str, list[int]] = defaultdict(list)
    for idx, rec in enumerate(records):
        doc_to_records[rec["doc_id"]].append(idx)

    doc_order = list(doc_to_records.keys())
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

    # Ingest
    header(f"Step 3 — Ingesting into MongoDB ({DB_NAME}.{coll_name})")
    mongo = pymongo.MongoClient(MONGODB_URI)
    coll  = mongo[DB_NAME][coll_name]
    coll.drop()

    print(f"  Inserting {len(records):,} chunks …")
    t0 = time.time()
    for i in range(0, len(records), INGEST_BATCH):
        coll.insert_many(records[i : i + INGEST_BATCH])
    print(f"  ✓ Done in {time.time() - t0:.1f}s")

    # Index
    header("Step 4 — Creating vector search index")
    index_def = SearchIndexModel(
        definition={"fields": [{
            "type": "vector", "path": "embedding",
            "numDimensions": dims, "similarity": "cosine",
        }]},
        name=INDEX_NAME, type="vectorSearch",
    )
    existing = [idx["name"] for idx in coll.list_search_indexes()]
    if INDEX_NAME in existing:
        print(f"  Index '{INDEX_NAME}' already exists.")
    else:
        coll.create_search_index(index_def)
        print(f"  Index '{INDEX_NAME}' created. Waiting for it to become queryable …")
        for _ in range(30):
            time.sleep(5)
            ready = any(idx["name"] == INDEX_NAME and idx.get("queryable")
                        for idx in coll.list_search_indexes())
            if ready:
                print("  ✓ Index is queryable.")
                break
        else:
            print("  Index still building — first query may be slow.")

    print()
    print(f"  Ingest complete. Run queries with:")
    print(f"      python3 demo.py --query {dataset}")
    print()
    mongo.close()


def cmd_query(dataset: str, num_queries: int) -> None:
    if not VOYAGE_API_KEY:
        raise SystemExit("VOYAGE_API_KEY not set in .env.")
    if not MONGODB_URI:
        raise SystemExit("MONGODB_URI not set in .env.")

    import voyageai
    import pymongo

    coll_name = collection_name(dataset)

    header(f"Query  ×  {dataset}  ×  voyage-context-3 / {QUERY_MODEL}")
    print(f"  Collection : {DB_NAME}.{coll_name}")

    # Verify collection exists & is populated
    mongo = pymongo.MongoClient(MONGODB_URI)
    coll  = mongo[DB_NAME][coll_name]
    count = coll.estimated_document_count()
    if count == 0:
        raise SystemExit(
            f"Collection '{coll_name}' is empty. "
            f"Run `python3 demo.py --ingest {dataset}` first."
        )
    print(f"  Chunks in collection: {count:,}")

    # Load queries + qrels (corpus not needed)
    _, queries, qrels, info = load_beir_dataset(dataset)

    # Pick first N queries that have at least one judged-relevant doc in the corpus
    eligible = [qid for qid in queries if any(s > 0 for s in qrels.get(qid, {}).values())]
    chosen_qids = eligible[:num_queries]
    if not chosen_qids:
        raise SystemExit(f"No queries with relevance judgments found for {dataset}.")

    print(f"  Running {len(chosen_qids)} queries (of {len(queries):,} total) …")

    voyage = voyageai.Client(api_key=VOYAGE_API_KEY, base_url=MONGODB_BASE_URL)
    query_texts = [queries[qid] for qid in chosen_qids]
    query_vecs  = embed_queries(voyage, query_texts)

    header("Results")
    ap_scores: list[float] = []
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

        # Dedupe to best chunk per doc
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


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    dataset_choices = list(DATASETS.keys())

    p = argparse.ArgumentParser(
        prog="demo.py",
        description="voyage-context-3 demo on BEIR datasets via MongoDB Atlas.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              demo.py --list
              demo.py --ingest touche2020
              demo.py --query touche2020
              demo.py --query scifact --num-queries 10
        """),
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--list",   action="store_true",
                       help="show supported BEIR datasets and exit")
    group.add_argument("--ingest", metavar="DATASET", choices=dataset_choices,
                       help=f"ingest a dataset into MongoDB ({', '.join(dataset_choices)})")
    group.add_argument("--query",  metavar="DATASET", choices=dataset_choices,
                       help="run queries against a previously-ingested dataset")

    p.add_argument("--num-queries", type=int, default=DEFAULT_QUERIES,
                   help=f"how many queries to run with --query (default: {DEFAULT_QUERIES})")
    p.add_argument("--sample", type=int, default=CORPUS_SAMPLE,
                   help=f"how many documents to ingest with --ingest (default: {CORPUS_SAMPLE})")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.list:
        cmd_list()
    elif args.ingest:
        cmd_ingest(args.ingest, corpus_sample=args.sample)
    elif args.query:
        cmd_query(args.query, args.num_queries)


if __name__ == "__main__":
    main()
