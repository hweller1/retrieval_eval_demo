"""
Shared helpers for the voyage-context-3 demo scripts (ingest.py, query.py).

Contains constants, the BEIR dataset registry, the recursive character text
splitter, the embedding helpers (contextualized for documents, standard for
queries), the MongoDB collection-name convention, and the dataset listing
command. The actual ingest/query pipelines live in their own scripts so each
script is independently runnable.
"""

from __future__ import annotations

import os
import re
import requests
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

CORPUS_SAMPLE    = 2_000
CHUNK_SIZE       = 1_000
CHUNK_OVERLAP    = 150
TOP_K            = 5
DEFAULT_QUERIES  = 5
INGEST_BATCH     = 500
EMBED_BATCH_DOCS = 50


# ── BEIR dataset registry ────────────────────────────────────────────────────

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
    """Embed queries with QUERY_MODEL via the standard /v1/embeddings endpoint."""
    return client.embed(texts, model=QUERY_MODEL, input_type="query").embeddings


# ── Dataset loader ───────────────────────────────────────────────────────────

def load_beir_dataset(name: str):
    """Download (if needed) and load a BEIR dataset. Returns (corpus, queries, qrels, info)."""
    if name not in DATASETS:
        raise SystemExit(
            f"Unknown dataset '{name}'. Run `python3 ingest.py --list` to see options."
        )

    info      = DATASETS[name]
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


# ── Pretty print ─────────────────────────────────────────────────────────────

def rule(char: str = "═", w: int = 72) -> None:
    print(char * w)


def header(title: str) -> None:
    print()
    rule()
    print(f"  {title}")
    rule()


def print_dataset_list() -> None:
    header("Supported BEIR datasets")
    print()
    width = max(len(k) for k in DATASETS)
    for name, info in DATASETS.items():
        print(f"  {name:<{width + 2}} {info['description']}")
    print()
    print("  Usage:")
    print("    python3 ingest.py <dataset>")
    print("    python3 query.py  <dataset> [--num-queries N]")
    print()


# ── Env-var guards ────────────────────────────────────────────────────────────

def require_credentials() -> None:
    if not VOYAGE_API_KEY:
        raise SystemExit("VOYAGE_API_KEY not set. Add it to .env (Atlas → AI Models → API Keys).")
    if not MONGODB_URI:
        raise SystemExit("MONGODB_URI not set. Add it to .env (Atlas → Database → Connect).")
