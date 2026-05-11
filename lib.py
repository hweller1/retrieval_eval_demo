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
    # Non-BEIR datasets use a `loader_fn` instead of `url` + `folder`.
    # The loader returns (corpus, queries, qrels, info) just like BEIR.
    "sec-10k": {
        "loader"     : "data_loaders.sec_10k:load",
        "split"      : "test",
        "description": "SEC 10-K filings — 15 US tech companies, FY2021-2024 (300 trader queries, no qrels — uses LLM judge)",
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

# voyage-context-3 hard limit is 32,000 tokens per inner-list. We approximate
# tokens as chars/4 (cl100k tokenizer averages ~3.7 chars/token on English
# prose; we use 4 to be conservative). Cap at 28k tokens (≈110k chars) per
# inner-list to leave headroom for the doc-anchor and for tokenizer slack.
MAX_CHARS_PER_INNER_LIST = 100_000


def _split_chunks_into_segments(
    full_doc: str, chunks: list[str], max_chars: int = MAX_CHARS_PER_INNER_LIST,
) -> list[list[str]]:
    """
    Build inner-lists for one document.

    If the full doc + all its chunks fit under `max_chars`, return a single
    inner-list of [full_doc, *chunks] — the full doc anchors all chunks.

    Otherwise we drop the anchor and split chunks into segments that each
    fit. Within a segment the chunks still cross-context against each
    other (the model's job), they just don't see the whole doc.
    """
    chunks_chars = sum(len(c) for c in chunks)
    if len(full_doc) + chunks_chars + 64 <= max_chars:
        return [[full_doc, *chunks]]

    segments: list[list[str]] = []
    current: list[str] = []
    current_chars = 0
    for c in chunks:
        if current and current_chars + len(c) > max_chars:
            segments.append(current)
            current = []
            current_chars = 0
        current.append(c)
        current_chars += len(c)
    if current:
        segments.append(current)
    return segments


def embed_contextualized(
    doc_chunks: list[tuple[str, list[str]]],
    batch_docs: int = EMBED_BATCH_DOCS,
    max_chars_per_batch: int = 400_000,
) -> list[list[float]]:
    """
    POST /v1/contextualizedembeddings.

    Each (full_doc, chunks) pair is converted into one or more inner-lists:
      - If full_doc + chunks fit in 28k tokens → one inner-list with full_doc
        as the contextual anchor; we skip that anchor in the response.
      - If too big → split chunks into anchorless segments that each fit.

    Adaptive request batching: respects a soft char cap per HTTP call to
    stay under the 6M TPM rate limit. Retries with exponential backoff on
    429 / 5xx.

    Returns chunk embeddings in the same order as the input `chunks`,
    flattened across all docs.
    """
    import time as _time

    url     = f"{MONGODB_BASE_URL}/contextualizedembeddings"
    headers = {"Authorization": f"Bearer {VOYAGE_API_KEY}", "Content-Type": "application/json"}

    # ── Pre-flatten: build (inner_list, n_chunks_in_it, has_anchor) records.
    # Each doc may produce multiple records.
    flat: list[tuple[list[str], int, bool, int]] = []  # (inner, n_chunks, has_anchor, doc_idx)
    chunks_per_doc: list[int] = []
    for di, (full_doc, chunks) in enumerate(doc_chunks):
        chunks_per_doc.append(len(chunks))
        segments = _split_chunks_into_segments(full_doc, chunks)
        if len(segments) == 1 and segments[0] and segments[0][0] is full_doc:
            # one segment with the full-doc anchor at index 0
            flat.append((segments[0], len(chunks), True, di))
        else:
            for seg in segments:
                flat.append((seg, len(seg), False, di))

    # Pre-allocate output list keyed by doc index then chunk position
    out_per_doc: list[list[list[float] | None]] = [
        [None] * n for n in chunks_per_doc
    ]
    next_chunk_pos: list[int] = [0] * len(doc_chunks)

    total_records = len(flat)

    def _record_chars(rec) -> int:
        return sum(len(s) for s in rec[0])

    i = 0
    while i < total_records:
        # Greedily grow batch by chars + doc-count cap (here doc-count refers
        # to inner-lists, not original docs)
        end = i
        batch_chars = 0
        while end < total_records and (end - i) < batch_docs:
            rc = _record_chars(flat[end])
            if batch_chars + rc > max_chars_per_batch and end > i:
                break
            batch_chars += rc
            end += 1
        batch = flat[i:end]
        inputs = [rec[0] for rec in batch]
        payload = {"model": MODEL, "inputs": inputs}

        for attempt in range(6):
            response = requests.post(url, json=payload, headers=headers, timeout=180)
            if response.status_code == 200:
                break
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 5:
                wait = 2 ** attempt + 1
                _time.sleep(wait)
                continue
            try:
                detail = response.json()
            except Exception:
                detail = response.text[:500]
            raise requests.HTTPError(
                f"{response.status_code} {response.reason} for {url}\nBody: {detail}"
            )
        else:
            response.raise_for_status()

        result = response.json()
        for rec_offset, doc_group in enumerate(result["data"]):
            inner_list, n_chunks, has_anchor, doc_idx = batch[rec_offset]
            items = sorted(doc_group["data"], key=lambda x: x["index"])
            chunk_items = items[1:] if has_anchor else items
            for item in chunk_items:
                pos = next_chunk_pos[doc_idx]
                out_per_doc[doc_idx][pos] = item["embedding"]
                next_chunk_pos[doc_idx] += 1

        i = end
        print(f"    {i:>{len(str(total_records))}}/{total_records} segments embedded …", end="\r")

    print()

    # Flatten in original doc order
    flat_vecs: list[list[float]] = []
    for vecs in out_per_doc:
        if any(v is None for v in vecs):
            raise RuntimeError("internal error: some chunks did not receive embeddings")
        flat_vecs.extend(vecs)
    return flat_vecs


def embed_queries(client, texts: list[str]) -> list[list[float]]:
    """Embed queries with QUERY_MODEL via the standard /v1/embeddings endpoint."""
    return client.embed(texts, model=QUERY_MODEL, input_type="query").embeddings


# ── Dataset loader ───────────────────────────────────────────────────────────

def load_beir_dataset(name: str):
    """Load a dataset by registry name. Returns (corpus, queries, qrels, info).

    Two backends are supported:
      - BEIR archive: entry has 'url' + 'folder' (downloaded + unzipped via beir).
      - Custom loader: entry has 'loader' = "module.path:func_name" — we
        import the function and call it. Used for non-BEIR sources like
        SEC 10-Ks where the corpus is fetched live and qrels don't exist.
    """
    if name not in DATASETS:
        raise SystemExit(
            f"Unknown dataset '{name}'. Run `python3 ingest.py --list` to see options."
        )
    info = DATASETS[name]

    # Custom loader path
    if "loader" in info:
        import importlib
        module_path, func_name = info["loader"].split(":")
        mod = importlib.import_module(module_path)
        load_fn = getattr(mod, func_name)
        return load_fn()

    # BEIR archive path
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
    print("    python3 ingest.py <dataset> [--sample N]")
    print("    python3 query.py  <dataset> [--mode vector|text|hybrid] "
          "[--rewriter none|hyde|multi|decompose] [--num-queries N]")
    print()


# ── Env-var guards ────────────────────────────────────────────────────────────

def require_credentials() -> None:
    if not VOYAGE_API_KEY:
        raise SystemExit("VOYAGE_API_KEY not set. Add it to .env (Atlas → AI Models → API Keys).")
    if not MONGODB_URI:
        raise SystemExit("MONGODB_URI not set. Add it to .env (Atlas → Database → Connect).")
