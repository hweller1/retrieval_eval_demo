# CLAUDE.md — Working knowledge for this repo

This file is durable context for future Claude sessions on this codebase.
Read it before making changes.

## What this project is

A demo of `voyage-context-3` (Voyage AI's contextualized chunk embedding
model) running through the **MongoDB-hosted Voyage AI endpoint**, evaluated
against **BEIR** retrieval benchmarks with chunks stored in MongoDB Atlas
Vector Search.

Three entry points and a shared library:
- `lib.py` — shared dataset registry, splitter, embedding helpers, constants
- `ingest.py` — CLI: `python3 ingest.py <dataset> [--sample N]` (also `--list`)
- `query.py` — CLI: `python3 query.py <dataset> [--num-queries N]` (also `--list`)
- `test_harness.py` — end-to-end validation across all supported BEIR datasets,
  imports `ingest.ingest()` and `query.query()` directly

## Key facts you must not relearn

### MongoDB-hosted Voyage AI is a separate API surface

- Base URL: `https://ai.mongodb.com/v1`
- Auth: `Authorization: Bearer <VOYAGE_API_KEY>` where the key is
  **issued by Atlas** (under "AI Models"), not by voyageai.com. Keys start
  with `al-`.
- Standard embedding endpoint: `POST /v1/embeddings`
  - Used by `voyageai.Client(base_url=...).embed(...)`.
  - Supports models like `voyage-4-large`, `voyage-3-large`, `voyage-3.5`,
    `voyage-code-3`, etc.
  - **Does NOT support `voyage-context-3`.** The error message lists
    supported models — context-3 is conspicuously absent there but IS
    listed on the Atlas Rate Limits page.

### `voyage-context-3` lives on a different endpoint

- Endpoint: `POST /v1/contextualizedembeddings`
- Request body shape:
  ```json
  {
    "model": "voyage-context-3",
    "inputs": [
      ["doc1_text", "doc1_chunk1", "doc1_chunk2"],
      ["doc2_text", "doc2_chunk1"]
    ],
    "input_type": "document"   // optional
  }
  ```
  - `inputs` is `array[array[string]]`. Each inner list is one document's
    text elements; **all elements in an inner list are embedded with shared
    context**.
  - We prepend the full document text as inputs[i][0] so each chunk is
    contextualized against the whole document, then **discard index 0** in
    the response.
  - Limits: 1,000 inner lists per call, 120K total tokens, 16K chunks.
- Response shape (nested):
  ```json
  {
    "data": [
      { "data": [
          { "index": 0, "embedding": [...] },
          { "index": 1, "embedding": [...] }
        ]
      },
      ...
    ]
  }
  ```
- The official `voyageai` Python SDK does **not** wrap this endpoint.
  `demo.embed_contextualized()` calls it directly with `requests`.

### Query embedding model

We embed queries with `voyage-3-large` via the standard `/v1/embeddings`
endpoint. Reasoning: same generation as voyage-context-3, compatible
embedding space for retrieval, and `voyage-context-3` is not accepted on
the standard endpoint. Going through the contextualized endpoint for a
single-string query is awkward, so we use the SDK for queries and only
hit the contextualized endpoint for documents.

### MongoDB schema

- Database: `voyage_context_demo` (constant `DB_NAME`)
- Collections: `chunks_<dataset>` (e.g. `chunks_touche2020`,
  `chunks_scifact`). Built by `demo.collection_name(dataset)`.
- Document shape: `{doc_id, chunk_idx, title, text, embedding}`
- Vector index name: `voyage_vector_index` (constant `INDEX_NAME`),
  cosine similarity, dims discovered from the first embedding (1024 by
  default).

### Recursive splitter

`demo.split_text` is a from-scratch reimplementation of LangChain's
`RecursiveCharacterTextSplitter`:
- Separator priority: `["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""]`
- For each block, picks the **first** separator that appears in the text,
  splits, then merges back to chunk_size with chunk_overlap. Recurses on
  oversized parts using the next separator.
- Default chunk size: 1000 chars (~250 tokens), overlap: 150 chars.

### Sample logic in `cmd_ingest`

When `corpus_sample < len(corpus)`:
1. Find docs marked relevant by **any** query in `qrels` so `--query`
   has hits to retrieve.
2. **Filter to docs that actually exist in the corpus** — some BEIR
   datasets (notably `arguana`) have qrels referencing missing doc IDs.
   Forgetting this guard caused a `KeyError` during ingestion.
3. Cap `must_include` at `corpus_sample` so the ingestion respects the
   user's limit even when there are huge numbers of relevant docs (e.g.
   nfcorpus had 9k relevant docs across all queries, blowing past
   sample=100 before this cap was added).

### Known dataset quirks

- `arguana`: qrels reference some doc IDs not in the corpus dump → must
  intersect with `corpus.keys()`.
- `trec-covid`: very low MAP at small sample sizes (relevant docs unlikely
  to land in a 100-doc sample). Test harness still passes since at least
  one hit usually appears in P@5.
- `nfcorpus`: many "relevant" docs per query, so unfiltered must_include
  explodes the sample size.
- `touche2020` corpus is large (382k docs, 217MB download).

## File layout

```
voyage-demos/
├── .env                    # VOYAGE_API_KEY, MONGODB_URI (gitignored)
├── lib.py                  # shared registry/splitter/embedding helpers/constants
├── ingest.py               # CLI: ingest <dataset> [--sample N] [--list]
├── query.py                # CLI: query  <dataset> [--num-queries N] [--list]
├── test_harness.py         # end-to-end validation across all BEIR datasets
├── README.md               # user-facing docs
└── CLAUDE.md               # this file
```

Datasets cache to `/tmp/beir_datasets/<folder>/` — first ingest of a
dataset downloads it.

`test_harness.py` imports `ingest.ingest` and `query.query` directly (no
subprocess) so it can capture stdout cheaply and assert on metrics.

## Useful commands

```bash
# List datasets (either script works)
python3 ingest.py --list

# Ingest one dataset (default sample 2000)
python3 ingest.py touche2020

# Ingest with smaller sample for fast iteration
python3 ingest.py scifact --sample 200

# Run queries (collection must exist)
python3 query.py touche2020 --num-queries 5

# Smoke test: 3 small datasets, ~2 minutes
python3 test_harness.py --quick --sample 100 --num-queries 3

# Full validation: all 8 datasets, ~6 minutes
python3 test_harness.py --sample 100 --num-queries 3
```

## Conventions / gotchas in this codebase

- Don't reach for `mkdir`/`os.makedirs` for `/tmp/beir_datasets` — the BEIR
  loader handles that.
- The `voyageai` package is at version 0.3.7+ and supports the `base_url`
  kwarg on `Client(...)`. If you switch to a much older version this will
  break.
- The user's machine uses `pip3 install --break-system-packages` because
  Python 3.13 is brew-managed (PEP 668). Keep that in any install
  instructions you give the user.
- `test_harness.py` sets `TQDM_DISABLE=1` before importing `demo` to keep
  the BEIR progress bars from polluting the harness output.
- The MongoDB-hosted endpoint is **preview** as of writing; model availability
  may change. The Atlas UI's Rate Limits page (Project → AI Models → Rate
  Limits) is the source of truth for which models the user's project has
  access to.

## When something breaks

| Symptom | Likely cause |
|---|---|
| `Model voyage-context-3 is not supported` from the SDK | You're calling `/v1/embeddings`. Use `embed_contextualized()` instead, which hits `/v1/contextualizedembeddings`. |
| `404 Not Found` on contextualized endpoint | Endpoint path is one word: `contextualizedembeddings`, no slash before "embeddings". |
| `Authentication failed` on MongoDB | URI credentials wrong, or password contains unencoded special chars (`@` → `%40`). The connection itself is fine if the cluster hostname resolves. |
| `KeyError` during ingest sample step | qrels referencing doc IDs not in the corpus. Intersect with `corpus.keys()`. |
| Index never becomes queryable | Atlas vector index typically takes 30–60s on a fresh collection; `cmd_ingest` polls for 150s, then proceeds anyway. |

## What NOT to do

- Don't try to use the Voyage AI direct API (`api.voyageai.com`) with the
  MongoDB-issued key — they're separate auth surfaces.
- Don't add a "fallback to chunked standard embeddings" path. The whole
  point of this demo is the contextualized endpoint.
- Don't change `DB_NAME` or `collection_name()` without realizing it
  orphans previously-ingested collections.
