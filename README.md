# voyage-context-3 × MongoDB Atlas Vector Search

A demo of [`voyage-context-3`](https://www.mongodb.com/docs/voyageai/models/contextualized-chunk-embeddings/),
Voyage AI's contextualized chunk-embedding model, running entirely through
the **MongoDB-hosted Voyage AI endpoint**. Documents are split with a
semantic-boundary-aware recursive splitter, embedded with full-document
context, ingested into MongoDB Atlas, and retrieved via `$vectorSearch`.
Quality is measured against official **BEIR** relevance judgments.

```
BEIR dataset → recursive split → /v1/contextualizedembeddings → MongoDB → $vectorSearch
```

## What you need

Both credentials come from a single MongoDB Atlas account — **no separate
Voyage AI account required**.

| Variable | Where to get it |
|---|---|
| `VOYAGE_API_KEY` | Atlas UI → your project → **AI Models** → **API Keys** → "Create API Key". The key starts with `al-`. |
| `MONGODB_URI` | Atlas UI → **Database** → "Connect" → "Drivers" → copy the `mongodb+srv://...` connection string and substitute your DB user's password. |

Both go in `.env`:

```bash
VOYAGE_API_KEY=al-...
MONGODB_URI=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority
```

> **Atlas prerequisites**
> - A cluster (M0 free tier works fine for the demo)
> - A database user with read/write access on `voyage_context_demo` (or admin)
> - Your IP allowlisted under **Network Access** (or `0.0.0.0/0` for testing)
> - The Voyage AI add-on enabled for the project so `AI Models` appears in the sidebar

## Install

The project is single-file Python with a few pip dependencies.

```bash
pip3 install voyageai pymongo python-dotenv requests beir
# On macOS with brew-managed Python you may need:
#   pip3 install --break-system-packages ...
```

## Usage

### See available datasets

```bash
python3 demo.py --list
```

```
Supported BEIR datasets

  touche2020   Argument retrieval — controversial debate topics (49 queries / 382k docs)
  scifact      Scientific claim verification (300 queries / 5.2k abstracts)
  fiqa         Financial Q&A — long-form opinionated answers (648 queries / 57k docs)
  nfcorpus     Medical literature retrieval (323 queries / 3.6k docs)
  arguana      Counter-argument retrieval (1.4k queries / 8.7k arguments)
  trec-covid   COVID-19 research retrieval (50 queries / 171k docs)
  scidocs      Scientific paper retrieval (1k queries / 25k docs)
  quora        Duplicate question retrieval (10k queries / 523k docs)
```

### Ingest a dataset

```bash
python3 demo.py --ingest touche2020              # default sample: 2000 docs
python3 demo.py --ingest scifact --sample 500    # smaller for fast iteration
```

This step:
1. Downloads the BEIR archive (cached in `/tmp/beir_datasets/`).
2. Splits each document with a recursive character text splitter that
   respects paragraph → line → sentence → word boundaries.
3. Embeds chunks via `POST https://ai.mongodb.com/v1/contextualizedembeddings`
   — each chunk is embedded together with its full parent document, so
   the resulting vector captures full-document context.
4. Inserts chunks + embeddings into `voyage_context_demo.chunks_<dataset>`.
5. Creates a vector search index named `voyage_vector_index` (cosine,
   1024 dimensions) and waits for it to become queryable.

The collection is dropped and rebuilt on every ingest of the same dataset.
Different datasets coexist in their own collections.

### Run queries

```bash
python3 demo.py --query touche2020                  # 5 queries (default)
python3 demo.py --query scifact --num-queries 20    # more queries
```

Queries are embedded with `voyage-3-large` (same generation as
voyage-context-3, compatible embedding space). Each query runs through a
`$vectorSearch` aggregation, results are deduplicated to one chunk per
parent document, and the top-K are scored against the official qrels.

Output for each query shows:
- The query text
- Number of relevant docs in the ingested sample
- Precision@5 and Average Precision
- The top-5 retrieved docs marked `[✓]` if relevant, with snippets

A summary block at the end reports mean P@5 and MAP.

## Test harness

`test_harness.py` validates the full pipeline against every supported
dataset. It runs `--ingest` then `--query` for each, checks MongoDB state,
parses the metrics, and prints a pass/fail table.

```bash
python3 test_harness.py --quick                 # 3 small datasets (~2 min)
python3 test_harness.py                          # all 8 datasets (~6 min)
python3 test_harness.py --datasets scifact fiqa  # specific datasets
python3 test_harness.py --sample 500 --num-queries 10
```

Sample output:

```
Dataset            Ingest      Query     Chunks     MAP     P@5  Status
────────────── ────────── ────────── ────────── ─────── ───────  ──────
touche2020          36.1s       7.5s        546   0.652   1.000  PASS
scifact             39.7s       2.3s        309   0.778   0.200  PASS
fiqa                35.6s       2.2s        148   0.900   0.267  PASS
nfcorpus            41.8s       1.9s        296   0.255   1.000  PASS
arguana             30.1s       2.1s        123   0.833   0.200  PASS
trec-covid          46.6s       2.5s        244   0.007   0.333  PASS
scidocs             69.9s       2.3s        216   0.768   0.667  PASS
quora               40.4s       2.4s        100   1.000   0.600  PASS

8/8 datasets passed
```

The harness exits non-zero if any dataset fails, so it's CI-friendly.

## How it differs from a standard RAG ingestion

| | Standard chunked embeddings | voyage-context-3 |
|---|---|---|
| Endpoint | `POST /v1/embeddings` | `POST /v1/contextualizedembeddings` |
| Input shape | `["chunk1", "chunk2", ...]` | `[[full_doc, chunk1, chunk2], ...]` |
| Per-chunk context | None — each chunk is embedded in isolation | Each chunk is embedded with awareness of the entire parent document |
| Best for | Short, self-contained passages | Long documents where chunks lose meaning when extracted (debate arguments, legal text, scientific abstracts) |

Same downstream workflow — the resulting vectors plug into MongoDB Vector
Search like any other 1024-dimensional embedding.

## File layout

```
voyage-context-3-testing/
├── .env                  # VOYAGE_API_KEY, MONGODB_URI
├── demo.py               # CLI: --list / --ingest / --query
├── test_harness.py       # validates ingest + query for all datasets
├── README.md             # this file
└── CLAUDE.md             # working notes for AI agents
```

## Troubleshooting

**`Authentication failed` (MongoDB)** — your `MONGODB_URI` credentials are
wrong or the password isn't URL-encoded. Special chars like `@`, `#`, `:`
must be percent-encoded (e.g. `@` → `%40`).

**`Model voyage-context-3 is not supported`** — you're hitting the standard
embeddings endpoint. The model only works on `/v1/contextualizedembeddings`.
The demo handles this correctly — make sure you're not calling it
yourself via `voyage.embed(model="voyage-context-3", ...)`.

**Index never becomes queryable** — Atlas vector indexes can take 30–60s
on a fresh collection. The script waits up to 150s; if it gives up you
can re-run `--query` later when the index is ready (check Atlas UI).

**`Collection ... is empty`** — you ran `--query` before `--ingest`, or
the previous ingest used a different collection name. Re-ingest.

## License / attribution

- BEIR datasets © their respective authors; this project loads them via
  the [BEIR Python package](https://github.com/beir-cellar/beir).
- voyage-context-3 is a Voyage AI model hosted by MongoDB.
