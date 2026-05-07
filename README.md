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
> - A cluster (M0 free tier works fine)
> - A database user with read/write access on `voyage_context_demo` (or admin)
> - Your IP allowlisted under **Network Access** (or `0.0.0.0/0` for testing)
> - The Voyage AI add-on enabled for the project so `AI Models` appears in the sidebar

## Install

```bash
pip3 install voyageai pymongo python-dotenv requests beir
# On macOS with brew-managed Python you may need:
#   pip3 install --break-system-packages ...
```

## Layout

```
voyage-demos/
├── .env              # VOYAGE_API_KEY, MONGODB_URI
├── lib.py            # shared: dataset registry, splitter, embedding helpers
├── ingest.py         # CLI: ingest a BEIR dataset into MongoDB
├── query.py          # CLI: run queries against an ingested dataset
├── test_harness.py   # validates ingest + query for every dataset
├── README.md
└── CLAUDE.md         # working notes for AI agents
```

The two scripts are independently runnable and share their dataset
registry, splitter, and embedding helpers via `lib.py`.

## Usage

### See available datasets

```bash
python3 ingest.py --list      # or:  python3 query.py --list
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
python3 ingest.py touche2020              # default sample: 2000 docs
python3 ingest.py scifact --sample 500    # smaller for fast iteration
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
python3 query.py touche2020                       # default mode: hybrid
python3 query.py scifact --mode vector            # pure vector search
python3 query.py scifact --mode text              # pure $search (BM25)
python3 query.py scifact --mode hybrid            # RRF over vector + text
python3 query.py scifact --num-queries 20
```

**Three retrieval modes** are available, all backed by the same indexed
collection:

| Mode | Backend | When it wins |
|---|---|---|
| `vector` | `$vectorSearch` over voyage-context-3 embeddings | Semantic / paraphrased queries; long-context preservation |
| `text` | `$search` (Atlas Search, BM25) over chunk text | Exact strings, codes, named entities, abbreviations |
| `hybrid` | RRF (k=60) over vector + text rankings | Real-world workloads — typically best on average |

`ingest.py` builds **both** indexes per dataset, so all three modes
are available without re-ingesting.

Each mode returns results deduplicated to one chunk per parent document
and scores them against the official qrels.

Output for each query shows:
- The query text
- Number of relevant docs in the ingested sample
- Per-query metrics: **P@5, R@5, NDCG@5, MRR, AP**
- The top-5 retrieved docs marked `[✓]` if relevant, with snippets

A summary block at the end reports the aggregate **P@5, R@5, NDCG@5,
NDCG@10, MRR, MAP** across all queries.

## Test harness

`test_harness.py` validates the full pipeline against every supported
dataset. It runs `ingest` then `query` for each, checks MongoDB state,
parses the metrics, and prints a pass/fail table.

```bash
python3 test_harness.py --quick                       # 3 small datasets, all 3 modes
python3 test_harness.py                                # all 8 datasets, all 3 modes
python3 test_harness.py --datasets scifact fiqa        # specific datasets
python3 test_harness.py --modes vector hybrid          # subset of modes
python3 test_harness.py --sample 500 --num-queries 10
python3 test_harness.py --quick --report report.md     # also write a markdown comparison
```

For each (dataset × mode) the harness runs ingest once and queries
three times (one per mode), then prints a Δ-vs-vector table and per-metric
comparison charts.

Output has three parts: a per-dataset summary table, ASCII bar charts
comparing each metric across datasets, and an optional Markdown file
suitable for sharing or pasting into a PR description.

```
Dataset         Ingest   Query   Chunks      P@5      R@5   NDCG@5  NDCG@10      MRR      MAP  Status
────────────── ─────── ─────── ──────── ──────── ──────── ──────── ──────── ──────── ────────  ──────
scifact          38.4s    5.2s      309    0.200    1.000    0.833    0.833    0.778    0.778  PASS
nfcorpus         35.3s    1.8s      296    1.000    0.144    0.914    0.879    1.000    0.410  PASS
arguana          34.9s    2.4s      123    0.200    1.000    0.877    0.877    0.833    0.833  PASS

3/3 datasets passed
```

```
NDCG@10
  scifact   ████████████████████      0.833
  nfcorpus  █████████████████████▏    0.879
  arguana   █████████████████████     0.877

MAP
  scifact   ██████████████████▋       0.778
  nfcorpus  █████████▉                0.410
  arguana   ████████████████████      0.833
…
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

## Troubleshooting

**`Authentication failed` (MongoDB)** — your `MONGODB_URI` credentials are
wrong or the password isn't URL-encoded. Special chars like `@`, `#`, `:`
must be percent-encoded (e.g. `@` → `%40`).

**`Model voyage-context-3 is not supported`** — you're hitting the standard
embeddings endpoint. The model only works on `/v1/contextualizedembeddings`.
The scripts handle this correctly — make sure you're not calling it
yourself via `voyage.embed(model="voyage-context-3", ...)`.

**Index never becomes queryable** — Atlas vector indexes can take 30–60s
on a fresh collection. The script waits up to 150s; if it gives up you
can re-run `query.py` later when the index is ready (check Atlas UI).

**`Collection ... is empty`** — you ran `query.py` before `ingest.py`, or
a previous ingest used a different collection name. Re-ingest.

## License / attribution

- BEIR datasets © their respective authors; this project loads them via
  the [BEIR Python package](https://github.com/beir-cellar/beir).
- voyage-context-3 is a Voyage AI model hosted by MongoDB.
