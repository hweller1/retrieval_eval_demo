# CLAUDE.md — Working knowledge for this repo

This file is durable context for future Claude sessions on this codebase.
Read it before making changes.

## What this project is

A demo of `voyage-context-3` (Voyage AI's contextualized chunk embedding
model) running through the **MongoDB-hosted Voyage AI endpoint**, evaluated
against **BEIR** retrieval benchmarks with chunks stored in MongoDB Atlas
Vector Search.

Three entry points, a shared library, a metrics module, and a retrieval module:
- `lib.py` — shared dataset registry, splitter, embedding helpers, constants
- `lib_metrics.py` — IR metrics (P@K, R@K, NDCG@K, MRR, AP) over
  `(ranked_doc_ids, relevant_set_or_qrels_dict)`. `compute_query_metrics`
  returns a dict; `aggregate_metrics` reduces to MAP / mean of others.
  `METRIC_KS = (5, 10)` is the source of truth for which Ks are reported.
- `retrieve.py` — `vector_only`, `text_only`, `hybrid`. The hybrid
  uses Atlas's native `$rankFusion` operator (8.0+) for server-side
  weighted RRF, passing `weights={vector: alpha, text: 1-alpha}` so a
  single `alpha ∈ [0, 1]` controls the blend. `MODES = ("vector",
  "text", "hybrid")`. `DEFAULT_CANDIDATES = 100` per first-stage.
  `retrieve(mode, …, alpha)` dispatches; `multi_query_retrieve(...)`
  is still client-side because it RRFs across multiple *rewritten
  queries* of the same mode (rankFusion fuses pipelines, not query
  variants). Constants `INDEX_NAME` (from `lib`) and `TEXT_INDEX_NAME`
  here. Note: an earlier client-side `comb_sum` mode was removed —
  Atlas 8.3+ also ships native Relative Score Fusion via `$rankFusion`
  with sigmoid normalization, so we don't reimplement it.
- `llm_client.py` — thin OpenAI wrapper. Lazy-imports the `openai`
  package and only fails on missing `OPENAI_API_KEY` when actually
  invoked. Default model is `gpt-4o-mini`.
- `query_rewriter.py` — `rewrite(strategy, query) -> list[str]` with
  `REWRITERS = ("none", "hyde", "multi", "decompose")`. Each rewriter
  may return one or many texts; `none` is a passthrough that needs
  no OpenAI key. `query.py` flattens all rewrites into one batched
  Voyage embed call to keep latency down.
- `query_classifier.py` — `predict_strategy(query) -> Strategy` where
  `Strategy(alpha, rerank, rewriter, reasoning)` is a full per-query
  routing decision from one cheap-LLM (gpt-4o-mini) call. Few-shot
  prompt with 8 examples spanning the alpha range (0.15 → 0.80) and
  showing when each rewriter / rerank is appropriate. Used by
  `query.py` when `--strategy dynamic`. Cached by query string so
  repeats are free. Backwards-compat shims: `predict_alpha(query)` and
  `predict_alphas(queries)` still exist and just return `.alpha`.
- `rerank.py` — second-stage cross-encoder via `rerank-2.5`
  (POST `/v1/rerank`). Orthogonal to `--mode` and `--rewriter`: when
  enabled, `query.py` first fetches `RERANK_CANDIDATES=50` candidates
  from the chosen mode, then sends them with the original query to
  the reranker. Returns top-K with the cross-encoder's
  `relevance_score` as the `score` field.
- `ingest.py` — builds **both** the vector index and the Atlas Search text
  index in one pass. CLI: `python3 ingest.py <dataset> [--sample N] [--list]`
- `query.py` — `python3 query.py <dataset> [--mode vector|text|hybrid]
  [--num-queries N] [--list]`. Default mode is `hybrid`.
  `query.query(...)` returns a `RunResult` with `.aggregate`, `.per_query`,
  and `.mode`. When `verbose=False` it is silent — the harness uses this
  to capture metrics without stdout-scraping.
- `test_harness.py` — imports `ingest.ingest` and `query.query` directly.
  Runs every dataset × mode combination, renders summary + Δ-vs-vector
  table + per-metric grouped bar charts (one bar per mode per dataset).
  Optional `--report PATH` writes a Markdown comparison.

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
├── .env                    # VOYAGE_API_KEY, MONGODB_URI, OPENAI_API_KEY
├── lib.py                  # shared registry/splitter/embedding helpers/constants
├── lib_metrics.py          # IR metrics: P@K, R@K, NDCG@K, MRR, AP/MAP
├── retrieve.py             # vector / text / hybrid; multi_query_retrieve
├── llm_client.py           # thin OpenAI wrapper (lazy-imported)
├── query_rewriter.py       # none / hyde / multi / decompose
├── rerank.py               # second-stage cross-encoder (rerank-2.5)
├── ingest.py               # builds vector + text indexes per dataset
├── query.py                # CLI with --mode --rewriter --rerank
├── test_harness.py         # dataset × strategy; --rerank off|on|both
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

## Hybrid retrieval findings (500 docs / 10 queries)

Compiled from `experiments/compare_fusion_strategies.py` on 3 datasets.
NDCG@10:

| Strategy | scifact | nfcorpus | touche2020 |
|---|---|---|---|
| vector              | 0.874 | **0.781** | 0.913 |
| text                | 0.711 | 0.419 | 0.911 |
| hybrid α=0.5 (RRF)  | 0.833 | 0.662 | 0.941 |
| hybrid α=0.8 (RRF)  | 0.872 | 0.745 | **0.946** |
| dynamic (per-query) | 0.872 | 0.726 | 0.944 |

Takeaways:

1. The original "uniform-RRF" hybrid (α=0.5) was just badly weighted —
   it dragged in BM25 noise on datasets where vector was the stronger
   signal. Manually bumping α to 0.8 closes most of the gap.
2. The **dynamic per-query classifier** matches manually-tuned static
   α=0.8 on BEIR (all three datasets within ~0.02 NDCG@10) WITHOUT any
   per-dataset tuning. Same prompt, same model, decisions made per
   query.
3. BEIR queries are *homogeneous within each dataset* — all scientific
   claims, or all health questions, or all debate prompts. So the
   classifier picks the same α (≈0.80) for almost every query. Where
   dynamic should pull ahead: **heterogeneous query streams** where
   some queries are exact-string ("CVE-2021-44228" → α=0.15), some are
   single-word ("vaping" → α=0.55 +hyde +rerank), some are compound
   ("How does microbiome affect mood and is it different in vegans?"
   → α=0.75 +decompose), and some are scientific claims (α=0.80). On
   such streams no static config can win — that's where the dynamic
   layer adds real value.
4. Atlas 8.3 ships **native Relative Score Fusion** via `$rankFusion`,
   which is the same idea as our (now removed) `comb_sum`. Use the
   native operator for production; we keep the demo simple with
   weighted RRF.

## What has been verified

As of 2026-05-07, after Stages 1–3:

- IR metrics math (P@K, R@K, NDCG@K, AP, MRR, MAP) checked against
  hand-computed expected values, including BEIR-style graded relevance.
- RRF math in `retrieve.multi_query_retrieve`: single-query passthrough,
  multi-query fan-in, score summation when the same doc appears in
  multiple sub-rankings. Verified d1 score == 1/61 + 1/61 ≈ 0.0328 in a
  two-ranking overlap case.
- Recursive splitter respects paragraph (`\n\n`) > line > sentence > word
  > char fallback, and handles empty / short / no-separator inputs.
- End-to-end smoke test: `scifact` × 3 modes × `none` rewriter — all PASS,
  index creation correct, vector + text indexes both queryable.
- Missing-`OPENAI_API_KEY` raises a clear actionable error from
  `llm_client.complete()` instead of an obscure ImportError or 401.
- `--list` works for both `ingest.py` and `query.py`; required-arg error
  is informative.

What has *not* been verified at run time (no harness coverage yet):

- LLM rewriters (`hyde`, `multi`, `decompose`) end-to-end against real
  OpenAI — needs `OPENAI_API_KEY` set in `.env`.
- "Hybrid beats vector on ≥5/8 BEIR datasets" — needs a full sample run
  (`test_harness.py --sample 1000+`); current 100-doc samples are too
  small for the comparison to stabilize.
- Stage 4–6 are unimplemented per the roadmap in
  `~/.claude/plans/what-is-a-logical-staged-breeze.md`.

## When something breaks

| Symptom | Likely cause |
|---|---|
| `Model voyage-context-3 is not supported` from the SDK | You're calling `/v1/embeddings`. Use `embed_contextualized()` instead, which hits `/v1/contextualizedembeddings`. |
| `404 Not Found` on contextualized endpoint | Endpoint path is one word: `contextualizedembeddings`, no slash before "embeddings". |
| `Authentication failed` on MongoDB | URI credentials wrong, or password contains unencoded special chars (`@` → `%40`). The connection itself is fine if the cluster hostname resolves. |
| `KeyError` during ingest sample step | qrels referencing doc IDs not in the corpus. Intersect with `corpus.keys()`. |
| Index never becomes queryable | Atlas vector index typically takes 30–60s on a fresh collection; the text index can take longer. `ingest.py` polls for ~5 min, then proceeds anyway. |
| `'SearchIndexModel' object is not subscriptable` | `SearchIndexModel(...)` returns an object, not a dict. Don't do `model["name"]`. Track names in a parallel list/tuple instead. |

## What NOT to do

- Don't try to use the Voyage AI direct API (`api.voyageai.com`) with the
  MongoDB-issued key — they're separate auth surfaces.
- Don't add a "fallback to chunked standard embeddings" path. The whole
  point of this demo is the contextualized endpoint.
- Don't change `DB_NAME` or `collection_name()` without realizing it
  orphans previously-ingested collections.
