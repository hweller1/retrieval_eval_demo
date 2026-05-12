# AGENTS.md — context for coding agents

This file is for **coding agents** (Claude Code, Cursor, Copilot, etc.)
that a learner is using to step through the retrieval-evaluation lab.
It tells you where things live, what jobs you can do, what conventions
to preserve, and how to verify the lab still works after you change
something.

> **Two tracks.** This repo has two parallel lab tracks. Humans
> stepping through MQL by hand use **`notebooks/`** (the hand-coded
> track). Agent-assisted users use **`agent-notebooks/`** (terser,
> library-driven). This file is the agent track's working manual; the
> hand-coded track is meant to be read sequentially without agent
> help.

## Codebase map

```
voyage-context-3-testing/
├── README.md                 # course-companion overview
├── CLAUDE.md                 # working knowledge for Claude Code sessions
├── AGENTS.md                 # this file
├── .env                      # VOYAGE_API_KEY, MONGODB_URI, OPENAI_API_KEY
│
├── lib.py                    # dataset registry, splitter, embedding helpers, constants
├── lib_metrics.py            # P@k, R@k, NDCG@k, MRR, AP/MAP
├── retrieve.py               # vector_only / text_only / hybrid (Atlas pipelines)
├── ingest.py                 # BEIR → chunks → voyage-context-3 → MongoDB + indexes
│
├── notebooks/                # HAND-CODED lab (default for humans)
│   ├── 00_setup_and_ingest.ipynb
│   ├── 01_evaluate_blackbox.ipynb
│   ├── 02_swap_blackbox.ipynb
│   └── 03_curate_eval_set.ipynb
│
├── agent-notebooks/          # AGENT-FRIENDLY lab (library-driven, terser)
│   └── (same four file names)
│
├── scripts/                  # notebook builders (single source of truth)
│   ├── _nb_helpers.py        # md/code/notebook/write primitives
│   ├── build_handcoded.py    # generates notebooks/
│   └── build_agent.py        # generates agent-notebooks/
│
└── phase4/                   # advanced — after-the-lab material
    ├── README.md
    ├── query.py, test_harness.py
    ├── query_rewriter.py, query_classifier.py, rerank.py, llm_client.py
    ├── llm_judge.py, judge_cache.json
    ├── data_loaders/         # non-BEIR (sec-10k)
    └── experiments/          # multi-dataset benchmark sweeps
```

### Module responsibilities

| Module | Exports | What's it for |
|---|---|---|
| `lib.py` | `DATASETS`, `VOYAGE_API_KEY`, `MONGODB_URI`, `DB_NAME`, `INDEX_NAME`, `load_beir_dataset`, `collection_name`, `split_text`, `embed_contextualized`, `embed_queries`, `require_credentials` | Config + the two unglamorous helpers (recursive splitter, contextualized-embeddings HTTP call) |
| `lib_metrics.py` | `precision_at_k`, `recall_at_k`, `ndcg_at_k`, `reciprocal_rank`, `average_precision`, `compute_query_metrics`, `aggregate_metrics`, `format_summary`, `METRIC_KS` | IR metric formulas. `METRIC_KS = (5, 10)` is the source of truth for which Ks both tracks report. |
| `retrieve.py` | `vector_only`, `text_only`, `hybrid`, `multi_query_retrieve`, `retrieve`, `MODES`, `DEFAULT_ALPHA`, `TEXT_INDEX_NAME` | The three retrieval modes as Python wrappers. `hybrid` uses Atlas's native `$rankFusion` (8.0+). |
| `ingest.py` | `ingest(dataset, corpus_sample)` | Full pipeline: load BEIR → split → embed → insert → build both search indexes. CLI: `python3 ingest.py <dataset>` |
| `phase4/*` | Everything advanced — see `phase4/README.md` | Per-query routing, query rewriters, reranker, LLM judge, multi-dataset experiment scripts |

## Common tasks

### Re-generate notebooks after editing

The `.ipynb` files are **generated**. Do not hand-edit them — your
changes will be overwritten the next time someone runs the builder.

```bash
python3 scripts/build_handcoded.py    # regenerates notebooks/
python3 scripts/build_agent.py        # regenerates agent-notebooks/
```

Edit the markdown/code blocks inside `scripts/build_handcoded.py` or
`scripts/build_agent.py` instead. They use a tiny DSL (`md("...")`
and `code("...")`) for readability.

### Add a new BEIR dataset

1. Add an entry to `DATASETS` in `lib.py` with `url`, `folder`, `split`, `description`.
2. Update the dataset table in **both** notebook builders (`scripts/build_handcoded.py` step 3 of `nb00()`, and the equivalent block in `scripts/build_agent.py`).
3. Update the dataset table in `README.md`.
4. Rebuild notebooks.

### Add a new retrieval strategy

1. Add the function to `retrieve.py`. Match the existing signature:
   `(coll, q_vec, query_text, top_k, ...) -> list[dict]` where each
   row has at minimum `doc_id`, `text`, `score`.
2. Extend `MODES` and the `retrieve()` dispatcher in `retrieve.py`.
3. If the strategy is advanced (LLM-driven, reranker-dependent),
   put it under `phase4/` instead and import it from there.
4. The hand-coded notebooks (`notebooks/`) deliberately *inline* the
   MQL — do not refactor them to call `retrieve.<new_strategy>`.
   That's the agent track's pattern, not the hand-coded one.

### Add a new metric

1. Add the formula to `lib_metrics.py` following the existing pattern:
   takes `(ranked: list[str], relevant: set | Mapping[str, int])`.
2. Add it to `compute_query_metrics()` so aggregations pick it up.
3. If `METRIC_KS` needs to change, change it there — *not* in any
   notebook (the hand-coded track inlines these formulas for teaching;
   when learners see `(5, 10)` they should see the same constant the
   library uses).

### Run the curated eval against the three strategies

Lab 3 in either track does this — the saved files are
`eval_sets/<dataset>_custom_queries.json` and
`<dataset>_custom_qrels.json`. To re-evaluate without re-curating:

```python
import json, pathlib
from lib import load_beir_dataset  # for the corpus, even with custom qrels
queries = json.load(open("eval_sets/scifact_custom_queries.json"))
qrels   = json.load(open("eval_sets/scifact_custom_qrels.json"))
# … pass to the same evaluation loop as Lab 2 ...
```

### Use the advanced retrieval features

Everything advanced is in `phase4/`. Each phase4 script has a
`sys.path` shim at the top so it can import root modules even when
run via `python3 phase4/foo.py` from the repo root.

```bash
python3 phase4/query.py scifact --strategy dynamic --rerank
python3 phase4/test_harness.py --quick
python3 phase4/experiments/compare_fusion_strategies.py scifact
```

## Conventions to preserve

These are the conventions that have already burned someone — please
respect them:

1. **Do not hand-edit `.ipynb` files.** They are generated from
   `scripts/build_handcoded.py` and `scripts/build_agent.py`. Hand
   edits are lost on the next build. **If a learner asks you to
   change a notebook, edit the builder and rerun it.**

2. **Hand-coded notebooks inline the MQL.** The whole reason the
   hand-coded track exists is so learners *see* the
   `$vectorSearch` / `$search` / `$rankFusion` pipelines. Don't
   refactor `notebooks/` to call `retrieve.text_only(...)` — that
   defeats the purpose. The agent track (`agent-notebooks/`) is
   where library calls belong.

3. **`phase4/` has a `sys.path` shim at the top of every entry
   script.** It looks like this:
   ```python
   import os, sys
   sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
   ```
   For scripts under `phase4/experiments/`, the shim adds both the
   repo root *and* `phase4/`. If you add a new phase4 script that
   imports `lib` / `retrieve`, copy the shim.

4. **`DB_NAME = "voyage_context_demo"` is fixed across the repo.**
   Changing it orphans every ingested collection. If you really need
   to scope by environment, add a prefix at call sites — don't
   rename the constant.

5. **`METRIC_KS = (5, 10)` is the single source of truth.** If a
   notebook learner asks why their numbers don't match the library,
   verify the constant first.

6. **The `sec-10k` dataset's loader path is
   `phase4.data_loaders.sec_10k:load`** — this is why
   `phase4/__init__.py` exists as an empty marker. Don't delete it.

7. **Don't commit `.env` or `phase4/judge_cache.json*`.** They're
   gitignored for a reason (`.env` has secrets; the judge cache is
   4MB and regenerable).

## Setup checklist

For a fresh clone:

```bash
# 1. Install Python deps (works on macOS Python via brew)
pip3 install --break-system-packages \
    pymongo voyageai beir python-dotenv requests \
    openai pandas matplotlib nbformat tqdm

# 2. Create .env at the repo root (see README.md for what each var is)
#    VOYAGE_API_KEY, MONGODB_URI, OPENAI_API_KEY (last one optional)

# 3. Verify the repo imports cleanly
python3 -c "import lib, retrieve, lib_metrics, ingest; print('OK')"

# 4. Verify phase4 still works after any reorg
python3 phase4/query.py --list

# 5. Verify both notebook tracks rebuild from source
python3 scripts/build_handcoded.py
python3 scripts/build_agent.py

# 6. Verify both notebook tracks validate
python3 -c "
import pathlib, nbformat
for d in ['notebooks', 'agent-notebooks']:
    for n in sorted(pathlib.Path(d).glob('*.ipynb')):
        nbformat.validate(nbformat.read(str(n), as_version=4))
        print(f'  ✓ {n}')
"
```

If any of those six steps fails after your edits, you have broken the
lab — please fix before declaring done.

## Quick verification snippet

To confirm the data path works end-to-end on a small query (assumes
`scifact` is already ingested):

```python
import sys; sys.path.insert(0, ".")
import pymongo, voyageai
from lib import MONGODB_URI, MONGODB_BASE_URL, VOYAGE_API_KEY, DB_NAME, collection_name, embed_queries
from retrieve import text_only, vector_only, hybrid

coll   = pymongo.MongoClient(MONGODB_URI)[DB_NAME][collection_name("scifact")]
voyage = voyageai.Client(api_key=VOYAGE_API_KEY, base_url=MONGODB_BASE_URL)
q      = "effects of CRISPR-Cas9 on tumour growth"
v      = embed_queries(voyage, [q])[0]
print("lexical:", len(text_only(coll, q, top_k=5)), "results")
print("vector :", len(vector_only(coll, v, top_k=5)), "results")
print("hybrid :", len(hybrid(coll, v, q, top_k=5, alpha=0.8)), "results")
```

Expected output: each line ends in `5 results`. If any is empty, the
collection or its search indexes don't exist — re-run Lab 0 from the
agent track first.
