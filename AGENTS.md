# AGENTS.md — context for coding agents

This file is for **coding agents** (Claude Code, Cursor, Copilot, etc.)
that a learner is using to step through the retrieval-evaluation lab.
It tells you where things live, what jobs you can do, what conventions
to preserve, and how to verify the lab still works after you change
something.

The four lab notebooks live at the repo root. Library code lives in
`src/`. Notebooks add `src/` to `sys.path` at startup so `import lib`,
`import retrieve`, etc. all resolve without package installation.

## Codebase map

```
voyage-context-3-testing/
├── README.md
├── AGENTS.md                         # this file
├── CLAUDE.md                         # working knowledge for Claude Code sessions
├── .env                              # VOYAGE_API_KEY, MONGODB_URI, OPENAI_API_KEY
├── 00_setup_and_ingest.ipynb
├── 01_evaluate_blackbox.ipynb
├── 02_swap_blackbox.ipynb
├── 03_curate_eval_set.ipynb
└── src/
    ├── lib.py                        # dataset registry, splitter, embedding helpers, constants
    ├── lib_metrics.py                # P@k, R@k, NDCG@k, MRR, AP/MAP
    ├── retrieve.py                   # vector_only / text_only / hybrid (Atlas pipelines)
    ├── ingest.py                     # BEIR → chunks → voyage-context-3 → MongoDB + indexes
    ├── scripts/                      # notebook builders (single source of truth)
    │   ├── _nb_helpers.py            # md/code/notebook/write primitives
    │   ├── build_handcoded.py        # generates the four root notebooks
    │   └── build_agent.py            # alternate builder (agent-style cells)
    └── phase4/                       # advanced — after-the-lab material
        ├── README.md
        ├── query.py, test_harness.py
        ├── query_rewriter.py, query_classifier.py, rerank.py, llm_client.py
        ├── llm_judge.py, judge_cache.json
        ├── data_loaders/             # non-BEIR (sec-10k)
        └── experiments/              # multi-dataset benchmark sweeps
```

### Module responsibilities

| Module | Exports | What's it for |
|---|---|---|
| `lib.py` | `DATASETS`, `VOYAGE_API_KEY`, `MONGODB_URI`, `DB_NAME`, `INDEX_NAME`, `load_beir_dataset`, `collection_name`, `split_text`, `embed_contextualized`, `embed_queries`, `require_credentials` | Config + the two unglamorous helpers (recursive splitter, contextualized-embeddings HTTP call) |
| `lib_metrics.py` | `precision_at_k`, `recall_at_k`, `ndcg_at_k`, `reciprocal_rank`, `average_precision`, `compute_query_metrics`, `aggregate_metrics`, `format_summary`, `METRIC_KS` | IR metric formulas. `METRIC_KS = (5, 10)` is the source of truth for which Ks are reported. |
| `retrieve.py` | `vector_only`, `text_only`, `hybrid`, `multi_query_retrieve`, `retrieve`, `MODES`, `DEFAULT_ALPHA`, `TEXT_INDEX_NAME` | The three retrieval modes as Python wrappers. `hybrid` uses Atlas's native `$rankFusion` (8.0+). |
| `ingest.py` | `ingest(dataset, corpus_sample)` | Full pipeline: load BEIR → split → embed → insert → build both search indexes. CLI: `python3 src/ingest.py <dataset>` |
| `phase4/*` | Everything advanced — see `src/phase4/README.md` | Per-query routing, query rewriters, reranker, LLM judge, multi-dataset experiment scripts |

## Common tasks

### Re-generate notebooks after editing

The `.ipynb` files are **generated**. Do not hand-edit them — your
changes will be overwritten the next time someone runs the builder.

```bash
python3 src/scripts/build_handcoded.py    # regenerates the four root notebooks
```

Edit the markdown/code blocks inside `src/scripts/build_handcoded.py`
instead. It uses a tiny DSL (`md("...")` and `code("...")`) for readability.

### Add a new BEIR dataset

1. Add an entry to `DATASETS` in `src/lib.py` with `url`, `folder`, `split`, `description`.
2. Update the dataset table in the notebook builder (`src/scripts/build_handcoded.py`, step 3 of `nb00()`).
3. Update the dataset table in `README.md`.
4. Rebuild notebooks.

### Add a new retrieval strategy

1. Add the function to `src/retrieve.py`. Match the existing signature:
   `(coll, q_vec, query_text, top_k, ...) -> list[dict]` where each
   row has at minimum `doc_id`, `text`, `score`.
2. Extend `MODES` and the `retrieve()` dispatcher in `src/retrieve.py`.
3. If the strategy is advanced (LLM-driven, reranker-dependent),
   put it under `src/phase4/` instead and import it from there.
4. The lab notebooks deliberately *inline* the MQL — do not refactor
   them to call `retrieve.<new_strategy>`. That's the source of their
   teaching value.

### Add a new metric

1. Add the formula to `src/lib_metrics.py` following the existing pattern:
   takes `(ranked: list[str], relevant: set | Mapping[str, int])`.
2. Add it to `compute_query_metrics()` so aggregations pick it up.
3. If `METRIC_KS` needs to change, change it there — *not* in any
   notebook (the notebooks inline these formulas for teaching; when
   learners see `(5, 10)` they should see the same constant the
   library uses).

### Run the curated eval against the three strategies

Lab 3 does this — the saved files are
`eval_sets/<dataset>_custom_queries.json` and
`<dataset>_custom_qrels.json`. To re-evaluate without re-curating:

```python
import json, pathlib, sys
sys.path.insert(0, "src")
from lib import load_beir_dataset
queries = json.load(open("eval_sets/scifact_custom_queries.json"))
qrels   = json.load(open("eval_sets/scifact_custom_qrels.json"))
# … pass to the same evaluation loop as Lab 2 ...
```

### Use the advanced retrieval features

Everything advanced is in `src/phase4/`. Each phase4 script has a
`sys.path` shim at the top so it can import `src/` modules when run
from the repo root.

```bash
python3 src/phase4/query.py scifact --strategy dynamic --rerank
python3 src/phase4/test_harness.py --quick
python3 src/phase4/experiments/compare_fusion_strategies.py scifact
```

## Conventions to preserve

1. **Do not hand-edit `.ipynb` files.** They are generated from
   `src/scripts/build_handcoded.py`. Hand edits are lost on the next
   build. **If a learner asks you to change a notebook, edit the
   builder and rerun it.**

2. **Notebooks inline the MQL.** The whole reason these notebooks
   exist is so learners *see* the `$vectorSearch` / `$search` /
   `$rankFusion` pipelines. Don't refactor cells to call
   `retrieve.text_only(...)` — that defeats the purpose.

3. **`src/phase4/` has a `sys.path` shim at the top of every entry
   script.** It adds `src/` (two levels up from `phase4/`) and
   `src/phase4/` to the path. If you add a new phase4 script that
   imports `lib` / `retrieve`, copy the shim:
   ```python
   import os, sys
   sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
   ```

4. **`DB_NAME = "voyage_context_demo"` is fixed across the repo.**
   Changing it orphans every ingested collection.

5. **`METRIC_KS = (5, 10)` is the single source of truth** for which
   Ks are reported. Change it in `src/lib_metrics.py` only.

6. **The `sec-10k` dataset's loader path is
   `phase4.data_loaders.sec_10k:load`** — this is why
   `src/phase4/__init__.py` exists as an empty marker. Don't delete it.

7. **Don't commit `.env` or `src/phase4/judge_cache.json`.** They're
   gitignored for a reason.

## Setup checklist

For a fresh clone:

```bash
# 1. Install Python deps (works on macOS Python via brew)
pip3 install --break-system-packages \
    pymongo voyageai beir python-dotenv requests \
    openai pandas matplotlib nbformat tqdm

# 2. Create .env at the repo root (see README.md for what each var is)
#    VOYAGE_API_KEY, MONGODB_URI, OPENAI_API_KEY (last one optional, Notebook 3 only)
#
#    The default judge model is gpt-4o-mini (set in src/phase4/llm_client.py).
#    If the retrieval problem is domain-specific or uses graded relevance,
#    a stronger model will give better grades. See "Curating your own evaluation
#    dataset" in README.md for a prompt to get a tailored recommendation.

# 3. Verify the repo imports cleanly
python3 -c "import sys; sys.path.insert(0, 'src'); import lib, retrieve, lib_metrics, ingest; print('OK')"

# 4. Verify phase4 still works after any reorg
python3 src/phase4/query.py --list

# 5. Verify notebooks rebuild from source
python3 src/scripts/build_handcoded.py

# 6. Verify notebooks validate
python3 -c "
import pathlib, nbformat
for n in sorted(pathlib.Path('.').glob('*.ipynb')):
    nbformat.validate(nbformat.read(str(n), as_version=4))
    print(f'  ✓ {n}')
"
```

If any of those steps fails after your edits, you have broken the
lab — please fix before declaring done.

## Quick verification snippet

To confirm the data path works end-to-end on a small query (assumes
`scifact` is already ingested):

```python
import sys; sys.path.insert(0, "src")
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
collection or its search indexes don't exist — re-run Notebook 0 first.
