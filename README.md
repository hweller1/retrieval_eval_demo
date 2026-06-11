# Retrieval evaluation — hands-on lab

A four-notebook lab on **information retrieval evaluation**. You'll measure a
retriever on a public benchmark, swap in vector and hybrid search, and finish
by curating your own domain-specific evaluation set with an LLM bootstrap.

Under the hood the lab uses **MongoDB Atlas Vector Search** with Voyage AI's
**`voyage-context-3`** contextualized chunk embeddings, accessed through the
MongoDB-hosted Voyage endpoint.

## What does each notebook teach?

| Notebook | Goal | Jump here if you want to… |
|---|---|---|
| `00_setup_and_ingest` | Ingest a BEIR dataset into Atlas and build vector + lexical indexes | Get the environment wired up, or understand how contextualized embeddings are stored |
| `01_evaluate_blackbox` | Measure a BM25 retriever end-to-end — compute P@k, R@k, NDCG@k, MRR by hand | Understand what retrieval metrics mean and how to compute them |
| `02_swap_blackbox` | Swap BM25 for vector search, then hybrid (`$rankFusion`), and compare all three | See when vector beats lexical (and vice versa), or tune a hybrid blend weight |
| `03_curate_eval_set` | Bootstrap a domain-specific eval set: LLM drafts queries + labels, you curate, then re-evaluate | Build an eval set for your own data, or understand why domain-specific qrels matter |

Work through them in order the first time. After that, each notebook is
self-contained enough to revisit on its own.

## Two tracks — pick the one that fits

| Track | Folder | Pick this if… |
|---|---|---|
| **Hand-coded** *(default)* | `notebooks/` | You want to read every MQL pipeline and metric formula inline — MongoDB-quickstart style, nothing hidden |
| **Agent-assisted** | `agent-notebooks/` | You're working with a coding agent (Claude Code, Cursor, etc.) and prefer short cells with narrative markdown — ask the agent when you want to see what a library call does under the hood |

Both tracks cover the same four notebooks in the same order.
If you're using a coding agent, also read **`AGENTS.md`** — it's the working
manual for that track.

## Setup

You need a free MongoDB Atlas account with the Voyage AI add-on enabled. Both
`VOYAGE_API_KEY` and `MONGODB_URI` come from Atlas — no separate Voyage AI
account required.

| Variable | Where to get it | Used in |
|---|---|---|
| `VOYAGE_API_KEY` | Atlas → AI Models → API Keys → "Create API Key" (starts with `al-`) | Notebooks 0, 2, 3 |
| `MONGODB_URI` | Atlas → Database → Connect → Drivers — copy the `mongodb+srv://...` string | All notebooks |
| `OPENAI_API_KEY` | platform.openai.com → API Keys — or any OpenAI-compatible provider | Notebook 3 only |

Drop them in `.env` at the repo root:

```bash
VOYAGE_API_KEY=al-...
MONGODB_URI=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority
OPENAI_API_KEY=sk-...   # only for Notebook 3
```

Atlas prerequisites:

- A cluster (M0 free tier is fine for `scifact` / `nfcorpus`)
- A DB user with read/write on `voyage_context_demo`
- Your IP allowlisted (or `0.0.0.0/0` for testing)
- The **Voyage AI add-on** enabled so `AI Models` appears in the Atlas sidebar

Install the Python deps:

```bash
pip3 install --break-system-packages \
    pymongo voyageai beir python-dotenv requests \
    openai pandas matplotlib nbformat tqdm notebook
```

## Run the lab

```bash
# Hand-coded (default)
jupyter notebook notebooks/

# Agent-assisted
jupyter notebook agent-notebooks/
```

Lab 0 takes ~30–90 s depending on dataset sample size. Labs 1–2 take a few
seconds each. Lab 3 takes ~1–2 minutes (LLM calls).

## Supported BEIR datasets

| name | description |
|---|---|
| `scifact` | scientific claim verification (300 queries / 5.2k abstracts) — **recommended for first run** |
| `nfcorpus` | medical literature retrieval (323 queries / 3.6k docs) |
| `fiqa` | financial Q&A — opinionated long answers (648 queries / 57k docs) |
| `arguana` | counter-argument retrieval (1.4k queries / 8.7k arguments) |
| `scidocs` | scientific paper retrieval (1k queries / 25k docs) |
| `trec-covid` | COVID-19 research retrieval (50 queries / 171k docs) |
| `touche2020` | controversial-topic argument retrieval (49 queries / 382k docs) |
| `quora` | duplicate-question retrieval (10k queries / 523k docs) |

Start with `scifact` — it's small, fast to ingest, and all three retrieval
strategies produce meaningfully different scores on it.

## Curating your own evaluation dataset (Notebook 3)

`03_curate_eval_set.ipynb` walks you through bootstrapping a domain-specific
evaluation set on your own corpus: an LLM drafts candidate queries and
relevance labels, you review and curate them, then re-run the metrics from
Notebooks 1–2 against your new qrels.

### Choosing the right judge model

The notebook uses `gpt-4o-mini` by default (controlled by `DEFAULT_MODEL` in
`phase4/llm_client.py`). That's a solid general-purpose choice. For
domain-specific or nuanced relevance judgements — legal documents, medical
literature, graded relevance — a stronger model will produce more reliable
grades.

To get a recommendation tailored to your use case, paste the prompt below into
any capable LLM chat (Claude, GPT-4o, Gemini, etc.) with your own details:

```
I am building an LLM-as-a-judge evaluator for a retrieval system.
Please assess the difficulty of the judgement task and recommend a judge model tier.

Application goal: <one sentence — e.g. "find relevant legal clauses in SEC filings">
Data model: <brief description — e.g. "300-word chunks from 10-K filings, each tagged with section name">
Query types: <e.g. "short keyword queries, natural-language questions, or both">
Relevance definition: <e.g. "binary — the chunk either answers the question or it doesn't" / "graded — partial credit for tangentially related chunks">

Based on the above, answer:
1. How hard is this judgement task? (easy / moderate / hard — and why)
2. Which judge model tier do you recommend? (e.g. gpt-4o-mini, gpt-4o, claude-3-5-sonnet, etc.)
3. What DEFAULT_MODEL string should I set in phase4/llm_client.py?
4. Any prompt-level changes that would improve judgement quality for this domain?
```

Once you have an answer, set `DEFAULT_MODEL` in `phase4/llm_client.py` — that
one constant controls every LLM call in the judge pipeline. To switch
providers, also pass `base_url="https://<provider>/v1"` to the `OpenAI()`
constructor in that file and set `OPENAI_API_KEY` to your provider's key.

## Going further

After the lab, `phase4/` has advanced material: per-query routing, query
rewriters, cross-encoder reranking, LLM-as-a-judge evaluation, and
multi-dataset benchmark sweeps. See `phase4/README.md`.

## Repo layout

```
voyage-context-3-testing/
├── README.md             # this file
├── AGENTS.md             # working manual for the agent track
├── CLAUDE.md             # working knowledge for Claude Code sessions
├── lib.py                # dataset registry, splitter, embedding helpers
├── lib_metrics.py        # P@k, R@k, NDCG@k, MRR, AP/MAP
├── retrieve.py           # vector / text / hybrid wrappers
├── ingest.py             # BEIR → chunks → embeddings → MongoDB + indexes
├── notebooks/            # HAND-CODED lab (default)
├── agent-notebooks/      # AGENT-FRIENDLY lab
├── scripts/              # notebook builders (single source of truth)
└── phase4/               # advanced / after-the-lab material
```

The `.ipynb` files are **generated** from `scripts/build_handcoded.py` and
`scripts/build_agent.py`. Edit the builders, then rerun them — don't hand-edit
the JSON.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `Authentication failed` from Mongo | URI password contains unencoded special chars (`@` → `%40`) |
| `Model voyage-context-3 is not supported` | You're hitting `/v1/embeddings` instead of `/v1/contextualizedembeddings` — use `lib.embed_contextualized()` |
| `index not found` on first query after ingest | Atlas takes 30–60 s to make a new search index queryable. Wait, then re-run. |
| Notebook can't import `lib` | Make sure the first cell's `sys.path.insert(...)` ran before the import. Jupyter must be started from the repo root or one level up. |
| `KeyError` during ingest sample step | Old `lib.py`; the sample filter handles qrels referencing missing doc IDs (e.g. `arguana`). Pull latest. |
