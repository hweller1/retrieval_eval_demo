# Retrieval evaluation — hands-on lab

A four-notebook lab on **information retrieval evaluation**, designed as
a companion to a course covering:

- **Lesson 1** — *What is retrieval evaluation?*
- **Lesson 2** — *Retrieval evaluation metrics* (Precision, Recall, NDCG, MRR)
- **Lesson 3** — *Understanding evaluation datasets* (queries, documents, qrels)
- **Lesson 4** — *Retrieval evaluation walkthrough* (lexical vs vector vs hybrid)

You'll measure a black-box retriever on a public BEIR benchmark, swap
the black box for vector and hybrid search, then curate your own
domain-specific evaluation set with an LLM bootstrap and human review.

Under the hood the lab uses **MongoDB Atlas Vector Search** with Voyage
AI's **`voyage-context-3`** contextualized chunk embeddings, accessed
through the MongoDB-hosted Voyage endpoint.

## Two tracks — pick the one that fits

This repo ships the lab in two parallel forms:

| Track | Folder | For | Style |
|---|---|---|---|
| **Hand-coded** *(default)* | `notebooks/` | Human learners stepping through the material | Inline MQL aggregation pipelines, inline metric formulas, MongoDB-quickstart-style numbered steps |
| **Agent-assisted** | `agent-notebooks/` | Users working with a coding agent (Claude Code, Cursor, etc.) | Library-driven, terser cells, narrative-heavy markdown — assumes you'll ask the agent when you want to see what `text_only(coll, q)` is actually doing |

Both tracks teach the same four labs in the same order:

```
00_setup_and_ingest.ipynb      pick a BEIR dataset, ingest into Atlas, build indexes
01_evaluate_blackbox.ipynb     Lessons 1+2 — measure BM25 with P@k / R@k / NDCG@k / MRR
02_swap_blackbox.ipynb         Lesson 4   — swap to $vectorSearch, then $rankFusion hybrid
03_curate_eval_set.ipynb       Lesson 3   — LLM-draft queries+labels, curate, re-evaluate
```

The hand-coded track is the default. If you're using a coding agent,
also read **`AGENTS.md`** before starting — it's the working manual for
that track.

## Setup

You need a free MongoDB Atlas account with the Voyage AI add-on
enabled. Both credentials come from Atlas — no separate Voyage AI
account required.

| Variable | Where to get it | Used in |
|---|---|---|
| `VOYAGE_API_KEY` | Atlas → AI Models → API Keys → "Create API Key" (starts with `al-`) | Labs 0, 2, 3 |
| `MONGODB_URI`    | Atlas → Database → Connect → Drivers — copy the `mongodb+srv://...` string with your DB user's password | all labs |
| `OPENAI_API_KEY` | platform.openai.com → API Keys — or substitute credentials for any OpenAI-compatible provider (see note below) | Lab 3 only |

Drop them in `.env` at the repo root:

```bash
VOYAGE_API_KEY=al-...
MONGODB_URI=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority
OPENAI_API_KEY=sk-...   # only for Lab 3 — see note below if using a different LLM provider
```

> **Using a different LLM provider for Lab 3?**
> Lab 3's LLM-as-a-judge feature defaults to OpenAI (`gpt-4o-mini`), but any
> OpenAI-compatible provider will work. Set `OPENAI_API_KEY` to your provider's
> API key and update `phase4/llm_client.py` — specifically the `DEFAULT_MODEL`
> constant and the `OpenAI(api_key=..., base_url=...)` client initializer — to
> point at your provider's base URL and model name.
>
> **Choosing the right judge model for your use case**
> `gpt-4o-mini` is a solid default for general-purpose relevance judgement, but
> the right choice depends on how hard your retrieval problem is. To get a
> personalised recommendation, paste the prompt below into any capable LLM chat
> (Claude, GPT-4o, Gemini, etc.) with your own details filled in:
>
> ```
> I am building an LLM-as-a-judge evaluator for a retrieval system.
> Please assess the difficulty of the judgement task and recommend a judge model tier.
>
> Application goal: <one sentence — e.g. "find relevant legal clauses in SEC filings">
> Data model: <brief description — e.g. "300-word chunks from 10-K filings, each tagged with section name">
> Query types: <e.g. "short keyword queries, natural-language questions, or both">
> Relevance definition: <e.g. "binary — the chunk either answers the question or it doesn't" / "graded — partial credit for tangentially related chunks">
>
> Based on the above, answer:
> 1. How hard is this judgement task? (easy / moderate / hard — and why)
> 2. Which judge model tier do you recommend? (e.g. gpt-4o-mini, gpt-4o, claude-3-5-sonnet, etc.)
> 3. What DEFAULT_MODEL string should I set in phase4/llm_client.py?
> 4. Any prompt-level changes that would improve judgement quality for this domain?
> ```
>
> Once you have an answer, set `DEFAULT_MODEL` in `phase4/llm_client.py` to the
> recommended model name. That one constant controls every LLM call in the judge pipeline.

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

Open the track you picked:

```bash
# Hand-coded (default)
jupyter notebook notebooks/

# Agent-assisted
jupyter notebook agent-notebooks/
```

Work through them in order. Lab 0 takes ~30–90 s (depends on the
dataset sample size); Labs 1–2 take a few seconds each; Lab 3 takes
~1–2 minutes (LLM calls).

## Supported BEIR datasets

| name | description |
|---|---|
| `scifact`    | scientific claim verification (300 queries / 5.2k abstracts) — **recommended for first run** |
| `nfcorpus`   | medical literature retrieval (323 queries / 3.6k docs) |
| `fiqa`       | financial Q&A — opinionated long answers (648 queries / 57k docs) |
| `arguana`    | counter-argument retrieval (1.4k queries / 8.7k arguments) |
| `scidocs`    | scientific paper retrieval (1k queries / 25k docs) |
| `trec-covid` | COVID-19 research retrieval (50 queries / 171k docs) |
| `touche2020` | controversial-topic argument retrieval (49 queries / 382k docs) |
| `quora`      | duplicate-question retrieval (10k queries / 523k docs) |

After the lab, advanced material — per-query routing, query
rewriters, cross-encoder reranking, LLM-as-a-judge evaluation, and
multi-dataset benchmark sweeps — lives in **`phase4/`**. See
`phase4/README.md`.

## Repo layout (one screen)

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

The `.ipynb` files are **generated** from `scripts/build_handcoded.py`
and `scripts/build_agent.py`. Edit the builders, then rerun them — don't
hand-edit the JSON.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `Authentication failed` from Mongo | URI password contains unencoded special chars (`@` → `%40`) |
| `Model voyage-context-3 is not supported` | You're hitting `/v1/embeddings` instead of `/v1/contextualizedembeddings` — use `lib.embed_contextualized()` |
| `index not found` on first query after ingest | Atlas takes 30–60 s to make a new search index queryable. Wait, then re-run. |
| Notebook can't import `lib` | Make sure the first cell's `sys.path.insert(...)` ran before the import. Jupyter must be started from the repo root or one level up. |
| `KeyError` during ingest sample step | Old `lib.py`; the sample filter handles qrels referencing missing doc IDs (in e.g. `arguana`). Pull latest. |
