"""
Build the four AGENT-TRACK lab notebooks under agent-notebooks/.

The agent track leans on the project's library functions (text_only,
hybrid, compute_query_metrics, ...) — short cells, narrative-heavy
markdown. It's optimized for users with a coding agent at their side
who can navigate the library when they have a question.

Hand-coded version (for human learners stepping through MQL) lives in
scripts/build_handcoded.py.

Regenerate:
    python3 scripts/build_agent.py
"""

from __future__ import annotations

import pathlib

from _nb_helpers import md, code, write


NB_DIR = pathlib.Path(__file__).resolve().parent.parent / "agent-notebooks"


# ─────────────────────────────────────────────────────────────────────────────
# Shared setup snippet — adds repo root to sys.path so notebooks import
# lib.py / retrieve.py / lib_metrics.py / ingest.py from one level up.
# ─────────────────────────────────────────────────────────────────────────────

SETUP_CELL = """
import os, sys
# Notebooks live in agent-notebooks/. Add the repo root to the path so
# we can import the lab's library modules.
_REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
"""


# ─────────────────────────────────────────────────────────────────────────────
# Notebook 00 — Setup and ingest
# ─────────────────────────────────────────────────────────────────────────────

def nb00() -> list[dict]:
    return [
        md("""
        # 00 — Setup and ingest

        **One-time setup notebook.** Run this once per BEIR dataset you want
        to work with. The remaining lab notebooks (`01`, `02`, `03`) read
        from the collections this notebook builds.

        ## What this notebook does

        1. Checks your `.env` for the credentials the lab needs.
        2. Lets you pick one (or more) of 8 BEIR retrieval datasets.
        3. Ingests the dataset into MongoDB Atlas:
           - Splits documents into chunks.
           - Embeds the chunks with `voyage-context-3` (a *contextualized*
             chunk embedder — each chunk's vector is informed by the whole
             document it came from).
           - Writes chunks + vectors to a collection.
           - Builds both a **vector** search index and a **lexical (BM25)**
             search index so we can compare retrieval strategies later.

        Once a dataset is ingested, you can re-run the other notebooks
        against it as many times as you want without re-ingesting.

        > **Cost / time:** Each ingest at the default sample of 500
        > documents takes ~30–90 s and uses a few cents of Voyage credit.
        """),
        md("""
        ## Prerequisites

        Add the following to a file named `.env` at the repo root:

        ```
        VOYAGE_API_KEY=al-...        # from Atlas → AI Models → API Keys
        MONGODB_URI=mongodb+srv://...
        OPENAI_API_KEY=sk-...        # only needed in notebook 03
        ```

        Then install the Python deps from the repo root:

        ```
        pip3 install --break-system-packages \\
            requests pymongo voyageai beir python-dotenv \\
            tqdm openai matplotlib pandas
        ```
        """),
        code(SETUP_CELL),
        code("""
        # Confirm the credentials are loaded.
        import lib
        lib.require_credentials()
        print("VOYAGE_API_KEY loaded:", bool(lib.VOYAGE_API_KEY))
        print("MONGODB_URI loaded:   ", bool(lib.MONGODB_URI))
        """),
        md("""
        ## Pick a dataset

        BEIR (Benchmarking IR) is a collection of retrieval datasets with
        human-curated relevance judgements (qrels). Each entry below is a
        full `(corpus, queries, qrels)` triple:

        | name | what it tests |
        |---|---|
        | `scifact` | scientific claim verification (300 queries / 5.2k abstracts) |
        | `nfcorpus` | medical literature retrieval (323 queries / 3.6k docs) |
        | `fiqa` | financial Q&A — opinionated long answers (648 queries / 57k docs) |
        | `arguana` | counter-argument retrieval (1.4k queries / 8.7k arguments) |
        | `scidocs` | scientific paper retrieval (1k queries / 25k docs) |
        | `trec-covid` | COVID-19 research retrieval (50 queries / 171k docs) |
        | `touche2020` | controversial-topic argument retrieval (49 queries / 382k docs) |
        | `quora` | duplicate-question retrieval (10k queries / 523k docs) |

        **Recommendation for a first run: `scifact`.** It's small, the
        queries are short scientific claims, and the qrels are clean. Once
        you've worked through the lab on `scifact`, try a different domain
        (`fiqa` for financial, `nfcorpus` for medical) and watch how
        metrics change.
        """),
        code("""
        # ── EDIT ME ─────────────────────────────────────────────────────
        DATASET       = "scifact"   # one of the names above
        CORPUS_SAMPLE = 500          # how many docs to ingest
        # ────────────────────────────────────────────────────────────────

        from lib import DATASETS
        assert DATASET in DATASETS, f"Unknown dataset: {DATASET}"
        print(f"Will ingest {DATASET}: {DATASETS[DATASET]['description']}")
        """),
        md("""
        ## Ingest

        This runs the full pipeline: download → chunk → embed → write to
        MongoDB → build both indexes. A status bar in stdout tracks
        progress. Indexes take 30–60 s to become queryable after writes
        finish; the script waits for that automatically.

        Re-running this cell **drops** the dataset's collection and
        re-ingests from scratch.
        """),
        code("""
        from ingest import ingest
        ingest(DATASET, corpus_sample=CORPUS_SAMPLE)
        """),
        md("""
        ## Verify

        Sanity-check that the collection and both indexes exist and are
        queryable. If the text index is still building, the next notebook
        will fail with an `index not found` error; wait a minute and
        re-run the verification cell.
        """),
        code("""
        import pymongo
        from lib import MONGODB_URI, DB_NAME, INDEX_NAME, collection_name
        from retrieve import TEXT_INDEX_NAME

        client = pymongo.MongoClient(MONGODB_URI)
        coll   = client[DB_NAME][collection_name(DATASET)]

        print(f"Collection: {coll.full_name}")
        print(f"  chunks  : {coll.estimated_document_count():,}")
        print()
        print(f"Search indexes:")
        for idx in coll.list_search_indexes():
            print(f"  {idx['name']:<25}  queryable={idx.get('queryable', False)}")
        client.close()
        """),
        md("""
        ## Next

        Open **`01_evaluate_blackbox.ipynb`** to see how IR evaluation
        works — we'll treat lexical (BM25) retrieval as a "black box" and
        measure its quality with the same metrics you'd use to compare
        any retrieval system.
        """),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Notebook 01 — Lessons 1 + 2: what is eval, and the four metrics
# ─────────────────────────────────────────────────────────────────────────────

def nb01() -> list[dict]:
    return [
        md("""
        # 01 — Evaluate a black-box retriever

        **Companion to Lessons 1 & 2.**

        > *Imagine you're building a recruiting application. You have a
        > pool of résumés in a database and a job opening to fill. A
        > hiring manager needs a shortlist by the end of the day. You run
        > your retrieval system and get back a ranked list of candidates.*
        > *But how do you know if it's any good?*

        That's the question this notebook answers, end-to-end, on real
        data. The "retrieval system" here is **lexical (BM25) search** —
        the same family of keyword-matching algorithms that has powered
        full-text search engines for decades. We treat it as a black box:
        text goes in, a ranked list of candidate documents comes out. Our
        job is to score that ranked list against ground-truth relevance
        judgements.

        ## What you'll do

        1. Load a BEIR dataset (the same one you ingested in notebook 00)
           and look at one of its queries and its qrels.
        2. Run BM25 retrieval for that query.
        3. Compute **Precision@k**, **Recall@k**, **NDCG@k**, and **MRR**
           by hand on the result — once you've felt how the formulas
           work, you'll use the library functions for the rest.
        4. Loop over many queries and aggregate the metrics into a single
           score for the system.
        """),
        code(SETUP_CELL),
        code("""
        # Make sure DATASET matches what you ingested in notebook 00.
        DATASET = "scifact"

        import pymongo
        from lib import MONGODB_URI, DB_NAME, collection_name, load_beir_dataset

        client = pymongo.MongoClient(MONGODB_URI)
        coll   = client[DB_NAME][collection_name(DATASET)]
        corpus, queries, qrels, info = load_beir_dataset(DATASET)

        print(f"Dataset      : {DATASET}  ({info['description']})")
        print(f"Corpus       : {len(corpus):,} documents")
        print(f"Test queries : {len(queries):,}")
        print(f"Collection   : {coll.estimated_document_count():,} chunks ingested")
        """),
        md("""
        ## What is an evaluation dataset?

        Every BEIR dataset is a triple:

        - **`corpus`** — the haystack: a `dict` of `{doc_id: {"title": ..., "text": ...}}`.
        - **`queries`** — what users would ask: a `dict` of `{query_id: query_text}`.
        - **`qrels`** ("query relevance" judgements) — the answer key: a
          `dict` of `{query_id: {doc_id: relevance_score}}`. A non-zero
          score means a human judged that document relevant to that query.

        The qrels are sometimes called the **judgement list**, **golden
        dataset**, or **ground truth**. They're what makes evaluation
        possible: without a known answer, we can't measure error.

        Let's pull one query and look at it.
        """),
        code("""
        from retrieve import text_only

        # Pick the first query whose BM25 top-10 actually contains at least
        # one of its relevant docs — otherwise every metric below would
        # be 0. (Queries that share no keywords with their relevant docs
        # are great examples of where vector beats lexical, but they make a
        # confusing first metric walkthrough.)
        def has_bm25_hit(qid):
            q_qrels = qrels.get(qid, {})
            rel_set = {did for did, s in q_qrels.items() if s > 0}
            if not rel_set:
                return False
            ranked = text_only(coll, queries[qid], top_k=10)
            return any(r['doc_id'] in rel_set for r in ranked)

        sample_qid    = next(qid for qid in queries if has_bm25_hit(qid))
        sample_query  = queries[sample_qid]
        sample_qrels  = qrels.get(sample_qid, {})
        relevant_docs = {did: s for did, s in sample_qrels.items() if s > 0}

        print(f"Query ID     : {sample_qid}")
        print(f"Query text   : {sample_query!r}")
        print(f"Relevant docs (per the qrels): {len(relevant_docs)}")
        for did, score in list(relevant_docs.items())[:5]:
            title = corpus.get(did, {}).get('title', '<not in corpus>')
            print(f"  doc {did:<12}  relevance={score}  title={title!r}")
        """),
        md("""
        ### Graded vs binary relevance

        Notice the qrels are numbers, not booleans. BEIR datasets use
        **graded relevance**:

        - `0` — not relevant
        - `1` — relevant
        - `2` — highly relevant (in datasets that support it)

        Binary relevance is graded relevance forced to `{0, 1}`. Most
        metrics work with either; **NDCG** is the one that benefits the
        most from the extra resolution (we'll see why below).
        """),
        md("""
        ## Run the black-box retriever

        We'll call `retrieve.text_only(...)` — pure BM25 search via
        MongoDB Atlas's `$search`. Input: the query string. Output: a
        ranked list of documents, best-first, with a score.
        """),
        code("""
        from retrieve import text_only

        TOP_K = 10
        ranked = text_only(coll, sample_query, top_k=TOP_K)

        print(f"Top {TOP_K} BM25 results for query {sample_qid!r}:")
        print(f"  {sample_query!r}")
        print()
        for rank, row in enumerate(ranked, 1):
            grade = sample_qrels.get(row['doc_id'], 0)
            tag = "★" if grade > 0 else " "
            print(f"  {rank:>2}. {tag}  doc {row['doc_id']:<12}  "
                  f"score={row['score']:.3f}  grade={grade}  "
                  f"{row['title'][:60]!r}")
        """),
        md("""
        Each row marked ★ is one the qrels say is relevant; unmarked rows
        are either confirmed irrelevant (grade 0) or not judged at all.
        Take a moment to read the unmarked results — would *you* call any
        of them relevant? That gut check is what motivates **graded** vs
        **binary** relevance and the careful curation that goes into
        qrels.
        """),
        md("""
        ## Metric 1 — Precision@k

        > *Of the documents the system returned, what fraction are relevant?*

        Precision answers the **false positive** question: "how much
        noise is in my shortlist?" If we returned 5 candidates and 4 of
        them are relevant, Precision@5 = 0.8.

        $$\\text{Precision@k} = \\frac{|\\text{relevant} \\cap \\text{top-}k|}{k}$$
        """),
        code("""
        # By hand for k=5
        top5 = [row['doc_id'] for row in ranked[:5]]
        relevant_ids = {did for did, s in sample_qrels.items() if s > 0}

        hits_in_top5 = sum(1 for did in top5 if did in relevant_ids)
        precision_at_5 = hits_in_top5 / 5

        print(f"Top-5 doc IDs       : {top5}")
        print(f"Relevant in top-5   : {hits_in_top5}")
        print(f"Precision@5         : {precision_at_5:.3f}")

        # Cross-check with the library
        from lib_metrics import precision_at_k
        print(f"Library Precision@5 : {precision_at_k(top5, sample_qrels, 5):.3f}")
        """),
        md("""
        ## Metric 2 — Recall@k

        > *Of all the relevant documents that exist, what fraction did we
        > return?*

        Recall answers the **false negative** question: "how many
        relevant documents did I miss?" If 10 docs in the corpus are
        relevant and 7 of them appear in the top-k, Recall@k = 0.7.

        $$\\text{Recall@k} = \\frac{|\\text{relevant} \\cap \\text{top-}k|}{|\\text{relevant}|}$$

        > **Vocabulary note.** "Recall" also shows up in *vector index*
        > benchmarks where it means the fraction of true nearest
        > neighbours an approximate index returned. Same word, different
        > reference point — that's an index-quality metric, this is a
        > retrieval-quality metric.
        """),
        code("""
        from lib_metrics import recall_at_k

        top10 = [row['doc_id'] for row in ranked[:10]]
        hits_in_top10 = len(set(top10) & relevant_ids)

        print(f"# relevant docs in corpus : {len(relevant_ids)}")
        print(f"# relevant in top-10      : {hits_in_top10}")
        print(f"Recall@10  (by hand)      : {hits_in_top10 / max(1, len(relevant_ids)):.3f}")
        print(f"Recall@10  (library)      : {recall_at_k(top10, sample_qrels, 10):.3f}")
        """),
        md("""
        ### Precision/Recall trade-off

        These two metrics pull in opposite directions:

        - **Want higher precision?** Return fewer, higher-confidence
          results. You'll miss more — recall drops.
        - **Want higher recall?** Cast a wider net. You'll pull in more
          noise — precision drops.

        Which one matters more is an **application-level** decision. A
        legal-discovery tool can't afford to miss a relevant precedent
        (favour recall). A user-facing search box can't afford to bury
        useful results under junk (favour precision). The same retriever
        can be the right or wrong tool depending on the cost of each
        error type.
        """),
        md("""
        ## Metric 3 — NDCG@k

        Precision and Recall both treat the top-k as a *set* — they
        ignore the order within it. But users look at ranked lists
        top-down; a relevant document at rank 1 is much more valuable
        than the same document at rank 9. **NDCG** captures that.

        NDCG ("Normalized Discounted Cumulative Gain") has three pieces:

        1. **Gain** — the relevance grade of each retrieved document
           (so a graded judgement of 2 contributes more than a 1).
        2. **Discount** — divide the gain by `log₂(rank + 1)`, so
           position 1 gets weight `1 / log₂(2) = 1.0`, position 5 gets
           `1 / log₂(6) ≈ 0.39`, position 10 gets `≈ 0.29`. Lower
           positions count less.
        3. **Normalization** — divide by the score of an *ideal*
           ranking (relevant docs sorted from highest grade to lowest).
           That bounds NDCG to `[0, 1]`. A perfect ranking scores 1.0.

        $$\\text{DCG@k} = \\sum_{i=1}^{k} \\frac{\\text{grade}(d_i)}{\\log_2(i+1)}
        \\quad\\quad \\text{NDCG@k} = \\frac{\\text{DCG@k}}{\\text{IDCG@k}}$$
        """),
        code("""
        import math
        from lib_metrics import ndcg_at_k, dcg_at_k

        # Compute DCG@5 by hand
        ranked_ids_5 = [row['doc_id'] for row in ranked[:5]]
        print("DCG calculation (k=5):")
        dcg = 0.0
        for rank, did in enumerate(ranked_ids_5, 1):
            grade = sample_qrels.get(did, 0)
            disc  = math.log2(rank + 1)
            term  = grade / disc
            dcg  += term
            print(f"  rank {rank}: grade={grade}  discount=1/log2({rank+1})={1/disc:.3f}  term={term:.3f}")
        print(f"  → DCG@5 = {dcg:.3f}")

        # And IDCG@5 — the DCG of the best possible ranking
        ideal_grades = sorted(sample_qrels.values(), reverse=True)[:5]
        idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal_grades, 1) if g > 0)
        print(f"  → IDCG@5 = {idcg:.3f}  (best possible ordering of available grades)")
        print(f"  → NDCG@5 = DCG/IDCG = {dcg/idcg if idcg else 0:.3f}")
        print()
        print(f"Library NDCG@5  : {ndcg_at_k(ranked_ids_5, sample_qrels, 5):.3f}")
        print(f"Library NDCG@10 : {ndcg_at_k([r['doc_id'] for r in ranked[:10]], sample_qrels, 10):.3f}")
        """),
        md("""
        NDCG@k is **the** primary metric for embedding-model comparison;
        you'll see it on every public leaderboard (MTEB, RTEB, BEIR
        itself). It rewards systems that put the most relevant
        documents at the top and tolerates burying weaker (but
        still-relevant) ones lower.
        """),
        md("""
        ## Metric 4 — MRR (Mean Reciprocal Rank)

        > *On average, how high up does the first relevant document
        > appear?*

        Scan the ranked list from position 1 down until you hit a
        relevant doc. If it's at rank `r`, the reciprocal rank is
        `1/r`: 1.0 at position 1, 0.5 at position 2, 0.33 at position
        3, etc. Average across all queries to get MRR.

        MRR is most informative when there's **one** clearly best fit
        per query, or when you're evaluating a **reranker** whose only
        job is to bubble the best answer to the top.
        """),
        code("""
        from lib_metrics import reciprocal_rank

        ranked_ids = [row['doc_id'] for row in ranked]
        first_rel_rank = next(
            (i for i, did in enumerate(ranked_ids, 1) if did in relevant_ids),
            None,
        )
        if first_rel_rank:
            print(f"First relevant doc at rank {first_rel_rank}  →  RR = 1/{first_rel_rank} = {1/first_rel_rank:.3f}")
        else:
            print(f"No relevant doc retrieved at all  →  RR = 0")

        print(f"Library Reciprocal Rank : {reciprocal_rank(ranked_ids, sample_qrels):.3f}")
        """),
        md("""
        ## Putting it all together: evaluate over many queries

        A retrieval system isn't judged on one query — it's judged on a
        distribution of queries. We'll loop over the first `N` test
        queries, compute the per-query metric dict for each, and
        aggregate by averaging across queries (AP becomes MAP, others
        stay as means).
        """),
        code("""
        from lib import embed_queries
        from lib_metrics import compute_query_metrics, aggregate_metrics, format_summary

        N_QUERIES = 30   # bump up once you've confirmed it works
        TOP_K     = 10

        # Use a stable order so re-runs evaluate the same queries
        query_ids = list(queries.keys())[:N_QUERIES]

        per_query: list[dict] = []
        for qid in query_ids:
            q_text = queries[qid]
            ranked = text_only(coll, q_text, top_k=TOP_K)
            ranked_ids = [row['doc_id'] for row in ranked]
            metrics = compute_query_metrics(ranked_ids, qrels.get(qid, {}))
            per_query.append(metrics)

        agg = aggregate_metrics(per_query)
        print(f"BM25 lexical retrieval — {DATASET}, {N_QUERIES} queries")
        print(format_summary(agg))
        """),
        md("""
        ## Reading the aggregate

        Each metric on that single line tells you something different.
        For a quick gut-check:

        - **P@5 / P@10** — how clean the top of the list is. Higher =
          fewer false positives in what the user actually sees.
        - **R@5 / R@10** — coverage. Higher = fewer false negatives in
          the top-k window. Watch this when missing a relevant doc is
          costly (legal, medical, compliance).
        - **NDCG@5 / NDCG@10** — ranking quality. The number you'd
          quote when comparing two retrievers on a benchmark.
        - **MRR** — how prominently the *first* relevant result sits.
        - **MAP** — Mean Average Precision; a recall-and-rank summary
          across all positions where relevant docs land.

        Save these numbers — you'll compare them to a different black
        box in notebook 02.

        ## Next

        Open **`02_swap_blackbox.ipynb`** to swap BM25 for vector and
        hybrid retrieval over the same dataset, and see how the metric
        deltas explain *when* and *why* one approach beats another.
        """),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Notebook 02 — Lesson 4: lexical vs vector vs hybrid
# ─────────────────────────────────────────────────────────────────────────────

def nb02() -> list[dict]:
    return [
        md("""
        # 02 — Swap the black box

        **Companion to Lesson 4.**

        In notebook 01 you measured a lexical (BM25) retriever on a
        BEIR dataset. Now we'll keep everything else fixed — same
        dataset, same queries, same metrics — and swap out the
        retrieval strategy. This is the experimental discipline that
        lets you say something meaningful when comparing systems:
        change one variable, hold everything else constant.

        ## The three strategies

        | name | how it works | strength |
        |---|---|---|
        | **lexical (BM25)** | matches the *words* in the query against the *words* in each doc | exact terms, names, IDs |
        | **vector** | embeds the query and each doc; ranks by cosine similarity | semantic match, paraphrase |
        | **hybrid** | runs both and fuses the rankings (weighted RRF) | the best of both |

        Lexical is what powered search before embeddings; vector is the
        modern semantic approach (today's reference point is **Voyage's
        `voyage-context-3`**, which embeds each chunk *in the context
        of its parent document* — see notebook 00 for the ingest
        pipeline). Hybrid is the production default at most companies
        because the two signals are complementary.
        """),
        code(SETUP_CELL),
        code("""
        # Match this to whatever you ingested.
        DATASET   = "scifact"
        N_QUERIES = 30
        TOP_K     = 10

        import pymongo
        import voyageai
        from lib import (
            MONGODB_BASE_URL, MONGODB_URI, DB_NAME, VOYAGE_API_KEY,
            collection_name, embed_queries, load_beir_dataset,
        )
        from retrieve import text_only, vector_only, hybrid
        from lib_metrics import compute_query_metrics, aggregate_metrics, format_summary

        client     = pymongo.MongoClient(MONGODB_URI)
        coll       = client[DB_NAME][collection_name(DATASET)]
        voyage     = voyageai.Client(api_key=VOYAGE_API_KEY, base_url=MONGODB_BASE_URL)
        corpus, queries, qrels, info = load_beir_dataset(DATASET)

        query_ids = list(queries.keys())[:N_QUERIES]
        q_texts   = [queries[qid] for qid in query_ids]
        q_vecs    = embed_queries(voyage, q_texts)
        print(f"Embedded {len(q_vecs)} queries with voyage-3-large "
              f"({len(q_vecs[0])} dims).")
        """),
        md("""
        ## Run all three strategies

        We'll evaluate each strategy on the same set of queries and
        collect aggregate metrics in a single table.
        """),
        code("""
        STRATEGIES = {
            "lexical (BM25)" : lambda q_vec, q_text: text_only(coll, q_text, top_k=TOP_K),
            "vector"         : lambda q_vec, q_text: vector_only(coll, q_vec, top_k=TOP_K),
            "hybrid α=0.8"   : lambda q_vec, q_text: hybrid(coll, q_vec, q_text, top_k=TOP_K, alpha=0.8),
        }

        rows = {}
        for name, run in STRATEGIES.items():
            per_query = []
            for qid, q_vec, q_text in zip(query_ids, q_vecs, q_texts):
                ranked = run(q_vec, q_text)
                ranked_ids = [r['doc_id'] for r in ranked]
                per_query.append(compute_query_metrics(ranked_ids, qrels.get(qid, {})))
            rows[name] = aggregate_metrics(per_query)
            print(f"{name:<20}  {format_summary(rows[name])}")
        """),
        md("""
        ## Compare side-by-side

        Same data in tabular form so the deltas are visible at a glance.
        """),
        code("""
        import pandas as pd
        df = pd.DataFrame(rows).T[["P@5", "R@5", "NDCG@5", "NDCG@10", "MRR", "MAP"]]
        df.round(3)
        """),
        md("""
        ## Hybrid weight `α` — controlling the blend

        The `hybrid(..., alpha=0.8)` call passes a weight that controls
        how much the **vector** ranking matters relative to the
        **lexical** one:

        - `alpha = 1.0` — vector only (equivalent to `vector_only`).
        - `alpha = 0.0` — text only (equivalent to `text_only`).
        - `alpha = 0.5` — naïve uniform fusion. *Often the wrong
          default* — it drags vector quality down when vector is the
          stronger signal.
        - `alpha = 0.8` — vector-favored. A reasonable starting point
          on most BEIR datasets where embeddings dominate.

        Let's sweep `α` to see how aggregate NDCG@10 moves.
        """),
        code("""
        ALPHAS = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
        sweep_ndcg = []
        for a in ALPHAS:
            per_query = []
            for qid, q_vec, q_text in zip(query_ids, q_vecs, q_texts):
                ranked = hybrid(coll, q_vec, q_text, top_k=TOP_K, alpha=a)
                ranked_ids = [r['doc_id'] for r in ranked]
                per_query.append(compute_query_metrics(ranked_ids, qrels.get(qid, {})))
            agg = aggregate_metrics(per_query)
            sweep_ndcg.append(agg["NDCG@10"])
            print(f"  α={a:.1f}  NDCG@10={agg['NDCG@10']:.3f}  MAP={agg['MAP']:.3f}")
        """),
        code("""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(ALPHAS, sweep_ndcg, marker="o")
        ax.set_xlabel("hybrid α  (0 = lexical, 1 = vector)")
        ax.set_ylabel("NDCG@10")
        ax.set_title(f"{DATASET} — NDCG@10 vs hybrid alpha ({N_QUERIES} queries)")
        ax.grid(alpha=0.3)
        plt.show()
        """),
        md("""
        ## Reading the comparison

        Three things to look for:

        1. **Does vector beat lexical on NDCG@10?** On most modern
           BEIR datasets, yes — embeddings handle paraphrase and
           topical overlap that BM25 misses. But it's not universal:
           datasets dominated by exact strings (codes, names, IDs)
           still favour lexical.

        2. **Does hybrid beat both?** Often yes, but only if `α` is
           weighted toward the stronger of the two single-mode
           strategies. The uniform `α=0.5` is the classic "hybrid
           hurts" trap — it drags vector quality down when vector is
           already the better signal.

        3. **Which metric improves most?** If `vector` and `hybrid`
           both lift Recall@10 a lot more than NDCG@10, it means
           they're *finding* more relevant docs but not necessarily
           putting them at the top. That's the signal that a
           **reranker** would help (see `phase4/` for a cross-encoder
           rerank stage).

        ## What this is NOT telling you

        BEIR's queries are **homogeneous within a dataset** — all
        scientific claims, or all health questions. A single
        well-chosen retrieval setup wins almost every query in such
        a benchmark. Real production traffic looks nothing like that
        — you'll get a mix of exact-match lookups, single-word topic
        queries, multi-hop compound questions, and chatty natural
        language all from the same users. Per-query routing /
        rewriting / reranking comes into its own there. See
        `phase4/query_classifier.py` for an example router.

        ## Next

        Open **`03_curate_eval_set.ipynb`** — BEIR is great for
        benchmarks but rarely matches your real users' queries. To
        know which strategy works in *your* context, you need *your
        own* eval data.
        """),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Notebook 03 — Lesson 3: curate your own evaluation dataset
# ─────────────────────────────────────────────────────────────────────────────

def nb03() -> list[dict]:
    return [
        md("""
        # 03 — Curate your own evaluation dataset

        **Companion to Lesson 3.**

        BEIR is a fixed benchmark — it tells you how a retrieval setup
        does on *scientific claim verification* or *medical literature*
        or whichever academic test set you picked. That's useful for
        comparing approaches in the abstract. It is *not* useful for
        answering the question your team actually has: "**does this
        retrieve well on the kind of queries our users ask, against
        the documents we actually have?**" For that you need a
        domain-specific evaluation set built on your own corpus.

        ## What you'll build

        A small **judgement list** (queries + relevance labels) over
        the corpus you already ingested, using the standard
        bootstrap pattern:

        1. **Sample** documents from your corpus.
        2. **Draft queries** with an LLM — one realistic query per
           doc, in the voice of an actual user.
        3. **Pool candidates** — retrieve top-K for each draft query
           with whatever black box you want (here: hybrid α=0.8).
        4. **Grade** the pooled `(query, doc)` pairs with an LLM
           judge on a 0–3 scale.
        5. **Curate** — *you* spot-check and edit. The LLM is fast
           and consistent, but it doesn't know what *you* mean by
           relevant. The human-in-the-loop step is what makes the
           dataset trustworthy.
        6. **Save & re-evaluate** — write the judgements to disk and
           rerun the metrics from notebook 01 against them.

        > **Requires `OPENAI_API_KEY`** in `.env`. The full bootstrap
        > on 20 docs ≈ 60 LLM calls — costs a few cents on
        > `gpt-4o-mini`.
        """),
        code(SETUP_CELL),
        code("""
        DATASET   = "scifact"   # match what you ingested
        N_DOCS    = 20          # sample size for the bootstrap
        TOP_K     = 8           # candidates per query to grade
        MODEL_LLM = "gpt-4o-mini"

        import os
        assert os.environ.get("OPENAI_API_KEY"), \\
            "Set OPENAI_API_KEY in .env before running this notebook."

        import pymongo
        from lib import MONGODB_URI, DB_NAME, collection_name

        client = pymongo.MongoClient(MONGODB_URI)
        coll   = client[DB_NAME][collection_name(DATASET)]
        """),
        md("""
        ## Step 1 — Sample documents from the corpus

        We pull `N_DOCS` distinct parent documents from the ingested
        chunks collection. Each will seed one (or more) draft
        queries.
        """),
        code("""
        import random
        random.seed(42)

        sample_docs = []
        seen = set()
        # Walk chunks; keep first chunk per parent doc until we have N_DOCS.
        for row in coll.find({}, {"doc_id": 1, "title": 1, "text": 1}).limit(N_DOCS * 4):
            did = row["doc_id"]
            if did in seen:
                continue
            seen.add(did)
            sample_docs.append({
                "doc_id": did,
                "title" : row.get("title", ""),
                "text"  : row["text"][:1500],
            })
            if len(sample_docs) >= N_DOCS:
                break

        print(f"Sampled {len(sample_docs)} documents.")
        for d in sample_docs[:3]:
            print(f"  doc {d['doc_id']:<12}  {d['title'][:60]!r}")
        """),
        md("""
        ## Step 2 — Draft queries with an LLM

        We ask the LLM to write **one realistic query** that a
        domain user might type to find each document. The prompt is
        the lever here — phrase it for your real users. Want
        keyword-style searches? Say so. Want long natural-language
        questions? Say so.
        """),
        code("""
        from openai import OpenAI
        oai = OpenAI()

        QUERY_PROMPT = (
            "You are helping build a retrieval evaluation set. Given a document, "
            "write ONE concise, realistic search query that a domain user might type "
            "to retrieve this exact document. The query should be 3-15 words and read "
            "naturally — not a paraphrase of the title. Do NOT include the document ID.\\n\\n"
            "Document title : {title}\\n"
            "Document text  : {text}\\n\\n"
            "Output ONLY the query, no quotes or commentary."
        )

        draft_queries = []
        for i, d in enumerate(sample_docs, 1):
            resp = oai.chat.completions.create(
                model=MODEL_LLM,
                messages=[{
                    "role": "user",
                    "content": QUERY_PROMPT.format(title=d["title"], text=d["text"]),
                }],
                temperature=0.7,
            )
            q = resp.choices[0].message.content.strip().strip('"').strip("'")
            draft_queries.append({"qid": f"my_q_{i:03d}", "query": q, "seed_doc_id": d["doc_id"]})
            print(f"  [{i:>2}/{len(sample_docs)}] {q}")
        """),
        md("""
        ## Step 3 — Pool candidates for each draft query

        For each draft query, retrieve the top-K with our hybrid
        black box. The union of these results is the **pool** — the
        set of `(query, doc)` pairs we'll judge. Pooling is the
        standard IR evaluation pattern (TREC and BEIR both use it):
        we don't grade every doc in the corpus, only the docs that
        *some* retriever surfaced.
        """),
        code("""
        import voyageai
        from lib import MONGODB_BASE_URL, VOYAGE_API_KEY, embed_queries
        from retrieve import hybrid

        voyage = voyageai.Client(api_key=VOYAGE_API_KEY, base_url=MONGODB_BASE_URL)
        q_texts = [q["query"] for q in draft_queries]
        q_vecs  = embed_queries(voyage, q_texts)

        pool: dict[str, list[dict]] = {}
        for q, q_vec in zip(draft_queries, q_vecs):
            ranked = hybrid(coll, q_vec, q["query"], top_k=TOP_K, alpha=0.8)
            pool[q["qid"]] = ranked

        total_pairs = sum(len(rows) for rows in pool.values())
        print(f"Pooled {total_pairs} (query, candidate) pairs across {len(pool)} queries.")
        """),
        md("""
        ## Step 4 — Grade with an LLM judge

        For each `(query, candidate doc)` pair, ask the LLM to grade
        relevance on the BEIR-standard 0–3 scale:

        - **3** — highly relevant: directly answers the query
        - **2** — relevant: clearly addresses the query topic
        - **1** — marginally relevant: tangentially related
        - **0** — not relevant

        The LLM is consistent and cheap; the prompt below uses the
        same wording the [TREC pooling tradition](https://trec.nist.gov/)
        uses to keep grades comparable.
        """),
        code("""
        import json
        import re

        JUDGE_PROMPT = (
            "You are an information retrieval relevance judge. Given a search "
            "query and a passage from a document, grade how relevant the passage "
            "is to the query on this scale:\\n"
            "  3 - Highly relevant: directly answers the query.\\n"
            "  2 - Relevant: clearly addresses the query topic.\\n"
            "  1 - Marginally relevant: tangentially related.\\n"
            "  0 - Not relevant.\\n\\n"
            "Query  : {query}\\n"
            "Passage: {passage}\\n\\n"
            "Output ONLY a single JSON object: "
            '{{"grade": <0|1|2|3>, "reason": "<short>"}}'
        )

        def judge(query: str, passage: str) -> tuple[int, str]:
            resp = oai.chat.completions.create(
                model=MODEL_LLM,
                messages=[{"role": "user",
                           "content": JUDGE_PROMPT.format(query=query, passage=passage[:1500])}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            return int(data.get("grade", 0)), data.get("reason", "")

        draft_qrels: dict[str, dict[str, int]] = {}
        reasons:     dict[tuple[str, str], str] = {}
        total = sum(len(v) for v in pool.values())
        done = 0
        for q in draft_queries:
            qid = q["qid"]
            draft_qrels[qid] = {}
            for row in pool[qid]:
                grade, reason = judge(q["query"], row["text"])
                draft_qrels[qid][row["doc_id"]] = grade
                reasons[(qid, row["doc_id"])] = reason
                done += 1
                if done % 10 == 0:
                    print(f"  graded {done}/{total} pairs …")
        print(f"  graded {done}/{total} pairs (done).")
        """),
        md("""
        ## Step 5 — Curate (human in the loop)

        This is the critical step. The LLM judge is a *draft*. Skim
        the high-grade and low-grade ends — that's where mistakes
        hide. Tweak any cell of `draft_qrels` directly. A small,
        carefully reviewed set beats a large, blindly-trusted one.
        """),
        code("""
        import pandas as pd

        rows = []
        for q in draft_queries:
            qid = q["qid"]
            for row in pool[qid]:
                did = row["doc_id"]
                rows.append({
                    "qid"      : qid,
                    "query"    : q["query"],
                    "doc_id"   : did,
                    "grade"    : draft_qrels[qid].get(did, 0),
                    "reason"   : reasons.get((qid, did), ""),
                    "title"    : row.get("title", "")[:60],
                })
        review_df = pd.DataFrame(rows).sort_values(["qid", "grade"], ascending=[True, False])
        review_df.head(20)
        """),
        code("""
        # Want to override a judgement? Edit it directly. Example:
        #   draft_qrels["my_q_001"]["1234"] = 2   # bump from 1 to 2
        # Then re-run the cell above to see the updated table.
        """),
        md("""
        ## Step 6 — Save and re-evaluate

        Save the curated qrels alongside the draft queries, then
        rerun the same evaluation we did in notebook 01 — this time
        against *our* judgements.
        """),
        code("""
        import pathlib, json

        EVAL_DIR = pathlib.Path("../eval_sets")
        EVAL_DIR.mkdir(exist_ok=True)
        (EVAL_DIR / f"{DATASET}_custom_queries.json").write_text(
            json.dumps({q["qid"]: q["query"] for q in draft_queries}, indent=2),
            encoding="utf-8",
        )
        (EVAL_DIR / f"{DATASET}_custom_qrels.json").write_text(
            json.dumps(draft_qrels, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote {EVAL_DIR.resolve()}/")
        """),
        code("""
        from retrieve import text_only, vector_only, hybrid
        from lib_metrics import compute_query_metrics, aggregate_metrics, format_summary

        STRATEGIES = {
            "lexical (BM25)" : lambda q_vec, q_text: text_only(coll, q_text, top_k=TOP_K),
            "vector"         : lambda q_vec, q_text: vector_only(coll, q_vec, top_k=TOP_K),
            "hybrid α=0.8"   : lambda q_vec, q_text: hybrid(coll, q_vec, q_text, top_k=TOP_K, alpha=0.8),
        }
        for name, run in STRATEGIES.items():
            per_query = []
            for q, q_vec in zip(draft_queries, q_vecs):
                ranked = run(q_vec, q["query"])
                ranked_ids = [r["doc_id"] for r in ranked]
                per_query.append(compute_query_metrics(ranked_ids, draft_qrels[q["qid"]]))
            agg = aggregate_metrics(per_query)
            print(f"{name:<20}  {format_summary(agg)}")
        """),
        md("""
        ## What's next

        You've just built a tiny custom evaluation set. To make it
        production-quality:

        - **Scale up.** 20 queries is a smoke test. Aim for 100+
          before you trust the numbers to drive decisions.
        - **Diversify queries.** Sample from real user logs if you
          have them. If not, prompt the LLM for multiple query
          *styles* per doc (keyword, natural-language, multi-hop).
        - **Tighten the judging prompt.** Add a couple of in-prompt
          examples from your domain. Calibrate the LLM on a handful
          of human-graded examples before scaling.
        - **Re-curate periodically.** Eval sets decay: corpora
          change, user needs change. Treat the eval set as code, not
          a finished artifact.

        And once you have a trustworthy eval set, the **advanced**
        lab in `phase4/` is where you'd go next — per-query alpha
        routing, query rewriters (HyDE, decompose), cross-encoder
        reranking. None of those are worth tuning without a solid
        eval set to measure against. Which is exactly the point.
        """),
    ]


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    write(NB_DIR, "00_setup_and_ingest.ipynb",   nb00())
    write(NB_DIR, "01_evaluate_blackbox.ipynb",  nb01())
    write(NB_DIR, "02_swap_blackbox.ipynb",      nb02())
    write(NB_DIR, "03_curate_eval_set.ipynb",    nb03())
    print("done")


if __name__ == "__main__":
    main()
