"""
Build the four lab notebooks at the repo root.

Written in the MongoDB quickstart style — short numbered steps, inline MQL
aggregation pipelines ($vectorSearch / $search / $rankFusion), and inline
metric formulas (Precision / Recall / NDCG / MRR shown as plain Python you
can follow line-by-line). The only things we don't inline are the unglamorous
helpers: the recursive text splitter (lib.split_text) and the contextualized
embedding HTTP call (lib.embed_contextualized).

Regenerate:
    python3 src/scripts/build_handcoded.py
"""

from __future__ import annotations

import pathlib

from _nb_helpers import md, code, write


# Notebooks live at the repo root (two levels up from src/scripts/).
NB_DIR = pathlib.Path(__file__).resolve().parent.parent.parent


SETUP_CELL = """
import os, sys
# Notebooks live at the repo root; library modules live in src/.
_REPO_ROOT = os.path.abspath(os.getcwd())
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
"""


# ─────────────────────────────────────────────────────────────────────────────
# Notebook 00 — Setup, ingest, create indexes
# ─────────────────────────────────────────────────────────────────────────────

def nb00() -> list[dict]:
    return [
        md("""
        # Lab 0 — Set up MongoDB Atlas + ingest a retrieval dataset

        This is a one-time setup notebook. You'll:

        1. Confirm your Atlas + Voyage credentials.
        2. Pick a public **BEIR** retrieval benchmark dataset.
        3. Split each document into chunks and embed the chunks with
           **`voyage-context-3`** — Voyage's contextualized chunk-embedding
           model — via the MongoDB-hosted endpoint.
        4. Insert the chunks into a MongoDB Atlas collection.
        5. Create a **vector search index** and a **lexical (BM25) search
           index** so that later notebooks can compare retrieval
           strategies on identical data.

        The next three notebooks (Labs 1 / 2 / 3) all read from the
        collection this notebook builds. You only need to run this lab
        once per dataset.
        """),
        md("""
        ## Step 1 — Prerequisites

        Create a `.env` file at the repo root with these values:

        ```
        VOYAGE_API_KEY=al-...        # Atlas → AI Models → API Keys
        MONGODB_URI=mongodb+srv://...
        OPENAI_API_KEY=sk-...        # only needed in Lab 3
        ```

        Then install the Python packages:

        ```
        pip3 install --break-system-packages \\
            pymongo voyageai beir python-dotenv requests \\
            openai pandas matplotlib nbformat
        ```
        """),
        code(SETUP_CELL),
        code("""
        # Load credentials and sanity-check they exist.
        from dotenv import load_dotenv
        load_dotenv(os.path.join(_REPO_ROOT, ".env"))

        VOYAGE_API_KEY   = os.environ["VOYAGE_API_KEY"]
        MONGODB_URI      = os.environ["MONGODB_URI"]
        MONGODB_BASE_URL = "https://ai.mongodb.com/v1"

        print("VOYAGE_API_KEY loaded:", VOYAGE_API_KEY[:8] + "…")
        print("MONGODB_URI    loaded:", MONGODB_URI.split("@")[-1][:40] + "…")
        """),
        md("""
        ## Step 2 — Connect to Atlas

        Open a `pymongo` client and pick a database + collection name.
        We'll namespace each BEIR dataset to its own collection so you
        can ingest as many as you want without overwriting.
        """),
        code("""
        import pymongo

        DB_NAME    = "voyage_context_demo"
        DATASET    = "scifact"                  # pick from the table below
        COLL_NAME  = f"chunks_{DATASET.replace('-', '_')}"

        client = pymongo.MongoClient(MONGODB_URI)
        db     = client[DB_NAME]
        coll   = db[COLL_NAME]

        print(f"Database  : {DB_NAME}")
        print(f"Collection: {COLL_NAME}")
        """),
        md("""
        ## Step 3 — Pick a BEIR dataset

        [BEIR](https://github.com/beir-cellar/beir) ships eight retrieval
        benchmark datasets. Each one is a `(corpus, queries, qrels)`
        triple with human-curated relevance judgements. Pick one to
        ingest — `scifact` is small (5.2k docs) and clean, so start
        there.

        | name | description |
        |---|---|
        | `scifact`    | scientific claim verification (300 queries / 5.2k abstracts) — *recommended for first run* |
        | `nfcorpus`   | medical literature retrieval (323 queries / 3.6k docs) |
        | `fiqa`       | financial Q&A — opinionated long answers (648 queries / 57k docs) |
        | `arguana`    | counter-argument retrieval (1.4k queries / 8.7k arguments) |
        | `scidocs`    | scientific paper retrieval (1k queries / 25k docs) |
        | `trec-covid` | COVID-19 research retrieval (50 queries / 171k docs) |
        | `touche2020` | controversial-topic arguments (49 queries / 382k docs) |
        | `quora`      | duplicate-question retrieval (10k queries / 523k docs) |
        """),
        code("""
        # Load the BEIR dataset. This downloads the raw files to
        # /tmp/beir_datasets/<DATASET>/ on first run.
        from lib import load_beir_dataset
        corpus, queries, qrels, info = load_beir_dataset(DATASET)

        print(f"{DATASET}: {info['description']}")
        print(f"  corpus  : {len(corpus):,} documents")
        print(f"  queries : {len(queries):,} (test split)")
        print(f"  qrels   : {len(qrels):,} queries have relevance judgements")
        """),
        code("""
        # Look at one document, one query, and the qrels for that query.
        sample_qid = next(iter(qrels))
        sample_did = next(iter(qrels[sample_qid]))

        print("Sample query   :", repr(queries[sample_qid]))
        print()
        print("Sample document:")
        print("  doc_id :", sample_did)
        print("  title  :", repr(corpus[sample_did]['title']))
        print("  text   :", repr(corpus[sample_did]['text'][:200] + '…'))
        print()
        print(f"qrels[{sample_qid!r}] = {qrels[sample_qid]}")
        print("  (each value is the relevance grade: 0=not relevant, 1+=relevant)")
        """),
        md("""
        ## Step 4 — Sample documents to ingest

        BEIR corpora can be large. For a fast first pass, pick a small
        sample. We'll bias the sample to include every doc that's
        marked relevant by at least one query, so the test queries
        actually have something to retrieve.
        """),
        code("""
        import random
        random.seed(42)

        CORPUS_SAMPLE = 500

        # Collect all doc_ids that are marked relevant by any qrel,
        # then top up with a random sample from the rest until we have
        # CORPUS_SAMPLE documents.
        corpus_keys = set(corpus.keys())
        must_include = list({
            did
            for q_qrels in qrels.values()
            for did, score in q_qrels.items()
            if score > 0 and did in corpus_keys
        })[:CORPUS_SAMPLE]

        remaining   = [d for d in corpus.keys() if d not in set(must_include)]
        random_top  = random.sample(remaining, max(0, CORPUS_SAMPLE - len(must_include)))
        sample_ids  = must_include + random_top
        print(f"Sampling {len(sample_ids)} docs "
              f"({len(must_include)} guaranteed-relevant + {len(random_top)} random)")
        """),
        md("""
        ## Step 5 — Split each document into chunks

        `voyage-context-3` is a **chunk** embedding model. It expects
        each document to be split into smaller passages first, then
        embeds each passage *in the context of the whole document* so
        the resulting vectors carry the broader meaning.

        We use a recursive character splitter — it tries paragraph
        boundaries first, then lines, then sentences, then words.
        It's about 30 lines of code, so we leave it in `lib.split_text`
        and just call it here.
        """),
        code("""
        from lib import split_text

        CHUNK_SIZE    = 1000      # chars (~250 tokens)
        CHUNK_OVERLAP = 150

        records = []
        for did in sample_ids:
            doc  = corpus[did]
            full = f"{doc['title']}\\n\\n{doc['text']}" if doc['title'] else doc['text']
            for i, chunk in enumerate(split_text(full, CHUNK_SIZE, CHUNK_OVERLAP)):
                records.append({
                    "doc_id"   : did,
                    "chunk_idx": i,
                    "title"    : doc['title'],
                    "text"     : chunk,
                })

        print(f"{len(sample_ids)} docs → {len(records)} chunks")
        print(f"first chunk record:")
        print(f"  doc_id    : {records[0]['doc_id']}")
        print(f"  chunk_idx : {records[0]['chunk_idx']}")
        print(f"  text      : {records[0]['text'][:120]!r}…")
        """),
        md("""
        ## Step 6 — Embed each chunk with `voyage-context-3`

        The contextualized-embeddings endpoint is a separate API from
        the standard `/v1/embeddings`: the request shape is

        ```json
        {
          "model" : "voyage-context-3",
          "inputs": [
            ["full_doc_text", "chunk_1", "chunk_2", ...],
            ["full_doc_2",    "chunk_1", "chunk_2", ...]
          ]
        }
        ```

        — each inner list contains all chunks of one document plus the
        full doc text as an "anchor" so the chunks share context.
        That logic (batching, retries, splitting docs that exceed the
        token cap) is ~100 lines, so it lives in `lib.embed_contextualized`.
        """),
        code("""
        from collections import defaultdict
        from lib import embed_contextualized

        # Group chunks back by parent doc.
        chunks_per_doc = defaultdict(list)
        for r in records:
            chunks_per_doc[r['doc_id']].append(r)

        doc_chunk_pairs = []
        doc_order = list(chunks_per_doc.keys())
        for did in doc_order:
            full = (f"{corpus[did]['title']}\\n\\n{corpus[did]['text']}"
                    if corpus[did]['title'] else corpus[did]['text'])
            doc_chunk_pairs.append((full, [r['text'] for r in chunks_per_doc[did]]))

        # POST /v1/contextualizedembeddings  →  one vector per chunk.
        vectors = embed_contextualized(doc_chunk_pairs)
        print(f"{len(vectors)} embeddings returned, dims={len(vectors[0])}")

        # Attach vectors back to records in the same order they were built.
        flat = 0
        for did in doc_order:
            for r in chunks_per_doc[did]:
                r['embedding'] = vectors[flat]
                flat += 1
        """),
        md("""
        ## Step 7 — Insert chunks into MongoDB

        Drop the collection (in case we re-ran ingest) and insert in
        batches. Each chunk becomes one document; the embedding is
        stored alongside the chunk text.
        """),
        code("""
        coll.drop()

        BATCH = 500
        for i in range(0, len(records), BATCH):
            coll.insert_many(records[i : i + BATCH])

        print(f"Inserted {coll.estimated_document_count()} chunks into {DB_NAME}.{COLL_NAME}")
        print("schema of one stored chunk:")
        print(list(coll.find_one().keys()))
        """),
        md("""
        ## Step 8 — Create the vector search index

        Atlas needs a **search index** to support `$vectorSearch`. The
        index definition tells Atlas which field is the vector, how
        many dimensions it has, and which similarity metric to use.

        We use cosine similarity here because that's what
        `voyage-context-3` is trained on.
        """),
        code("""
        from pymongo.operations import SearchIndexModel

        VECTOR_INDEX_NAME = "voyage_vector_index"
        DIMS = len(records[0]['embedding'])

        vector_index = SearchIndexModel(
            name=VECTOR_INDEX_NAME,
            type="vectorSearch",
            definition={
                "fields": [{
                    "type"         : "vector",
                    "path"         : "embedding",
                    "numDimensions": DIMS,
                    "similarity"   : "cosine",
                }]
            },
        )

        existing = {idx['name'] for idx in coll.list_search_indexes()}
        if VECTOR_INDEX_NAME not in existing:
            coll.create_search_index(vector_index)
            print(f"Created vector index '{VECTOR_INDEX_NAME}' ({DIMS} dims, cosine)")
        else:
            print(f"Index '{VECTOR_INDEX_NAME}' already exists.")
        """),
        md("""
        ## Step 9 — Create the lexical (BM25) search index

        Atlas Search's standard text index supports BM25-style scoring.
        We index the `text` and `title` fields with the standard Lucene
        analyzer. This is the index `$search` will use in Lab 1.
        """),
        code("""
        TEXT_INDEX_NAME = "voyage_text_index"

        text_index = SearchIndexModel(
            name=TEXT_INDEX_NAME,
            type="search",
            definition={
                "mappings": {
                    "dynamic": False,
                    "fields": {
                        "text" : {"type": "string", "analyzer": "lucene.standard"},
                        "title": {"type": "string", "analyzer": "lucene.standard"},
                    },
                }
            },
        )

        if TEXT_INDEX_NAME not in {idx['name'] for idx in coll.list_search_indexes()}:
            coll.create_search_index(text_index)
            print(f"Created text index '{TEXT_INDEX_NAME}'")
        else:
            print(f"Index '{TEXT_INDEX_NAME}' already exists.")
        """),
        md("""
        ## Step 10 — Wait for indexes to become queryable

        Atlas builds search indexes asynchronously. Vector indexes
        usually take 30–60s; text indexes can be slower. Poll until
        both are `queryable: True`.
        """),
        code("""
        import time

        WAIT_SECONDS = 300        # bail out after 5 minutes
        targets = {VECTOR_INDEX_NAME, TEXT_INDEX_NAME}

        for _ in range(WAIT_SECONDS // 5):
            statuses = {idx['name']: idx.get('queryable', False)
                        for idx in coll.list_search_indexes()
                        if idx['name'] in targets}
            print("  ", statuses)
            if all(statuses.values()):
                print("Both indexes are queryable.")
                break
            time.sleep(5)
        else:
            print("Timed out waiting. First query may be slow.")
        """),
        md("""
        ## Done

        You now have:

        - `voyage_context_demo.chunks_scifact` (or whichever dataset
          you chose) with chunk documents that each include a
          contextualized embedding.
        - A `voyage_vector_index` for `$vectorSearch`.
        - A `voyage_text_index` for `$search` (BM25).

        **Next:** open **`01_evaluate_blackbox.ipynb`** to see how IR
        evaluation works — we'll treat lexical (BM25) retrieval as a
        "black box" and measure its quality with the same metrics you'd
        use to compare any retrieval system.
        """),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Notebook 01 — Lessons 1+2: measure a BM25 black box
# ─────────────────────────────────────────────────────────────────────────────

def nb01() -> list[dict]:
    return [
        md("""
        # Lab 1 — Evaluate a black-box retriever

        **Companion to course Lessons 1 and 2.**

        > *Imagine you're building a recruiting application. A hiring
        > manager needs a shortlist by the end of the day. You run your
        > retrieval system and get back a ranked list of candidates.
        > But how do you know if it's any good?*

        That's the question this lab answers, end-to-end, on real
        data. The "retrieval system" is **lexical (BM25) search** —
        the keyword-matching family of algorithms that has powered
        full-text search for decades. We treat it as a black box: text
        in, ranked list of documents out. Our job is to **score** that
        ranked list against ground-truth relevance judgements.

        You'll compute **Precision@k**, **Recall@k**, **NDCG@k**, and
        **MRR** by hand on one query (so you can see exactly how the
        math works), then aggregate across many queries.
        """),
        md("""
        ## Step 1 — Setup

        Match `DATASET` to whatever you ingested in Lab 0.
        """),
        code(SETUP_CELL),
        code("""
        from dotenv import load_dotenv
        load_dotenv(os.path.join(_REPO_ROOT, ".env"))

        import pymongo
        from lib import load_beir_dataset

        DB_NAME           = "voyage_context_demo"
        DATASET           = "scifact"
        COLL_NAME         = f"chunks_{DATASET.replace('-', '_')}"
        TEXT_INDEX_NAME   = "voyage_text_index"

        client = pymongo.MongoClient(os.environ["MONGODB_URI"])
        coll   = client[DB_NAME][COLL_NAME]
        corpus, queries, qrels, info = load_beir_dataset(DATASET)

        print(f"Connected to {DB_NAME}.{COLL_NAME} — {coll.estimated_document_count():,} chunks")
        print(f"Loaded {DATASET}: {len(queries)} test queries, qrels for {len(qrels)} of them")
        """),
        md("""
        ## Step 2 — What is an evaluation dataset?

        Every BEIR dataset is three Python dicts:

        - **`corpus`** — the haystack: `{doc_id: {"title": ..., "text": ...}}`.
        - **`queries`** — what users ask: `{query_id: query_text}`.
        - **`qrels`** ("query relevance" judgements) — the answer key:
          `{query_id: {doc_id: relevance_score}}`. A non-zero score
          means a human judged that document relevant to that query.

        The qrels are sometimes called the **judgement list**, **golden
        dataset**, or **ground truth**. Without them, evaluation isn't
        possible — there's nothing to compare retrieval results against.

        Pull one query and look at it:
        """),
        code("""
        # We want a worked-example query where BM25 actually retrieves at
        # least one relevant doc in the top 10 — otherwise the metric demo
        # below would be all zeros. (Some queries share no keywords with
        # their relevant docs; BM25 can't bridge that gap. Those are great
        # examples of where vector search wins, which we'll see in Lab 2.)
        def has_top_10_hit(qid):
            q_qrels = qrels.get(qid, {})
            rel_set = {did for did, s in q_qrels.items() if s > 0}
            if not rel_set:
                return False
            pipeline = [
                {"$search": {"index": TEXT_INDEX_NAME,
                              "text": {"path": "text", "query": queries[qid]}}},
                {"$limit": 40},
            ]
            seen_docs = set()
            for row in coll.aggregate(pipeline):
                seen_docs.add(row['doc_id'])
                if len(seen_docs) >= 10:
                    break
            return bool(seen_docs & rel_set)

        sample_qid    = next(qid for qid in queries if has_top_10_hit(qid))
        sample_query  = queries[sample_qid]
        sample_qrels  = qrels.get(sample_qid, {})
        relevant_docs = {did: s for did, s in sample_qrels.items() if s > 0}

        print(f"Query ID     : {sample_qid}")
        print(f"Query text   : {sample_query!r}")
        print(f"Relevant docs: {len(relevant_docs)}")
        for did, score in list(relevant_docs.items())[:3]:
            title = corpus.get(did, {}).get('title', '<missing>')
            print(f"  {did:<12} grade={score}  title={title[:60]!r}")
        """),
        md("""
        ### Graded vs binary relevance

        BEIR qrels use **graded relevance**:

        - `0` — not relevant
        - `1` — relevant
        - `2` — highly relevant (in datasets that distinguish)

        Binary relevance treats anything `> 0` as relevant. Most
        metrics work with either form; NDCG benefits the most from
        the extra resolution. We'll see why below.
        """),
        md("""
        ## Step 3 — Run the BM25 black box

        Atlas Search's `$search` stage runs a BM25-family query
        against the text index we built in Lab 0. The aggregation
        pipeline is the entire interface — no library functions —
        and looks like this:
        """),
        code("""
        TOP_K = 10

        pipeline = [
            # Stage 1: lexical (BM25) search against our text index.
            {"$search": {
                "index": TEXT_INDEX_NAME,
                "text" : {"path": "text", "query": sample_query},
            }},
            # Stage 2: pull the BM25 score from the index metadata into a regular field.
            {"$addFields": {"score": {"$meta": "searchScore"}}},
            # Stage 3: keep the top N candidates. We over-fetch and dedupe below.
            {"$limit": TOP_K * 4},
            {"$sort":  {"score": -1}},
        ]

        results = list(coll.aggregate(pipeline))
        print(f"$search returned {len(results)} chunks (before dedup).")
        """),
        md("""
        Our collection stores **chunks**, not parent documents — one
        BEIR doc may have produced several chunks. So we dedupe to the
        best chunk per parent doc, then keep the top-K.
        """),
        code("""
        # Keep the highest-scoring chunk per parent doc.
        seen = {}
        for row in results:
            did = row['doc_id']
            if did not in seen or row['score'] > seen[did]['score']:
                seen[did] = row

        ranked = sorted(seen.values(), key=lambda r: r['score'], reverse=True)[:TOP_K]

        print(f"Top-{TOP_K} for query {sample_qid!r}:")
        for rank, row in enumerate(ranked, 1):
            grade = sample_qrels.get(row['doc_id'], 0)
            tag = "★" if grade > 0 else " "
            print(f"  {rank:>2}. {tag} doc {row['doc_id']:<12} "
                  f"score={row['score']:>6.3f}  grade={grade}  "
                  f"{row['title'][:55]!r}")
        """),
        md("""
        Rows marked ★ are ones the qrels confirm are relevant.
        Unmarked rows are either confirmed irrelevant (grade 0) or
        **not judged at all** — most public IR datasets only have
        judgements for a small subset of the corpus. That's the
        pooling assumption: docs that no retriever surfaced are
        treated as irrelevant.
        """),
        md("""
        ## Step 4 — Precision@k

        > *Of the documents the system returned, what fraction are relevant?*

        Precision answers the **false positive** question: "how much
        noise is in my shortlist?" If 4 of 5 returned candidates are
        relevant, Precision@5 = 0.8.

        $$\\text{Precision@k} = \\frac{\\#\\text{ relevant in top-}k}{k}$$
        """),
        code("""
        ranked_ids   = [r['doc_id'] for r in ranked]
        relevant_ids = {did for did, score in sample_qrels.items() if score > 0}

        k = 5
        top_k = ranked_ids[:k]
        hits  = sum(1 for did in top_k if did in relevant_ids)
        precision_at_5 = hits / k

        print(f"top-5 doc_ids : {top_k}")
        print(f"# relevant    : {hits}")
        print(f"Precision@5   = {hits}/{k} = {precision_at_5:.3f}")
        """),
        md("""
        ## Step 5 — Recall@k

        > *Of all the relevant documents that exist, what fraction did we return?*

        Recall answers the **false negative** question: "how many
        relevant documents did I miss?" If 10 docs in the corpus are
        relevant and 7 appear in the top-k, Recall@k = 0.7.

        $$\\text{Recall@k} = \\frac{\\#\\text{ relevant in top-}k}{\\#\\text{ relevant in corpus}}$$

        > **Note.** "Recall" also shows up in vector-index benchmarks
        > where it means the fraction of true nearest neighbours an
        > approximate index returned. Same word, different reference
        > point — that's an index-quality metric; this is a
        > retrieval-quality metric.
        """),
        code("""
        k = 10
        top_k         = ranked_ids[:k]
        hits_in_top_k = len(set(top_k) & relevant_ids)
        recall_at_10  = hits_in_top_k / len(relevant_ids)

        print(f"# relevant in corpus : {len(relevant_ids)}")
        print(f"# relevant in top-{k}  : {hits_in_top_k}")
        print(f"Recall@{k}            = {hits_in_top_k}/{len(relevant_ids)} = {recall_at_10:.3f}")
        """),
        md("""
        ### Precision/Recall trade-off

        These two pull in opposite directions. Want higher precision?
        Return fewer, higher-confidence results — recall drops. Want
        higher recall? Cast a wider net — precision drops. Which
        matters more is an application-level decision: a legal-discovery
        tool can't miss relevant precedent (favour recall); a
        user-facing search box can't bury good results under junk
        (favour precision).
        """),
        md("""
        ## Step 6 — NDCG@k (the most important one)

        Precision and Recall treat the top-k as a *set* — they ignore
        order. But users look at ranked lists top-down. A relevant
        document at rank 1 is much more valuable than the same
        document at rank 9.

        **NDCG** captures that. It has three pieces:

        1. **Gain** — the relevance grade of each retrieved document
           (graded relevance pays off here).
        2. **Discount** — divide the gain by `log₂(rank + 1)`, so
           rank 1 weighs 1.0, rank 5 weighs ~0.39, rank 10 weighs ~0.29.
        3. **Normalize** — divide by the score of an *ideal* ranking
           (relevant docs sorted from highest grade to lowest), so the
           final score sits in `[0, 1]`. A perfect ranking scores 1.0.

        $$\\text{DCG@k} = \\sum_{i=1}^{k} \\frac{\\text{grade}(d_i)}{\\log_2(i+1)}
        \\qquad \\text{NDCG@k} = \\frac{\\text{DCG@k}}{\\text{IDCG@k}}$$
        """),
        code("""
        import math

        k = 5
        print(f"DCG@{k} calculation:")
        dcg = 0.0
        for rank, did in enumerate(ranked_ids[:k], 1):
            grade    = sample_qrels.get(did, 0)
            discount = math.log2(rank + 1)
            term     = grade / discount
            dcg     += term
            print(f"  rank {rank}: grade={grade}  log2({rank+1})={discount:.3f}  "
                  f"term={grade}/{discount:.3f} = {term:.3f}")
        print(f"DCG@{k}  = {dcg:.3f}")

        # The ideal ranking: every relevant doc sorted by descending grade.
        ideal_grades = sorted((s for s in sample_qrels.values() if s > 0), reverse=True)[:k]
        idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal_grades, 1))
        print(f"IDCG@{k} = {idcg:.3f}   (best possible ordering of available grades)")
        print(f"NDCG@{k} = DCG/IDCG = {dcg/idcg if idcg else 0:.3f}")
        """),
        md("""
        NDCG@k is *the* primary metric for embedding-model comparison.
        You'll see it on every public leaderboard (MTEB, RTEB, BEIR
        itself). It rewards putting the most relevant documents at the
        top while still tolerating burying weaker (but still-relevant)
        ones lower in the list.
        """),
        md("""
        ## Step 7 — MRR (Mean Reciprocal Rank)

        > *On average, how high up does the first relevant document appear?*

        Scan the ranked list from position 1 downward until you hit a
        relevant doc. The reciprocal rank is `1 / first_rel_rank`:
        1.0 at position 1, 0.5 at position 2, 0.33 at position 3.
        Average across all queries to get MRR.

        MRR is most informative when there's one clearly best fit per
        query — a lookup, a known-item search, or when evaluating a
        reranker whose only job is to bubble the best answer to the
        top.
        """),
        code("""
        first_rel_rank = next(
            (i for i, did in enumerate(ranked_ids, 1) if did in relevant_ids),
            None,
        )
        if first_rel_rank:
            rr = 1.0 / first_rel_rank
            print(f"First relevant doc at rank {first_rel_rank}  →  RR = 1/{first_rel_rank} = {rr:.3f}")
        else:
            rr = 0.0
            print("No relevant doc retrieved at all  →  RR = 0")
        """),
        md("""
        ## Step 8 — Evaluate over many queries

        A retrieval system isn't judged on one query — it's judged on
        a distribution. We'll wrap the per-query math into a single
        function, then loop over the first `N` test queries.

        This is the function you'll re-use in Labs 2 and 3.
        """),
        code("""
        import math

        def query_metrics(ranked: list[str], qrels: dict[str, int], ks=(5, 10)) -> dict:
            \"\"\"Compute P@k, R@k, NDCG@k for each k, plus MRR and AP.\"\"\"
            rel_set = {did for did, s in qrels.items() if s > 0}
            out = {}
            for k in ks:
                top_k = ranked[:k]
                out[f"P@{k}"] = sum(1 for d in top_k if d in rel_set) / k
                out[f"R@{k}"] = len(set(top_k) & rel_set) / max(1, len(rel_set))
                # NDCG@k
                dcg = sum(qrels.get(d, 0) / math.log2(i + 1)
                          for i, d in enumerate(top_k, 1))
                ideal = sorted((s for s in qrels.values() if s > 0), reverse=True)[:k]
                idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal, 1))
                out[f"NDCG@{k}"] = dcg / idcg if idcg else 0.0
            # MRR: 1 / rank of first relevant
            out["MRR"] = next(
                (1.0 / i for i, d in enumerate(ranked, 1) if d in rel_set), 0.0,
            )
            # AP: mean of precision-at-each-hit
            cum = hits = 0
            for i, d in enumerate(ranked, 1):
                if d in rel_set:
                    hits += 1
                    cum  += hits / i
            out["AP"] = cum / len(rel_set) if rel_set else 0.0
            return out

        # Sanity check on the query we already analysed.
        print(query_metrics(ranked_ids, sample_qrels))
        """),
        code("""
        def bm25_search(query_text: str, top_k: int = 10) -> list[str]:
            \"\"\"Run a $search pipeline and return deduplicated doc_ids.\"\"\"
            pipeline = [
                {"$search": {
                    "index": TEXT_INDEX_NAME,
                    "text" : {"path": "text", "query": query_text},
                }},
                {"$addFields": {"score": {"$meta": "searchScore"}}},
                {"$limit": top_k * 4},
                {"$sort":  {"score": -1}},
            ]
            seen = {}
            for row in coll.aggregate(pipeline):
                did = row['doc_id']
                if did not in seen or row['score'] > seen[did]['score']:
                    seen[did] = row
            return [r['doc_id'] for r in sorted(seen.values(),
                    key=lambda r: r['score'], reverse=True)[:top_k]]


        N_QUERIES = 30
        query_ids = list(queries.keys())[:N_QUERIES]
        per_query = []
        for qid in query_ids:
            ranked = bm25_search(queries[qid], top_k=10)
            per_query.append(query_metrics(ranked, qrels.get(qid, {})))

        # Aggregate: mean each metric; AP-mean becomes MAP.
        keys = per_query[0].keys()
        agg = {k: sum(q[k] for q in per_query) / len(per_query) for k in keys}
        agg["MAP"] = agg.pop("AP")
        print(f"BM25 lexical retrieval — {DATASET}, {N_QUERIES} queries")
        for k, v in agg.items():
            print(f"  {k:<8} {v:.3f}")
        """),
        md("""
        ## Reading the numbers

        Each metric tells you something different:

        - **P@5 / P@10** — top-of-list cleanliness (low = lots of false positives).
        - **R@5 / R@10** — coverage of the relevant set (low = lots of false negatives).
        - **NDCG@5 / NDCG@10** — ranking quality. The number you'd
          quote when comparing two retrievers on a benchmark.
        - **MRR** — how prominently the first relevant result sits.
        - **MAP** — Mean Average Precision; recall-and-rank summary
          across all positions where relevant docs land.

        Write these numbers down — you'll compare them to **vector**
        and **hybrid** retrievers in Lab 2.

        **Next:** open **`02_swap_blackbox.ipynb`**.
        """),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Notebook 02 — Lesson 4: lexical vs vector vs hybrid
# ─────────────────────────────────────────────────────────────────────────────

def nb02() -> list[dict]:
    return [
        md("""
        # Lab 2 — Swap the black box: vector and hybrid search

        **Companion to course Lesson 4.**

        In Lab 1 you measured a lexical (BM25) retriever. Now we keep
        everything else fixed — same dataset, same queries, same
        metrics — and swap out the retrieval strategy. This is the
        experimental discipline that lets you say something meaningful
        when comparing systems: change one variable, hold everything
        else constant.

        ## The three strategies

        | name | aggregation stage | strength |
        |---|---|---|
        | **lexical (BM25)** | `$search` | exact terms, names, IDs |
        | **vector** | `$vectorSearch` | semantic match, paraphrase |
        | **hybrid** | `$rankFusion` | best of both, weighted fusion |

        You'll see all three as MQL pipelines — no library indirection.
        """),
        code(SETUP_CELL),
        code("""
        from dotenv import load_dotenv
        load_dotenv(os.path.join(_REPO_ROOT, ".env"))

        import math, pymongo, voyageai
        from lib import load_beir_dataset, embed_queries

        DATASET           = "scifact"
        DB_NAME           = "voyage_context_demo"
        COLL_NAME         = f"chunks_{DATASET.replace('-', '_')}"
        VECTOR_INDEX_NAME = "voyage_vector_index"
        TEXT_INDEX_NAME   = "voyage_text_index"
        MONGODB_BASE_URL  = "https://ai.mongodb.com/v1"
        TOP_K             = 10
        N_QUERIES         = 30

        client = pymongo.MongoClient(os.environ["MONGODB_URI"])
        coll   = client[DB_NAME][COLL_NAME]
        corpus, queries, qrels, info = load_beir_dataset(DATASET)

        voyage = voyageai.Client(
            api_key=os.environ["VOYAGE_API_KEY"],
            base_url=MONGODB_BASE_URL,
        )
        print(f"Connected to {DB_NAME}.{COLL_NAME}, {len(queries)} test queries available.")
        """),
        md("""
        ## Step 1 — Embed the queries with `voyage-3-large`

        For retrieval, we embed the user's query with a query-side
        model and compare it to the chunk vectors we stored in Lab 0.
        `voyage-context-3` is a *document* embedder — for queries we
        use the matching standard model, `voyage-3-large`.

        The embedding helper just wraps `voyage.embed(...)` — we use
        it directly here.
        """),
        code("""
        query_ids = list(queries.keys())[:N_QUERIES]
        q_texts   = [queries[qid] for qid in query_ids]
        q_vecs    = embed_queries(voyage, q_texts)
        print(f"Embedded {len(q_vecs)} queries to dimension {len(q_vecs[0])}.")
        """),
        md("""
        ## Step 2 — Vector search via `$vectorSearch`

        `$vectorSearch` does approximate nearest-neighbour lookup
        against the vector index we built in Lab 0. The pipeline
        below is the entire interface — embed the query, drop the
        vector in as `queryVector`, and ask Atlas for the top-K.
        """),
        code("""
        def dedup_by_doc(rows, top_k):
            \"\"\"Collapse chunk-level results to one row per parent doc.\"\"\"
            seen = {}
            for row in rows:
                did = row['doc_id']
                if did not in seen or row['score'] > seen[did]['score']:
                    seen[did] = row
            return sorted(seen.values(), key=lambda r: r['score'], reverse=True)[:top_k]


        def vector_search(q_vec: list[float], top_k: int = TOP_K) -> list[dict]:
            pipeline = [
                # Stage 1: ANN over our vector index.
                {"$vectorSearch": {
                    "index"        : VECTOR_INDEX_NAME,
                    "path"         : "embedding",
                    "queryVector"  : q_vec,
                    "numCandidates": top_k * 20,   # over-fetch for re-ranking
                    "limit"        : top_k * 4,
                }},
                # Stage 2: surface the cosine-similarity score.
                {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
                {"$sort": {"score": -1}},
            ]
            return dedup_by_doc(coll.aggregate(pipeline), top_k)


        # Smoke-test on the first query
        sample = vector_search(q_vecs[0], top_k=5)
        print(f"Vector top-5 for {q_texts[0]!r}:")
        for rank, row in enumerate(sample, 1):
            print(f"  {rank}. doc {row['doc_id']:<12} score={row['score']:.3f}  "
                  f"{row['title'][:55]!r}")
        """),
        md("""
        ## Step 3 — Hybrid via `$rankFusion`

        Atlas 8.0+ ships `$rankFusion`, a native Reciprocal Rank
        Fusion operator that combines multiple ranking pipelines
        server-side. You give it the two sub-pipelines (vector and
        text) and per-pipeline weights; it computes

        $$\\text{RRFscore}(d) = \\sum_i \\frac{w_i}{60 + \\text{rank}_i(d)}$$

        across both rankings. A single weight `α ∈ [0, 1]` controls
        the blend: we pass `weights={vector: α, text: 1-α}`.
        """),
        code("""
        def hybrid_search(q_vec: list[float], q_text: str,
                          alpha: float = 0.8, top_k: int = TOP_K) -> list[dict]:
            \"\"\"Weighted RRF of vector + lexical via $rankFusion (Atlas 8.0+).\"\"\"
            # $rankFusion requires non-zero weights; nudge edges by a tiny epsilon.
            EPS    = 1e-3
            w_vec  = max(alpha,       EPS)
            w_text = max(1.0 - alpha, EPS)
            n      = 100   # per-pipeline candidate depth

            pipeline = [
                {"$rankFusion": {
                    "input": {
                        "pipelines": {
                            "vector": [
                                {"$vectorSearch": {
                                    "index"        : VECTOR_INDEX_NAME,
                                    "path"         : "embedding",
                                    "queryVector"  : q_vec,
                                    "numCandidates": n * 4,
                                    "limit"        : n,
                                }},
                            ],
                            "text": [
                                {"$search": {
                                    "index": TEXT_INDEX_NAME,
                                    "text" : {"path": "text", "query": q_text},
                                }},
                                {"$limit": n},
                            ],
                        },
                    },
                    "combination": {"weights": {"vector": w_vec, "text": w_text}},
                }},
                {"$addFields": {"score": {"$meta": "score"}}},
                {"$limit": top_k * 4},
            ]
            return dedup_by_doc(coll.aggregate(pipeline), top_k)


        sample = hybrid_search(q_vecs[0], q_texts[0], alpha=0.8, top_k=5)
        print(f"Hybrid (α=0.8) top-5 for {q_texts[0]!r}:")
        for rank, row in enumerate(sample, 1):
            print(f"  {rank}. doc {row['doc_id']:<12} score={row['score']:.3f}  "
                  f"{row['title'][:55]!r}")
        """),
        md("""
        ## Step 4 — Lexical, for the comparison

        Same as Lab 1, repeated here so the three search functions
        sit side-by-side.
        """),
        code("""
        def bm25_search(q_text: str, top_k: int = TOP_K) -> list[dict]:
            pipeline = [
                {"$search": {
                    "index": TEXT_INDEX_NAME,
                    "text" : {"path": "text", "query": q_text},
                }},
                {"$addFields": {"score": {"$meta": "searchScore"}}},
                {"$limit": top_k * 4},
                {"$sort":  {"score": -1}},
            ]
            return dedup_by_doc(coll.aggregate(pipeline), top_k)
        """),
        md("""
        ## Step 5 — Evaluate all three on the same queries

        Reuse the `query_metrics` we built in Lab 1.
        """),
        code("""
        def query_metrics(ranked, qrels, ks=(5, 10)):
            rel_set = {did for did, s in qrels.items() if s > 0}
            out = {}
            for k in ks:
                top_k = ranked[:k]
                out[f"P@{k}"] = sum(1 for d in top_k if d in rel_set) / k
                out[f"R@{k}"] = len(set(top_k) & rel_set) / max(1, len(rel_set))
                dcg = sum(qrels.get(d, 0) / math.log2(i + 1)
                          for i, d in enumerate(top_k, 1))
                ideal = sorted((s for s in qrels.values() if s > 0), reverse=True)[:k]
                idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal, 1))
                out[f"NDCG@{k}"] = dcg / idcg if idcg else 0.0
            out["MRR"] = next(
                (1.0 / i for i, d in enumerate(ranked, 1) if d in rel_set), 0.0,
            )
            cum = hits = 0
            for i, d in enumerate(ranked, 1):
                if d in rel_set:
                    hits += 1
                    cum  += hits / i
            out["AP"] = cum / len(rel_set) if rel_set else 0.0
            return out


        def evaluate(strategy_name, search_fn):
            per_query = []
            for qid, qv, qt in zip(query_ids, q_vecs, q_texts):
                ranked = [r['doc_id'] for r in search_fn(qv, qt)]
                per_query.append(query_metrics(ranked, qrels.get(qid, {})))
            keys = per_query[0].keys()
            agg = {k: sum(q[k] for q in per_query) / len(per_query) for k in keys}
            agg["MAP"] = agg.pop("AP")
            return agg


        results = {
            "lexical (BM25)" : evaluate("lexical", lambda v, t: bm25_search(t)),
            "vector"         : evaluate("vector",  lambda v, t: vector_search(v)),
            "hybrid α=0.8"   : evaluate("hybrid",  lambda v, t: hybrid_search(v, t, alpha=0.8)),
        }

        import pandas as pd
        pd.DataFrame(results).T[["P@5", "R@5", "NDCG@5", "NDCG@10", "MRR", "MAP"]].round(3)
        """),
        md("""
        ## Step 6 — How does `α` move NDCG@10?

        The hybrid weight `α` is a knob, not a fixed setting:

        - `α = 1.0` — pure vector (same as `vector_search`).
        - `α = 0.0` — pure lexical (same as `bm25_search`).
        - `α = 0.5` — naïve uniform fusion. *Often the wrong default*.
        - `α = 0.8` — vector-favored. Good default on most BEIR data.

        Sweep `α` and watch NDCG@10:
        """),
        code("""
        ALPHAS = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
        ndcg_sweep = []
        for a in ALPHAS:
            per_query = []
            for qid, qv, qt in zip(query_ids, q_vecs, q_texts):
                ranked = [r['doc_id'] for r in hybrid_search(qv, qt, alpha=a)]
                per_query.append(query_metrics(ranked, qrels.get(qid, {})))
            mean_ndcg = sum(q['NDCG@10'] for q in per_query) / len(per_query)
            ndcg_sweep.append(mean_ndcg)
            print(f"  α={a:.1f}  NDCG@10={mean_ndcg:.3f}")
        """),
        code("""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(ALPHAS, ndcg_sweep, marker='o')
        ax.set_xlabel("hybrid α  (0 = lexical, 1 = vector)")
        ax.set_ylabel("NDCG@10")
        ax.set_title(f"{DATASET} — NDCG@10 vs α ({N_QUERIES} queries)")
        ax.grid(alpha=0.3)
        plt.show()
        """),
        md("""
        ## Reading the comparison

        Three things to look for:

        1. **Vector vs lexical NDCG@10** — embedding-based retrieval
           usually wins on BEIR-style datasets because it handles
           paraphrase and topical overlap. It loses on datasets
           dominated by exact strings (codes, names).

        2. **Does hybrid beat both?** Often yes — but only if α is
           weighted toward whichever single mode is stronger. Uniform
           α=0.5 drags vector down when vector is the better signal.

        3. **Which metric improves the most?** If vector and hybrid
           lift Recall@10 more than NDCG@10, they're *finding* more
           relevant docs but not ranking them at the top. That's the
           signal that a **cross-encoder reranker** would help — see
           `phase4/rerank.py` for the advanced track.

        ## What this is *not* telling you

        BEIR queries are homogeneous within a dataset — all scientific
        claims, or all health questions. A single α tuned on this
        dataset will win almost every query. Real production traffic
        is messier — a mix of exact-match lookups, single-word topics,
        compound questions, and chatty natural language all from the
        same users. To know what works on *your* mix, you need *your
        own* eval data.

        **Next:** open **`03_curate_eval_set.ipynb`**.
        """),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Notebook 03 — Lesson 3: curate your own evaluation dataset
# ─────────────────────────────────────────────────────────────────────────────

def nb03() -> list[dict]:
    return [
        md("""
        # Lab 3 — Curate your own evaluation dataset

        **Companion to course Lesson 3.**

        BEIR is a fixed benchmark. It tells you how a retrieval setup
        does on *scientific claim verification* or *medical literature
        retrieval*. That's useful for comparing approaches in the
        abstract. It is *not* useful for answering the question your
        team actually has: **does this retrieve well on the kind of
        queries our users ask, against the documents we actually
        have?** For that you need a domain-specific evaluation set
        built on your own corpus.

        ## What you'll build

        A small judgement list (queries + relevance labels) over the
        corpus you already ingested, using the standard bootstrap
        pattern:

        1. **Sample** documents from your corpus.
        2. **Draft queries** with an LLM — one realistic query per doc.
        3. **Pool** candidates by retrieving top-K for each draft
           query (we'll use `$rankFusion`).
        4. **Grade** each `(query, candidate)` pair with an LLM judge
           on a 0–3 scale.
        5. **Curate** — you spot-check and edit. The LLM is fast and
           consistent; *you* know what counts as relevant in your
           domain.
        6. **Save & re-evaluate** — write the judgements to disk and
           rerun Lab 1's metrics against them.

        > **Requires `OPENAI_API_KEY` in `.env`.** The full bootstrap
        > on 20 docs ≈ 60 LLM calls — a few cents on `gpt-4o-mini`.
        """),
        code(SETUP_CELL),
        code("""
        from dotenv import load_dotenv
        load_dotenv(os.path.join(_REPO_ROOT, ".env"))

        assert os.environ.get("OPENAI_API_KEY"), \\
            "Set OPENAI_API_KEY in .env before running this lab."

        import math, json, pathlib, pymongo, voyageai
        from openai import OpenAI
        from lib import embed_queries

        DATASET           = "scifact"
        DB_NAME           = "voyage_context_demo"
        COLL_NAME         = f"chunks_{DATASET.replace('-', '_')}"
        VECTOR_INDEX_NAME = "voyage_vector_index"
        TEXT_INDEX_NAME   = "voyage_text_index"
        MONGODB_BASE_URL  = "https://ai.mongodb.com/v1"

        N_DOCS    = 20            # corpus sample for the bootstrap
        TOP_K     = 8             # candidates per query to grade
        LLM_MODEL = "gpt-4o-mini"

        client = pymongo.MongoClient(os.environ["MONGODB_URI"])
        coll   = client[DB_NAME][COLL_NAME]
        voyage = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"],
                                 base_url=MONGODB_BASE_URL)
        oai    = OpenAI()
        """),
        md("""
        ## Step 1 — Sample documents from the corpus

        Pull `N_DOCS` distinct parent documents from our chunks
        collection. Each will seed one draft query.
        """),
        code("""
        sample_docs = []
        seen = set()
        for row in coll.find({}, {"doc_id": 1, "title": 1, "text": 1}).limit(N_DOCS * 4):
            did = row['doc_id']
            if did in seen:
                continue
            seen.add(did)
            sample_docs.append({
                "doc_id": did,
                "title" : row.get('title', ''),
                "text"  : row['text'][:1500],
            })
            if len(sample_docs) >= N_DOCS:
                break

        print(f"Sampled {len(sample_docs)} documents:")
        for d in sample_docs[:3]:
            print(f"  {d['doc_id']:<12} {d['title'][:55]!r}")
        """),
        md("""
        ## Step 2 — Draft a query for each document

        We ask the LLM to write **one realistic query** that a domain
        user might type to find each document. The prompt is the
        lever — phrase it for your real users (keyword-style vs
        natural-language vs multi-hop).
        """),
        code("""
        QUERY_PROMPT = (
            "You are helping build a retrieval evaluation set. Given a document, "
            "write ONE concise, realistic search query that a domain user might "
            "type to retrieve this exact document. The query should be 3-15 words "
            "and read naturally — not a paraphrase of the title.\\n\\n"
            "Document title : {title}\\n"
            "Document text  : {text}\\n\\n"
            "Output ONLY the query, no quotes or commentary."
        )

        draft_queries = []
        for i, d in enumerate(sample_docs, 1):
            resp = oai.chat.completions.create(
                model       = LLM_MODEL,
                temperature = 0.7,
                messages    = [{"role": "user",
                                "content": QUERY_PROMPT.format(title=d['title'], text=d['text'])}],
            )
            q = resp.choices[0].message.content.strip().strip("\\"'")
            draft_queries.append({
                "qid"        : f"my_q_{i:03d}",
                "query"      : q,
                "seed_doc_id": d['doc_id'],
            })
            print(f"  [{i:>2}] {q}")
        """),
        md("""
        ## Step 3 — Pool candidates

        For each draft query, retrieve the top-K with a hybrid
        `$rankFusion`. The union of these results is the **pool** —
        the set of `(query, doc)` pairs we'll grade. Pooling is the
        standard IR evaluation pattern (TREC + BEIR both use it): we
        don't grade every doc in the corpus, only the docs that
        *some* retriever surfaced.
        """),
        code("""
        # Embed the draft queries in one batch.
        q_vecs = embed_queries(voyage, [q['query'] for q in draft_queries])

        def hybrid_search(q_vec, q_text, alpha=0.8, top_k=TOP_K):
            EPS = 1e-3
            w_vec, w_text, n = max(alpha, EPS), max(1.0 - alpha, EPS), 50
            pipeline = [
                {"$rankFusion": {
                    "input": {"pipelines": {
                        "vector": [{"$vectorSearch": {
                            "index": VECTOR_INDEX_NAME, "path": "embedding",
                            "queryVector": q_vec,
                            "numCandidates": n * 4, "limit": n,
                        }}],
                        "text": [{"$search": {
                            "index": TEXT_INDEX_NAME,
                            "text" : {"path": "text", "query": q_text},
                        }}, {"$limit": n}],
                    }},
                    "combination": {"weights": {"vector": w_vec, "text": w_text}},
                }},
                {"$addFields": {"score": {"$meta": "score"}}},
                {"$limit": top_k * 4},
            ]
            seen = {}
            for row in coll.aggregate(pipeline):
                did = row['doc_id']
                if did not in seen or row['score'] > seen[did]['score']:
                    seen[did] = row
            return sorted(seen.values(), key=lambda r: r['score'], reverse=True)[:top_k]

        pool = {}
        for q, qv in zip(draft_queries, q_vecs):
            pool[q['qid']] = hybrid_search(qv, q['query'])

        total = sum(len(v) for v in pool.values())
        print(f"Pooled {total} (query, candidate) pairs across {len(pool)} queries.")
        """),
        md("""
        ## Step 4 — LLM judge

        Grade each `(query, candidate)` pair on the BEIR-standard 0–3
        scale:

        - **3** — highly relevant: directly answers the query.
        - **2** — relevant: clearly addresses the query topic.
        - **1** — marginally relevant: tangentially related.
        - **0** — not relevant.

        We force JSON output so we can parse it deterministically.
        """),
        code("""
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
            "Output ONLY a JSON object: "
            '{{"grade": <0|1|2|3>, "reason": "<short>"}}'
        )

        def judge(query: str, passage: str) -> tuple[int, str]:
            resp = oai.chat.completions.create(
                model           = LLM_MODEL,
                temperature     = 0.0,
                response_format = {"type": "json_object"},
                messages        = [{"role": "user",
                                    "content": JUDGE_PROMPT.format(
                                        query=query, passage=passage[:1500])}],
            )
            data = json.loads(resp.choices[0].message.content)
            return int(data.get('grade', 0)), data.get('reason', '')

        draft_qrels = {}
        reasons     = {}
        done = 0
        for q in draft_queries:
            qid = q['qid']
            draft_qrels[qid] = {}
            for row in pool[qid]:
                grade, reason = judge(q['query'], row['text'])
                draft_qrels[qid][row['doc_id']] = grade
                reasons[(qid, row['doc_id'])]   = reason
                done += 1
                if done % 10 == 0:
                    print(f"  graded {done} / ~{sum(len(v) for v in pool.values())}")
        print(f"  graded {done} pairs.")
        """),
        md("""
        ## Step 5 — Review and curate

        This is the critical human-in-the-loop step. The LLM judge is
        a *draft*. Skim the high-grade and low-grade ends — that's
        where mistakes hide. Tweak any cell of `draft_qrels` directly.
        A small, carefully reviewed set beats a large, blindly-trusted
        one.
        """),
        code("""
        import pandas as pd

        review_rows = []
        for q in draft_queries:
            qid = q['qid']
            for row in pool[qid]:
                did = row['doc_id']
                review_rows.append({
                    "qid"   : qid,
                    "query" : q['query'][:50],
                    "doc_id": did,
                    "grade" : draft_qrels[qid].get(did, 0),
                    "reason": reasons.get((qid, did), '')[:60],
                    "title" : (row.get('title') or '')[:50],
                })
        review_df = pd.DataFrame(review_rows).sort_values(
            ["qid", "grade"], ascending=[True, False],
        )
        review_df.head(20)
        """),
        code("""
        # Want to override a judgement? Edit it directly. Example:
        #   draft_qrels["my_q_001"]["12345"] = 2
        # then re-run the previous cell to see the updated table.
        """),
        md("""
        ## Step 6 — Save and re-evaluate

        Write the curated qrels to disk, then run the three search
        strategies from Lab 2 against *your* qrels. These numbers
        tell you which strategy works best **on your data**, not on
        BEIR's.
        """),
        code("""
        EVAL_DIR = pathlib.Path(_REPO_ROOT) / "eval_sets"
        EVAL_DIR.mkdir(exist_ok=True)

        (EVAL_DIR / f"{DATASET}_custom_queries.json").write_text(
            json.dumps({q['qid']: q['query'] for q in draft_queries}, indent=2),
        )
        (EVAL_DIR / f"{DATASET}_custom_qrels.json").write_text(
            json.dumps(draft_qrels, indent=2),
        )
        print(f"Wrote {EVAL_DIR}")
        print(f"  {DATASET}_custom_queries.json")
        print(f"  {DATASET}_custom_qrels.json")
        """),
        code("""
        def query_metrics(ranked, qrels_for_q, ks=(5, 10)):
            rel_set = {did for did, s in qrels_for_q.items() if s > 0}
            out = {}
            for k in ks:
                top_k = ranked[:k]
                out[f"P@{k}"] = sum(1 for d in top_k if d in rel_set) / k
                out[f"R@{k}"] = len(set(top_k) & rel_set) / max(1, len(rel_set))
                dcg = sum(qrels_for_q.get(d, 0) / math.log2(i + 1)
                          for i, d in enumerate(top_k, 1))
                ideal = sorted((s for s in qrels_for_q.values() if s > 0), reverse=True)[:k]
                idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal, 1))
                out[f"NDCG@{k}"] = dcg / idcg if idcg else 0.0
            out["MRR"] = next(
                (1.0 / i for i, d in enumerate(ranked, 1) if d in rel_set), 0.0,
            )
            cum = hits = 0
            for i, d in enumerate(ranked, 1):
                if d in rel_set:
                    hits += 1
                    cum  += hits / i
            out["AP"] = cum / len(rel_set) if rel_set else 0.0
            return out


        def dedup(rows, k):
            seen = {}
            for r in rows:
                did = r['doc_id']
                if did not in seen or r['score'] > seen[did]['score']:
                    seen[did] = r
            return sorted(seen.values(), key=lambda r: r['score'], reverse=True)[:k]

        def bm25_search(q_text, top_k=TOP_K):
            pipeline = [
                {"$search": {"index": TEXT_INDEX_NAME,
                              "text": {"path": "text", "query": q_text}}},
                {"$addFields": {"score": {"$meta": "searchScore"}}},
                {"$limit": top_k * 4},
                {"$sort":  {"score": -1}},
            ]
            return [r['doc_id'] for r in dedup(coll.aggregate(pipeline), top_k)]

        def vector_search(q_vec, top_k=TOP_K):
            pipeline = [
                {"$vectorSearch": {
                    "index": VECTOR_INDEX_NAME, "path": "embedding",
                    "queryVector": q_vec,
                    "numCandidates": top_k * 20, "limit": top_k * 4,
                }},
                {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
                {"$sort": {"score": -1}},
            ]
            return [r['doc_id'] for r in dedup(coll.aggregate(pipeline), top_k)]

        results = {}
        for name, search in [
            ("lexical (BM25)", lambda v, t: bm25_search(t)),
            ("vector",         lambda v, t: vector_search(v)),
            ("hybrid α=0.8",   lambda v, t: [r['doc_id'] for r in hybrid_search(v, t, alpha=0.8)]),
        ]:
            per_q = []
            for q, qv in zip(draft_queries, q_vecs):
                ranked = search(qv, q['query'])
                per_q.append(query_metrics(ranked, draft_qrels[q['qid']]))
            keys = per_q[0].keys()
            agg = {k: sum(p[k] for p in per_q) / len(per_q) for k in keys}
            agg["MAP"] = agg.pop("AP")
            results[name] = agg

        pd.DataFrame(results).T[["P@5", "R@5", "NDCG@5", "NDCG@10", "MRR", "MAP"]].round(3)
        """),
        md("""
        ## Where to go next

        You've just built a tiny custom evaluation set. To take it
        further:

        - **Scale up.** 20 queries is a smoke test; aim for 100+
          before trusting the numbers to drive decisions.
        - **Diversify queries.** Sample from real user logs if you
          have them. If not, prompt the LLM for multiple *styles*
          (keyword, natural-language, multi-hop).
        - **Tighten the judging prompt.** Add a couple of in-prompt
          examples from your domain. Calibrate the LLM on a handful
          of human-graded examples before scaling.
        - **Re-curate.** Eval sets decay as corpora and user needs
          change. Treat them as living code, not finished artefacts.

        Once you have a trustworthy eval set, the **advanced** lab
        under `phase4/` is where you'd go to push retrieval quality
        further — per-query strategy routing, query rewriters
        (HyDE, decompose), cross-encoder reranking. None of those
        are worth tuning without an eval set you trust. That's
        the whole point.
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
