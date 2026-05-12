# Phase 4 — advanced retrieval

Everything in this folder builds on top of the lab in `notebooks/`. None
of it is required to understand IR evaluation, but once you have a
reliable evaluation set (notebook 03), this is where you'd go to push
retrieval quality beyond a fixed black box.

## What's here

| File | Purpose |
|---|---|
| `query.py` | Full retrieval CLI with `--mode {vector,text,hybrid}`, `--rewriter`, `--rerank`, and `--strategy {static,dynamic}` flags |
| `test_harness.py` | Datasets × strategies × rerank-on/off matrix; renders Markdown + bar-chart comparison report |
| `query_rewriter.py` | LLM query rewriters: `none`, `hyde` (hypothetical-doc), `multi` (paraphrase fan-out), `decompose` (sub-questions) |
| `query_classifier.py` | One LLM call → per-query routing: `Strategy(alpha, rerank, rewriter)`. Used by `--strategy dynamic` |
| `rerank.py` | Second-stage cross-encoder via Voyage `rerank-2.5` |
| `llm_client.py` | Thin OpenAI wrapper used by the rewriter, classifier, and judge |
| `llm_judge.py` | LLM-as-a-judge: pools retrieval results across strategies, grades on the 0–3 scale, saves BEIR-shape qrels |
| `data_loaders/` | Non-BEIR dataset loaders. `sec_10k.py` builds a 15-company / 4-year SEC 10-K corpus live, with 300 LLM-generated trader queries |
| `experiments/` | Multi-dataset benchmark sweeps that generated the findings in the root `CLAUDE.md` |
| `judge_cache.json` | Cached LLM judge grades — keep this around between runs to amortize the cost |

## Running

All scripts here import from the repo-root library (`lib.py`,
`retrieve.py`, `lib_metrics.py`, `ingest.py`). A small `sys.path` shim
at the top of each entry script handles that, so you can run them from
the repo root:

```bash
# Full CLI with all the bells
python3 phase4/query.py scifact --mode hybrid --rerank --strategy dynamic

# Quick smoke test across 3 small datasets, 3 queries each
python3 phase4/test_harness.py --quick --sample 100 --num-queries 3

# Sweep strategies on one dataset
python3 phase4/experiments/compare_fusion_strategies.py scifact

# Full benchmark across all 8 BEIR datasets
python3 phase4/experiments/compare_all_datasets.py --sample 500 --num-queries 30

# LLM-judge eval against the SEC 10-K corpus
python3 phase4/experiments/sec_10k_eval.py --strategies vector hybrid_a08 dynamic
```

## What the experiments showed

(Reproduced from the root `CLAUDE.md` — these were generated with the
500-doc / 10-query setup that `experiments/compare_fusion_strategies.py`
runs by default.)

| Strategy | scifact | nfcorpus | touche2020 |
|---|---|---|---|
| vector              | 0.874 | **0.781** | 0.913 |
| text                | 0.711 | 0.419 | 0.911 |
| hybrid α=0.5 (RRF)  | 0.833 | 0.662 | 0.941 |
| hybrid α=0.8 (RRF)  | 0.872 | 0.745 | **0.946** |
| dynamic (per-query) | 0.872 | 0.726 | 0.944 |

(NDCG@10. Bold = best per dataset.)

Two main lessons:

1. The naïve α=0.5 hybrid is the wrong default — it drags vector
   quality down on datasets where embeddings dominate. α=0.8 closes
   most of the gap.
2. The per-query dynamic router matches static α=0.8 on these
   homogeneous BEIR datasets without any per-dataset tuning. Where
   the router actually pulls ahead is on **heterogeneous query
   streams** that mix exact-match lookups, single-word topics,
   compound questions, and natural-language asks — which is what
   real production traffic looks like, and what no static config
   can handle.
