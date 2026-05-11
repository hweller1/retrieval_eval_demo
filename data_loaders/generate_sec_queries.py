"""
One-shot script to generate ~300 trader-style queries against the SEC
10-K corpus and save them to data_loaders/sec_queries.json.

Three categories, ~100 queries each:

  business_model — How does the company make money? Customer types,
                   product mix, geographic split, distribution channels,
                   monetization shifts.

  key_risks      — Material risk factors a trader would screen for:
                   regulatory, customer concentration, supply chain,
                   technology disruption, litigation, FX exposure, etc.

  financial_health — Revenue sustainability, competitive moat, gross
                     margin trends, cost structure, management's
                     justification of underperformance, capital
                     allocation, free cash flow generation.

Mix is roughly:
  - 40% company-specific: "How does <COMPANY> generate revenue?"
  - 40% cross-company:    "Which of these companies has the highest
                           customer concentration risk?"
  - 20% structural:       "What are the most common AI-related risks
                           cited in tech 10-Ks?"

Generated once with gpt-4o-mini, cached as JSON, committed to repo.
Re-run only if you want a fresh set.

Usage:
  python3 data_loaders/generate_sec_queries.py
  python3 data_loaders/generate_sec_queries.py --n-per-category 100
"""

from __future__ import annotations

import sys
import json
import argparse
import pathlib

sys.path.insert(0, ".")

import llm_client
from data_loaders.sec_10k import COMPANIES


OUT_PATH = pathlib.Path(__file__).parent / "sec_queries.json"


SYSTEM = """\
You generate realistic search queries that a buy-side equity analyst or
quantitative trader would issue against a corpus of SEC 10-K annual
reports. The corpus contains 10-Ks from these US tech companies:

{tickers}

Each query should be:
  - 5–18 words long
  - Phrased as a natural-language question or a topic phrase
    (mix both, ~half and half)
  - Specific enough that a sensible top-10 retrieval would return
    text passages directly addressing it
  - Genuinely useful for assessing a trading thesis (entry/exit,
    risk/return)

Avoid queries that require numerical math the corpus can't answer
(e.g. "what is the P/E ratio") — stick to questions answerable from
narrative 10-K text (Items 1, 1A, 7).

Output ONLY a JSON array of strings, no commentary, no code fences."""


CATEGORY_PROMPTS = {
    "business_model": """\
Generate {n} CORE BUSINESS MODEL queries. Mix of:
  - company-specific: "How does {sample_co_1} make money?"
  - cross-company: "Which company has the largest software-as-a-service revenue mix?"
  - structural: "What are the typical revenue recognition policies for cloud-infrastructure providers?"

Cover: revenue segmentation, customer mix, geographic split, product
roadmap, partnership/channel reliance, pricing models, monetization
shifts (subscription vs perpetual, usage-based vs flat-fee), recent
acquisitions and how they're integrated.""",

    "key_risks": """\
Generate {n} KEY RISKS queries. Mix of:
  - company-specific: "What customer concentration risks does {sample_co_2} disclose?"
  - cross-company: "Which of these chip companies has the highest fab dependency on TSMC?"
  - structural: "What are the most-cited AI-related risk factors in tech 10-Ks?"

Cover: regulatory (antitrust, AI regulation, privacy), customer
concentration, supply-chain dependencies (foundries, cloud providers,
critical components), litigation exposure, FX and macro, cybersecurity
incidents, talent retention, technology disruption, ESG-linked.""",

    "financial_health": """\
Generate {n} FINANCIAL HEALTH queries focused on revenue sustainability,
competitive advantage, and management justifications for performance.
Mix of:
  - company-specific: "How does {sample_co_3} justify declining gross margins?"
  - cross-company: "Which of these companies has the most durable moat per their own MD&A?"
  - structural: "How do tech companies typically explain billings vs revenue divergence?"

Cover: revenue growth durability, gross/operating margin trajectory,
unit economics narrative, customer retention/churn signals, R&D
intensity, capital expenditure rationale, share buybacks vs reinvestment,
free cash flow conversion, management's framing of misses or guide-downs,
moat language (network effects, switching costs, scale economies)."""
}


def generate_for_category(category: str, n: int, tickers: list[str]) -> list[str]:
    sample_cos = (tickers * 4)[:8]  # cycle if list short
    prompt = CATEGORY_PROMPTS[category].format(
        n=n,
        sample_co_1=sample_cos[0],
        sample_co_2=sample_cos[1],
        sample_co_3=sample_cos[2],
    )
    raw = llm_client.complete(
        prompt=prompt,
        system=SYSTEM.format(tickers=", ".join(tickers)),
        model="gpt-4o-mini",
        temperature=0.7,    # we want diversity
        max_tokens=4000,
    )
    raw = raw.strip()
    # Strip code fences if the model added them
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
    queries = json.loads(raw)
    if not isinstance(queries, list):
        raise ValueError(f"expected JSON array, got {type(queries)}")
    return [q.strip() for q in queries if q and q.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-category", type=int, default=100)
    p.add_argument("--out", type=str, default=str(OUT_PATH))
    args = p.parse_args()

    tickers = [t for t, _, _ in COMPANIES]
    print(f"Generating {args.n_per_category} queries × 3 categories "
          f"({args.n_per_category * 3} total) for {len(tickers)} companies …")

    all_queries: dict[str, str] = {}
    qid_counter = 1
    for category in ("business_model", "key_risks", "financial_health"):
        print(f"  [{category}] generating {args.n_per_category} queries …")
        # The model sometimes returns slightly fewer than asked; iterate
        # until we have at least n.
        seen: set[str] = set()
        attempts = 0
        while len(seen) < args.n_per_category and attempts < 5:
            attempts += 1
            try:
                batch = generate_for_category(category, args.n_per_category, tickers)
            except Exception as e:
                print(f"    attempt {attempts} failed: {e}; retrying")
                continue
            for q in batch:
                if q.lower() not in {s.lower() for s in seen}:
                    seen.add(q)
                if len(seen) >= args.n_per_category:
                    break
        cat_queries = list(seen)[:args.n_per_category]
        for q in cat_queries:
            qid = f"{category}_{qid_counter:03d}"
            all_queries[qid] = q
            qid_counter += 1
        print(f"  [{category}] kept {len(cat_queries)}")

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(all_queries, indent=2), encoding="utf-8")
    print(f"\nWrote {len(all_queries)} queries to {args.out}")


if __name__ == "__main__":
    main()
