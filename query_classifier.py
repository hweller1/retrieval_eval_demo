"""
Per-query strategy router.

Given a query, a cheap LLM (gpt-4o-mini) returns a full retrieval
strategy in one call:

  Strategy(alpha, rerank, rewriter, reasoning)

  alpha    — vector weight in [0, 1] for hybrid fusion
  rerank   — apply rerank-2.5 cross-encoder as second stage
  rewriter — "none" | "hyde" | "multi" | "decompose"

The classifier is **dataset-agnostic**: it looks only at query
characteristics (length, exact-string artifacts, conceptual abstraction,
multi-hop structure) and routes accordingly.

Design choices:
  - Few-shot prompt with 8 diverse examples spanning the alpha range
    and showing when each rewriter / rerank is appropriate. Without
    these the LLM tends to bucket every query into one of "keyword
    (0.3)" / "balanced (0.5)" / "conceptual (0.85)".
  - JSON output for parseability. `reasoning` is included so traces are
    debuggable and demoable.
  - Cached by query string. Repeat calls free.
  - Fall-back strategy on parse failure: alpha=0.7, rerank=False,
    rewriter="none" — the conservative middle ground.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field
from typing import Iterable

import llm_client


DEFAULT_MODEL = "gpt-4o-mini"


@dataclass
class Strategy:
    alpha: float        # vector weight in [0, 1]
    rerank: bool        # apply rerank-2.5 second stage
    rewriter: str       # "none" | "hyde" | "multi" | "decompose"
    reasoning: str = "" # short rationale, for debuggability

    def label(self) -> str:
        parts = [f"α={self.alpha:.2f}"]
        if self.rewriter != "none":
            parts.append(f"+{self.rewriter}")
        if self.rerank:
            parts.append("+rerank")
        return " ".join(parts)


SYSTEM_PROMPT = """\
You are a retrieval strategy router. Given one search query against a
chunked document corpus, you pick four parameters:

  alpha    — vector weight in [0, 1] for hybrid (vector + BM25) fusion.
             0.0 = pure BM25, 1.0 = pure semantic, 0.5 = balanced.
             The goal is **topical relevance** (find documents *about*
             the query's subject), not literal string matching.
  rerank   — true/false. Cross-encoder over the top-50 candidates from
             first-stage. Costs ~3× latency. Enable ONLY when the query
             is ambiguous or under-specified enough that the first-stage
             ranking is likely noisy. Do NOT enable just because the
             query is technical.
  rewriter — pick ONE: "none" | "hyde" | "multi" | "decompose".
               none      — well-formed query, retrieve as-is.
               hyde      — single-word or ≤3-word query; LLM drafts a
                           hypothetical answer passage and embeds that.
               multi     — clear vocabulary-mismatch risk (idioms, slang,
                           casual phrasing).
               decompose — TRULY independent compound: two questions that
                           could each be Googled separately and would
                           return disjoint sets of relevant docs.
                           "X and Y" where X causes Y, or X is part of Y,
                           is NOT this — that's one topic, use "none".

Routing heuristics (conservative — when in doubt, prefer α≥0.7 + rewriter=none + rerank=false):

LOW α (≤0.30): query is purely an exact-string artifact — e.g. a CVE
  code, ticker symbol, error message, file path, ID, or a bare named
  entity with no surrounding sentence. A query with a NUMBER inside a
  full sentence (e.g. "5% of X are Y") is NOT this — it's a conceptual
  claim, use α≥0.7.

MID α (0.50–0.65): mix of exact-string + natural language (e.g.
  "log4shell java enterprise apps") OR very short queries (1–3 words)
  where neither vector nor BM25 alone is reliable.

HIGH α (0.70–0.85): well-formed sentences or sentence fragments asking
  about a topic, even when they contain numbers, named entities,
  acronyms, or technical jargon. This is the most common case. Default
  here unless the query clearly fits LOW or MID.

Rerank=true is REQUIRED only when the query is genuinely ambiguous or
short enough that first-stage ranking is unreliable. Default false.

Rewriter=none is the default. Switch only when the query clearly fits
one of the rewriter triggers above.

Output ONLY a single JSON object, no prose:
  {"alpha": 0.80, "rerank": false, "rewriter": "none",
   "reasoning": "short reason"}"""


# ── Few-shot examples deliberately span the alpha range and rewriter modes.
# These are appended to the user message before the actual query. The
# examples are chosen to be obviously different so the LLM doesn't collapse
# everything to one bucket.
FEW_SHOT_EXAMPLES = [
    # ── LOW α: pure exact-string artifacts ───────────────────────────────
    {
        "query": "CVE-2021-44228",
        "out": {"alpha": 0.15, "rerank": False, "rewriter": "none",
                 "reasoning": "bare CVE identifier — BM25 dominates"},
    },
    {
        "query": "AAPL",
        "out": {"alpha": 0.20, "rerank": False, "rewriter": "none",
                 "reasoning": "bare ticker symbol — exact-string match"},
    },
    # ── MID α: short or mixed ────────────────────────────────────────────
    {
        "query": "vaping",
        "out": {"alpha": 0.55, "rerank": True, "rewriter": "hyde",
                 "reasoning": "single word — too thin for vector; HyDE expands"},
    },
    {
        "query": "log4shell java enterprise impact",
        "out": {"alpha": 0.55, "rerank": False, "rewriter": "none",
                 "reasoning": "named vulnerability + brief scope — both signals matter"},
    },
    # ── HIGH α: full sentences, even with numbers / acronyms / names ────
    {
        "query": "5% of perinatal mortality is due to low birth weight.",
        "out": {"alpha": 0.80, "rerank": False, "rewriter": "none",
                 "reasoning": "full claim sentence — semantic match for relevance, "
                              "not literal-string match for the statistic"},
    },
    {
        "query": "ADAR1 binds to Dicer to cleave pre-miRNA.",
        "out": {"alpha": 0.80, "rerank": False, "rewriter": "none",
                 "reasoning": "scientific claim with technical names; vector finds "
                              "papers about this interaction better than BM25"},
    },
    {
        "query": "Should the death penalty be abolished in democratic societies?",
        "out": {"alpha": 0.80, "rerank": False, "rewriter": "none",
                 "reasoning": "well-formed conceptual debate question"},
    },
    {
        "query": "How does the gut microbiome affect mood and is it different in vegans?",
        "out": {"alpha": 0.75, "rerank": False, "rewriter": "decompose",
                 "reasoning": "compound multi-hop (microbiome→mood AND vegan diet)"},
    },
]


def _build_user_prompt(query: str) -> str:
    lines = ["Examples:"]
    for ex in FEW_SHOT_EXAMPLES:
        lines.append(f"  Query: {ex['query']}")
        lines.append(f"  → {json.dumps(ex['out'])}")
    lines.append("")
    lines.append(f"Now classify this query:")
    lines.append(f"  Query: {query}")
    lines.append("  → ")
    return "\n".join(lines)


_strategy_cache: dict[str, Strategy] = {}


def predict_strategy(query: str, model: str = DEFAULT_MODEL) -> Strategy:
    if query in _strategy_cache:
        return _strategy_cache[query]

    raw = llm_client.complete(
        prompt=_build_user_prompt(query),
        system=SYSTEM_PROMPT,
        model=model,
        temperature=0.0,
        max_tokens=120,
    )
    strat = _parse_strategy(raw)
    _strategy_cache[query] = strat
    return strat


def predict_strategies(queries: Iterable[str], model: str = DEFAULT_MODEL) -> list[Strategy]:
    return [predict_strategy(q, model=model) for q in queries]


# ── Backwards compat: the alpha-only API is preserved so older callers
# (and the experiment script's static-α path) keep working.

def predict_alpha(query: str, model: str = DEFAULT_MODEL) -> float:
    return predict_strategy(query, model=model).alpha


def predict_alphas(queries: Iterable[str], model: str = DEFAULT_MODEL) -> list[float]:
    return [s.alpha for s in predict_strategies(queries, model=model)]


# ── Parsing & fallback ───────────────────────────────────────────────────────

_FALLBACK = Strategy(alpha=0.7, rerank=False, rewriter="none",
                     reasoning="<<parse failed; fallback>>")


def _parse_strategy(raw: str) -> Strategy:
    raw = raw.strip()
    # Strip code-fence wrappers if present
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE)
        raw = raw.strip()

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        # Sometimes the LLM emits trailing prose; try to extract the first {...}
        match = re.search(r"\{.*?\}", raw, flags=re.DOTALL)
        if not match:
            return _FALLBACK
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            return _FALLBACK

    if not isinstance(obj, dict):
        return _FALLBACK

    try:
        alpha = float(obj.get("alpha", 0.7))
    except (TypeError, ValueError):
        alpha = 0.7
    alpha = max(0.0, min(1.0, alpha))

    rerank = bool(obj.get("rerank", False))
    rewriter = str(obj.get("rewriter", "none")).lower().strip()
    if rewriter not in ("none", "hyde", "multi", "decompose"):
        rewriter = "none"

    reasoning = str(obj.get("reasoning", ""))[:200]
    return Strategy(alpha=alpha, rerank=rerank, rewriter=rewriter, reasoning=reasoning)


def clear_cache() -> None:
    _strategy_cache.clear()
