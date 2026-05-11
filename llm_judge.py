"""
LLM-as-a-judge evaluation for novel datasets that lack human qrels.

The flow:
  1. Run all retrieval strategies on all queries; collect the top-K
     chunks each strategy returns.
  2. Pool those results — for each (query, doc_id) seen by any strategy,
     ask the LLM to grade how relevant doc_id's best chunk is.
  3. Save graded relevance to a qrels JSON file (BEIR-compatible:
     {qid: {doc_id: int_score}}).
  4. Re-use lib_metrics to compute NDCG, MAP, etc. from these qrels.

Pooled grading is the standard IR evaluation method (TREC, BEIR all use
pooling) — we grade only the union of what's actually retrieved, not the
full corpus, which keeps grading cost bounded.

Grading scale (BEIR / TREC convention):
  3 — Highly relevant: directly answers the query
  2 — Relevant: closely related, useful context
  1 — Marginally relevant: tangentially related
  0 — Not relevant
"""

from __future__ import annotations

import json
import re
import textwrap
import threading
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

import llm_client


JUDGE_MODEL = "gpt-4o-mini"
JUDGE_CACHE_PATH = pathlib.Path("judge_cache.json")
GRADE_RANGE = (0, 3)


SYSTEM = """\
You are an information retrieval relevance judge. Given a search query
and a passage from a document, grade how relevant the passage is to the
query on this scale:

  3 — Highly relevant: directly answers the query.
  2 — Relevant: clearly addresses the query topic, useful context.
  1 — Marginally relevant: tangentially related; mentions the topic
      but doesn't really answer.
  0 — Not relevant.

Output ONLY a single JSON object: {"grade": <0|1|2|3>, "reason": "<short>"}.
No prose, no code fences."""


_cache_lock = threading.Lock()
_cache: dict[str, dict] | None = None


def _load_cache() -> dict[str, dict]:
    """Lazy-load on first call. Subsequent calls return the shared in-memory
    dict. Thread-safe: file I/O is serialized under the lock, and an empty
    or corrupt file is treated as "no cache yet" rather than crashing.
    """
    global _cache
    with _cache_lock:
        if _cache is not None:
            return _cache
        if JUDGE_CACHE_PATH.exists():
            try:
                raw = JUDGE_CACHE_PATH.read_text(encoding="utf-8")
                _cache = json.loads(raw) if raw.strip() else {}
            except (json.JSONDecodeError, OSError):
                # Cache file is empty/corrupt mid-write. Try the .bak
                # snapshot, otherwise start fresh.
                bak = JUDGE_CACHE_PATH.with_suffix(".json.bak")
                if bak.exists():
                    try:
                        _cache = json.loads(bak.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        _cache = {}
                else:
                    _cache = {}
        else:
            _cache = {}
        return _cache


def _save_cache() -> None:
    """Atomic save: write to .tmp, fsync, rename over the target. Avoids
    readers seeing a half-written file."""
    if _cache is None:
        return
    tmp = JUDGE_CACHE_PATH.with_suffix(".json.tmp")
    bak = JUDGE_CACHE_PATH.with_suffix(".json.bak")
    payload = json.dumps(_cache, indent=2)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(payload)
        f.flush()
        import os as _os
        _os.fsync(f.fileno())
    # Snapshot the previous good copy before overwriting (best-effort)
    if JUDGE_CACHE_PATH.exists():
        try:
            JUDGE_CACHE_PATH.replace(bak)
        except OSError:
            pass
    tmp.replace(JUDGE_CACHE_PATH)


def _cache_key(query: str, passage: str) -> str:
    # Stable key: query | first 1000 chars of passage. The same query+passage
    # combination always grades to the same value (judge is deterministic at
    # temp=0, but cache lets us not re-pay tokens).
    return f"{query}\n||\n{passage[:1000]}"


def _parse_grade(raw: str) -> dict:
    """Returns {'grade': int 0-3, 'reason': str}. Falls back to grade=0."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*?\}", raw, flags=re.DOTALL)
        if not m:
            return {"grade": 0, "reason": "<unparseable>"}
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {"grade": 0, "reason": "<unparseable>"}
    try:
        g = int(obj.get("grade", 0))
    except (TypeError, ValueError):
        g = 0
    g = max(GRADE_RANGE[0], min(GRADE_RANGE[1], g))
    reason = str(obj.get("reason", ""))[:200]
    return {"grade": g, "reason": reason}


def grade_one(query: str, passage: str, model: str = JUDGE_MODEL) -> dict:
    cache = _load_cache()
    key = _cache_key(query, passage)
    with _cache_lock:
        cached = cache.get(key)
    if cached is not None:
        return cached

    raw = llm_client.complete(
        prompt=f"Query: {query}\n\nPassage:\n{passage[:1500]}",
        system=SYSTEM,
        model=model,
        temperature=0.0,
        max_tokens=80,
    )
    result = _parse_grade(raw)

    with _cache_lock:
        cache[key] = result
    return result


# ── Batched parallel grading ─────────────────────────────────────────────────

def grade_batch(
    pairs: list[tuple[str, str, str]],   # (qid, doc_id, passage)
    queries: dict[str, str],
    max_workers: int = 8,
    save_every: int = 100,
    progress: bool = True,
) -> dict[str, dict[str, int]]:
    """
    Grade a batch of (qid, doc_id, passage) triples in parallel.
    Returns qrels in BEIR shape: {qid: {doc_id: grade}}.

    Uses on-disk cache so re-runs of the same (query, passage) pairs are
    free. Saves cache every `save_every` graded pairs to survive crashes.
    """
    qrels: dict[str, dict[str, int]] = {}
    completed = 0

    def task(triple):
        qid, doc_id, passage = triple
        result = grade_one(queries[qid], passage)
        return qid, doc_id, result["grade"]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(task, t) for t in pairs]
        total = len(futures)
        for fut in as_completed(futures):
            qid, doc_id, grade = fut.result()
            qrels.setdefault(qid, {})[doc_id] = grade
            completed += 1
            if completed % save_every == 0:
                with _cache_lock:
                    _save_cache()
                if progress:
                    print(f"    judge {completed}/{total}", end="\r")
        if progress:
            print(f"    judge {completed}/{total}")

    with _cache_lock:
        _save_cache()
    return qrels


def merge_qrels(target: dict, addition: dict) -> dict:
    """Merge two qrels dicts; if both have a (qid, doc_id), take max grade."""
    for qid, dd in addition.items():
        for did, g in dd.items():
            if qid in target and did in target[qid]:
                target[qid][did] = max(target[qid][did], g)
            else:
                target.setdefault(qid, {})[did] = g
    return target


def save_qrels(qrels: dict, path: str | pathlib.Path) -> None:
    pathlib.Path(path).write_text(json.dumps(qrels, indent=2), encoding="utf-8")


def load_qrels(path: str | pathlib.Path) -> dict:
    p = pathlib.Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))
