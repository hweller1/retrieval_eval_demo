"""
SEC 10-K corpus loader.

Downloads recent 10-K filings for a curated set of US tech companies from
SEC EDGAR, parses the HTML to plain text, and exposes them in the same
shape as our BEIR datasets:

  load() → (corpus, queries, qrels, info)

Where:
  corpus   = {doc_id: {"title": str, "text": str}}
  queries  = {qid: str}    (loaded from sec_queries.json)
  qrels    = {}            (no human qrels — use llm_judge.py)
  info     = {"description": ..., "split": "test"}

Each doc_id is "<TICKER>_<FY>" (e.g. "AAPL_2023"). Cached to disk under
~/.cache/voyage-demos/sec_10k/ so re-runs don't hit EDGAR again.

EDGAR usage policy:
  https://www.sec.gov/os/accessing-edgar-data
  - User-Agent must identify the client (we set one with a contact email).
  - Rate limit: ≤10 requests/sec; we sleep 0.15s between calls to be safe.
"""

from __future__ import annotations

import os
import re
import json
import time
import pathlib
import requests
from dataclasses import dataclass

import warnings
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


# ── Companies ────────────────────────────────────────────────────────────────
# (ticker, CIK, display name). 15 US tech companies with mixed business
# models: mega-cap mature, growth, semiconductors, SaaS.

COMPANIES: list[tuple[str, str, str]] = [
    ("AAPL",  "0000320193", "Apple Inc."),
    ("MSFT",  "0000789019", "Microsoft Corp."),
    ("GOOGL", "0001652044", "Alphabet Inc."),
    ("AMZN",  "0001018724", "Amazon.com Inc."),
    ("META",  "0001326801", "Meta Platforms Inc."),
    ("NVDA",  "0001045810", "NVIDIA Corp."),
    ("TSLA",  "0001318605", "Tesla Inc."),
    ("NFLX",  "0001065280", "Netflix Inc."),
    ("ORCL",  "0001341439", "Oracle Corp."),
    ("CRM",   "0001108524", "Salesforce Inc."),
    ("AMD",   "0000002488", "Advanced Micro Devices Inc."),
    ("ADBE",  "0000796343", "Adobe Inc."),
    ("INTC",  "0000050863", "Intel Corp."),
    ("MDB",   "0001441816", "MongoDB Inc."),
    ("SNOW",  "0001640147", "Snowflake Inc."),
]


# ── Config ───────────────────────────────────────────────────────────────────

CACHE_DIR = pathlib.Path.home() / ".cache" / "voyage-demos" / "sec_10k"
USER_AGENT = "voyage-demos henry.weller@mongodb.com"
HEADERS = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}
RATE_SLEEP = 0.15  # seconds between EDGAR HTTP calls

# Filing year range: 10-Ks filed in calendar years 2022–2025 cover fiscal
# years 2021–2025 depending on fiscal-year-end month.
MIN_FILING_YEAR = 2022
MAX_FILING_YEAR = 2025

# Soft cap on full-text length per filing. 10-Ks can hit 1M+ chars when
# you include the financial statement exhibits; we keep the narrative
# (Items 1, 1A, 7) which is what trader queries care about.
MAX_FILING_CHARS = 350_000


# ── Disk cache ───────────────────────────────────────────────────────────────

def _cache_path(name: str) -> pathlib.Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / name


def _read_cache(name: str) -> str | None:
    p = _cache_path(name)
    if p.exists():
        return p.read_text(encoding="utf-8")
    return None


def _write_cache(name: str, content: str) -> None:
    _cache_path(name).write_text(content, encoding="utf-8")


# ── EDGAR fetch ──────────────────────────────────────────────────────────────

def _http_get(url: str) -> requests.Response:
    """GET with the SEC-required User-Agent and a polite delay."""
    time.sleep(RATE_SLEEP)
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r


def _list_10k_filings(cik: str) -> list[dict]:
    """
    Returns recent 10-K filings for `cik` as
      [{"accession": str, "filing_date": str, "primary_doc": str, "fiscal_year": int}, ...]

    `fiscal_year` here is approximated from the filing date — a 10-K filed
    in 2024 most likely covers FY2023 (or part of FY2024 for off-cycle
    fiscals like Apple). Good enough for our doc_id labelling.
    """
    cache_name = f"submissions_{cik}.json"
    cached = _read_cache(cache_name)
    if cached:
        data = json.loads(cached)
    else:
        url  = f"https://data.sec.gov/submissions/CIK{cik}.json"
        data = _http_get(url).json()
        _write_cache(cache_name, json.dumps(data))

    recent = data["filings"]["recent"]
    filings: list[dict] = []
    for i, form in enumerate(recent["form"]):
        if form != "10-K":
            continue
        filing_date = recent["filingDate"][i]
        year = int(filing_date.split("-")[0])
        if year < MIN_FILING_YEAR or year > MAX_FILING_YEAR:
            continue
        filings.append({
            "accession" : recent["accessionNumber"][i],
            "filing_date": filing_date,
            "primary_doc": recent["primaryDocument"][i],
            # 10-K filings are typically for the previous fiscal year:
            "fiscal_year": year - 1 if int(filing_date.split("-")[1]) <= 6 else year,
        })
    return filings


def _fetch_filing_html(cik: str, accession: str, primary_doc: str) -> str:
    cache_name = f"{cik}_{accession.replace('-', '')}_{primary_doc}".replace("/", "_")
    cached = _read_cache(cache_name)
    if cached:
        return cached
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/{primary_doc}"
    html = _http_get(url).text
    _write_cache(cache_name, html)
    return html


# ── HTML → text ──────────────────────────────────────────────────────────────

# Section markers we DO want to keep (Items 1, 1A, 7 are the narrative
# parts trader queries care about). The boundary patterns are loose
# because 10-K formatting varies wildly across companies.
SECTION_START_PATTERNS = [
    re.compile(r"\bItem\s*1\b\.?\s*[A-Z]"),      # Business
    re.compile(r"\bItem\s*1A\b\.?\s*Risk"),      # Risk Factors
    re.compile(r"\bItem\s*7\b\.?\s*Management", re.IGNORECASE),  # MD&A
]
SECTION_END_PATTERNS = [
    re.compile(r"\bItem\s*8\b\.?\s*Financial"),
    re.compile(r"\bItem\s*9\b"),
    re.compile(r"PART\s*III", re.IGNORECASE),
]


def _html_to_text(html: str) -> str:
    """Aggressive HTML→text. Strips XBRL, scripts, styles, tables of EDGAR
    boilerplate, and collapses whitespace. Retains paragraph breaks."""
    soup = BeautifulSoup(html, "lxml")

    # Remove XBRL/iXBRL hidden facts and other non-content
    for tag in soup(["script", "style", "head", "meta", "link"]):
        tag.decompose()
    for tag in soup.find_all(attrs={"style": re.compile(r"display\s*:\s*none", re.I)}):
        tag.decompose()
    for tag in soup.find_all(["ix:hidden"]):
        tag.decompose()

    # Get text with double newlines between blocks
    text = soup.get_text(separator="\n", strip=True)

    # Collapse 3+ newlines to 2; collapse runs of spaces
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _extract_narrative(text: str) -> str:
    """Try to slice out Items 1, 1A, 7 using regex anchors. Falls back
    to the whole document if anchors aren't found cleanly."""
    text_lower_pos = text.lower()

    # Find the first occurrence of "Item 1" that looks like a section header
    # (skip the table-of-contents copy at the very top by finding the second
    # occurrence)
    item1_starts = [m.start() for m in re.finditer(r"\bitem\s*1\b\.?\s*business",
                                                    text_lower_pos)]
    item8_starts = [m.start() for m in re.finditer(r"\bitem\s*8\b\.?\s*financial",
                                                    text_lower_pos)]
    if len(item1_starts) >= 2 and item8_starts:
        start = item1_starts[1]
        end   = next((s for s in item8_starts if s > start), None) or len(text)
        candidate = text[start:end].strip()
        if len(candidate) >= 5_000:
            return candidate

    # Fallback: trim front matter (cover page, signatures, table of contents)
    return text


# ── Top-level loader ────────────────────────────────────────────────────────

@dataclass
class FilingDoc:
    doc_id: str        # "AAPL_2023"
    ticker: str
    company_name: str
    fiscal_year: int
    filing_date: str
    title: str
    text: str
    char_count: int


def fetch_corpus(verbose: bool = False) -> list[FilingDoc]:
    """Fetch + parse all configured 10-Ks (cached). Returns FilingDoc list."""
    docs: list[FilingDoc] = []
    for ticker, cik, name in COMPANIES:
        try:
            filings = _list_10k_filings(cik)
        except Exception as e:
            if verbose:
                print(f"  [{ticker}] list failed: {e}")
            continue

        for f in filings:
            doc_id = f"{ticker}_{f['fiscal_year']}"
            if verbose:
                print(f"  [{ticker}] FY{f['fiscal_year']}  {f['filing_date']}  {f['primary_doc']}")
            try:
                html = _fetch_filing_html(cik, f["accession"], f["primary_doc"])
            except Exception as e:
                if verbose:
                    print(f"    fetch failed: {e}")
                continue
            text = _html_to_text(html)
            text = _extract_narrative(text)
            if len(text) > MAX_FILING_CHARS:
                text = text[:MAX_FILING_CHARS]
            title = f"{name} 10-K (Fiscal Year {f['fiscal_year']})"
            docs.append(FilingDoc(
                doc_id=doc_id, ticker=ticker, company_name=name,
                fiscal_year=f["fiscal_year"], filing_date=f["filing_date"],
                title=title, text=text, char_count=len(text),
            ))
    return docs


def load(verbose: bool = False):
    """BEIR-compatible load() — returns (corpus, queries, qrels, info)."""
    docs = fetch_corpus(verbose=verbose)

    # corpus shape mirrors BEIR
    corpus = {
        d.doc_id: {"title": d.title, "text": d.text}
        for d in docs
    }

    # queries from a committed JSON file (generated separately)
    queries_path = pathlib.Path(__file__).parent / "sec_queries.json"
    if queries_path.exists():
        queries = json.loads(queries_path.read_text(encoding="utf-8"))
    else:
        queries = {}

    qrels = {}  # no human qrels — populated by llm_judge.py at evaluation time

    info = {
        "description": "SEC 10-K filings for 15 US tech companies (FY2021–2024)",
        "split": "test",
        "size_docs": len(corpus),
        "size_queries": len(queries),
    }
    return corpus, queries, qrels, info


# ── CLI for one-off corpus inspection ───────────────────────────────────────

if __name__ == "__main__":
    docs = fetch_corpus(verbose=True)
    print()
    print(f"  Total docs: {len(docs)}")
    if docs:
        chars = sorted(d.char_count for d in docs)
        print(f"  Char count: min={chars[0]:,}  median={chars[len(chars)//2]:,}  max={chars[-1]:,}")
        print(f"  Sample titles:")
        for d in docs[:5]:
            print(f"    {d.doc_id:<14} {d.char_count:>9,} chars  {d.title}")
