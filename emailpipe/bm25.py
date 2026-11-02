"""Okapi BM25 from scratch: tokenizer, light stemmer, inverted index, scoring.

Scoring (per query term t, document d):

    idf(t)      = ln(1 + (N - n_t + 0.5) / (n_t + 0.5))          # Lucene-style, never negative
    score(d, q) = sum_t idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))

Only the postings of query terms are touched, so cost scales with the number of
matching documents, not the size of the archive.
"""
from __future__ import annotations

import heapq
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from .models import Email

STOPWORDS = frozenset(
    """a an and are as at be but by for from has have he her his i if in into is it its me my of on or our
    she that the their them there they this to was we were what when where which who whom why will with you your""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
# longest suffix first; (suffix, replacement)
_SUFFIXES = (("ings", ""), ("ing", ""), ("edly", ""), ("sses", "ss"), ("ies", "y"), ("ed", ""), ("es", ""), ("s", ""), ("e", ""))


def stem(token: str) -> str:
    """Deliberately light suffix stripper (approve/approved/approves -> approv)."""
    for suffix, repl in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            if suffix == "s" and token.endswith("ss"):
                continue
            return token[: -len(suffix)] + repl
    return token


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


@dataclass(frozen=True)
class Hit:
    doc_id: str
    score: float


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError("require k1 >= 0 and 0 <= b <= 1")
        self.k1, self.b = k1, b
        self.doc_ids: list[str] = []
        self.doc_len: list[int] = []
        self.total_len = 0
        self.postings: dict[str, dict[int, int]] = {}  # term -> {internal doc index: tf}

    # ---- indexing -------------------------------------------------------
    def add(self, doc_id: str, text: str) -> None:
        tokens = tokenize(text)
        idx = len(self.doc_ids)
        self.doc_ids.append(doc_id)
        self.doc_len.append(len(tokens))
        self.total_len += len(tokens)
        counts: dict[str, int] = defaultdict(int)
        for t in tokens:
            counts[t] += 1
        for term, tf in counts.items():
            self.postings.setdefault(term, {})[idx] = tf

    def build(self, docs: Iterable[tuple[str, str]]) -> "BM25Index":
        for doc_id, text in docs:
            self.add(doc_id, text)
        return self

    # ---- scoring --------------------------------------------------------
    @property
    def n_docs(self) -> int:
        return len(self.doc_ids)

    def idf(self, term: str) -> float:
        n = len(self.postings.get(term, ()))
        return math.log(1.0 + (self.n_docs - n + 0.5) / (n + 0.5))

    def _scores(self, query: str) -> dict[int, float]:
        if not self.n_docs:
            return {}
        avgdl = (self.total_len / self.n_docs) or 1.0
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = self.idf(term)
            for d, tf in plist.items():
                norm = tf + self.k1 * (1.0 - self.b + self.b * self.doc_len[d] / avgdl)
                scores[d] += idf * tf * (self.k1 + 1.0) / norm
        return scores

    def search(self, query: str, k: int = 5) -> list[Hit]:
        scores = self._scores(query)
        # ties broken by insertion order so results are deterministic
        top = heapq.nlargest(k, scores.items(), key=lambda kv: (kv[1], -kv[0]))
        return [Hit(self.doc_ids[d], s) for d, s in top]

    # ---- persistence ----------------------------------------------------
    def save(self, path: str | Path) -> None:
        payload = {
            "k1": self.k1, "b": self.b, "doc_ids": self.doc_ids, "doc_len": self.doc_len,
            "postings": {t: {str(d): tf for d, tf in p.items()} for t, p in self.postings.items()},
        }
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        p = json.loads(Path(path).read_text(encoding="utf-8"))
        idx = cls(p["k1"], p["b"])
        idx.doc_ids, idx.doc_len = p["doc_ids"], p["doc_len"]
        idx.total_len = sum(idx.doc_len)
        idx.postings = {t: {int(d): tf for d, tf in pl.items()} for t, pl in p["postings"].items()}
        return idx


class EmailSearch:
    """BM25 over an email archive. Subject is repeated to up-weight it; sender name is searchable."""

    def __init__(self, emails: Iterable[Email], k1: float = 1.5, b: float = 0.75):
        self.emails = {e.id: e for e in emails}
        self.index = BM25Index(k1, b)
        for e in self.emails.values():
            self.index.add(e.id, f"{e.subject} {e.subject} {e.sender_name} {e.body}")

    def search(self, query: str, k: int = 5) -> list[tuple[Email, float]]:
        return [(self.emails[h.doc_id], h.score) for h in self.index.search(query, k)]

    def get(self, email_id: str) -> Optional[Email]:
        return self.emails.get(email_id)
