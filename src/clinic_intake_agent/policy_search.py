"""Small BM25 search over the clinic's policy docs.

No embeddings or vector DB: the corpus is five short documents, and lexical
search gives exact, explainable citations with zero extra infrastructure or API
keys. Swap for a vector store when the corpus grows.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from clinic_intake_agent.data import parse_sections, policy_dir

_STOP = set(
    "a an and are as at be by can do does for from have how i if in is it my of on or our the "
    "to we what when where which who will with you your".split()
)


def _tokens(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9$]+", text.lower())
    return [t.rstrip("s") if len(t) > 3 else t for t in toks if t not in _STOP]


@dataclass(frozen=True)
class PolicyChunk:
    source: str
    section: str
    text: str

    @property
    def citation(self) -> str:
        return f"{self.source} > {self.section}"


@lru_cache(maxsize=1)
def load_chunks() -> tuple[PolicyChunk, ...]:
    chunks = []
    for path in sorted(policy_dir().glob("*.md")):
        for heading, body in parse_sections(path.read_text(encoding="utf-8")).items():
            if heading == "_preamble" or not body:
                continue
            chunks.append(PolicyChunk(source=path.name, section=heading, text=body))
    return tuple(chunks)


def search(query: str, k: int = 3, *, k1: float = 1.5, b: float = 0.75) -> list[tuple[PolicyChunk, float]]:
    chunks = load_chunks()
    docs = [_tokens(f"{c.section} {c.section} {c.text}") for c in chunks]  # heading counted twice
    avgdl = sum(len(d) for d in docs) / max(len(docs), 1)
    df = Counter(t for d in docs for t in set(d))
    n = len(docs)
    q = _tokens(query)
    scored = []
    for chunk, doc in zip(chunks, docs):
        tf = Counter(doc)
        score = 0.0
        for t in q:
            if t not in tf:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            score += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(doc) / avgdl))
        if score > 0:
            scored.append((chunk, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]
