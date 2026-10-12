"""Retrieval building blocks: tokenization for BM25 and Reciprocal Rank Fusion."""

import re
from collections.abc import Sequence
from typing import Any

Passage = dict[str, Any]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def rrf_fuse(vector_hits: Sequence[Passage], bm25_hits: Sequence[Passage], rrf_k: int = 60) -> list[Passage]:
    """Merge two rankings with Reciprocal Rank Fusion: each passage scores ``sum(1 / (rrf_k + rank))``.

    Passages are identified by their text. Each hit needs ``content`` and ``source``; the result
    also records ``vector_rank`` and ``bm25_rank`` so the agent's output can show where a tip came from.
    """
    fused: dict[str, Passage] = {}
    for rank_key, hits in (("vector_rank", vector_hits), ("bm25_rank", bm25_hits)):
        for rank, hit in enumerate(hits, start=1):
            item = fused.setdefault(
                hit["content"], {"content": hit["content"], "source": hit["source"], "rrf": 0.0}
            )
            item["rrf"] += 1 / (rrf_k + rank)
            item[rank_key] = rank
    return sorted(fused.values(), key=lambda p: p["rrf"], reverse=True)
