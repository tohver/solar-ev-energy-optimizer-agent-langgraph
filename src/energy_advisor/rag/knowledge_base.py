"""Search over the energy-saving knowledge base: hybrid retrieval followed by LLM re-ranking."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi

from energy_advisor.config import RAGSettings
from energy_advisor.rag.ingest import collection_exists, make_chroma_client
from energy_advisor.rag.rerank import LLMReranker
from energy_advisor.rag.retrieval import Passage, rrf_fuse, tokenize

logger = logging.getLogger(__name__)


class IndexNotBuiltError(RuntimeError):
    pass


class KnowledgeBase:
    """Owns the vector store and the BM25 index. Both are loaded lazily on the first search.

    Vectors catch paraphrases ("cut cooling bill" ~ "HVAC savings"); BM25 catches exact terms
    ("pool pump", "V2H") that embeddings can blur. Reciprocal Rank Fusion merges both rankings.
    """

    def __init__(
        self,
        persist_dir: Path,
        rag: RAGSettings,
        embeddings_factory: Callable[[], Embeddings],
        reranker_factory: Callable[[], LLMReranker] | None = None,
    ) -> None:
        self._persist_dir = persist_dir
        self._rag = rag
        self._embeddings_factory = embeddings_factory
        self._reranker_factory = reranker_factory
        self._store: Chroma | None = None
        self._bm25: BM25Okapi | None = None
        self._chunks: list[Passage] = []
        self._reranker: LLMReranker | None = None

    # ------------------------------------------------------------------ loading
    def is_built(self) -> bool:
        if not self._persist_dir.exists():
            return False
        return collection_exists(make_chroma_client(self._persist_dir), self._rag.collection_name)

    def reset(self) -> None:
        """Forget loaded indexes (call after the index was rebuilt in the same process)."""
        self._store, self._bm25, self._chunks = None, None, []

    def _vectorstore(self) -> Chroma:
        if self._store is None:
            if not self.is_built():
                raise IndexNotBuiltError(
                    "Knowledge base index not found. Run `energy-advisor build-index` first."
                )
            self._store = Chroma(
                client=make_chroma_client(self._persist_dir),
                collection_name=self._rag.collection_name,
                embedding_function=self._embeddings_factory(),
            )
        return self._store

    def _bm25_index(self) -> tuple[BM25Okapi, list[Passage]]:
        if self._bm25 is None:
            stored = self._vectorstore().get(include=["documents", "metadatas"])
            self._chunks = [
                {"content": doc, "source": (meta or {}).get("source", "unknown")}
                for doc, meta in zip(stored["documents"], stored["metadatas"], strict=True)
            ]
            self._bm25 = BM25Okapi([tokenize(c["content"]) for c in self._chunks])
        return self._bm25, self._chunks

    # ------------------------------------------------------------------ search
    def vector_search(self, query: str, k: int = 5) -> list[Passage]:
        """Plain semantic search (the baseline)."""
        return [
            {
                "content": d.page_content,
                "source": d.metadata.get("source", "unknown"),
                "distance": round(float(s), 4),
            }
            for d, s in self._vectorstore().similarity_search_with_score(query, k=k)
        ]

    def bm25_search(self, query: str, k: int = 10) -> list[Passage]:
        bm25, chunks = self._bm25_index()
        scores = bm25.get_scores(tokenize(query))
        top = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:k]
        return [chunks[i] for i in top if scores[i] > 0]  # no keyword overlap -> not a hit

    def hybrid_search(self, query: str, k: int = 5) -> list[Passage]:
        n = self._rag.candidates
        fused = rrf_fuse(self.vector_search(query, k=n), self.bm25_search(query, k=n), self._rag.rrf_k)
        return fused[:k]

    def reranked_search(self, query: str, k: int = 5) -> list[Passage]:
        if self._reranker is None:
            if self._reranker_factory is None:
                raise RuntimeError("No re-ranker configured")
            self._reranker = self._reranker_factory()
        return self._reranker.rerank(query, self.hybrid_search(query, k=max(8, k)), k=k)

    def search(self, query: str, max_results: int = 5) -> dict[str, Any]:
        """Tool output: ranked tips with source file and scores. Falls back to the fused order."""
        max_results = max(1, min(int(max_results), 10))
        method = "hybrid (vector + BM25, RRF) + LLM re-ranking"
        try:
            ranked = self.reranked_search(query, k=max_results)
        except IndexNotBuiltError:
            raise
        except Exception as exc:  # the re-ranker is an optional quality boost
            logger.warning("Re-ranking failed, using fused order: %s", exc)
            ranked = self.hybrid_search(query, k=max_results)
            method = "hybrid (vector + BM25, RRF); re-ranking unavailable"

        tips = []
        for i, item in enumerate(ranked):
            score = item.get("rerank_score")
            if score is not None:
                relevance = "high" if score >= 7 else "medium" if score >= 4 else "low"
            else:
                relevance = "high" if i < 2 else "medium" if i < 4 else "low"
            tips.append(
                {
                    "rank": i + 1,
                    "content": item["content"],
                    "source": item["source"],
                    "rerank_score": score,
                    "vector_rank": item.get("vector_rank"),
                    "bm25_rank": item.get("bm25_rank"),
                    "relevance_score": relevance,
                }
            )
        return {"query": query, "retrieval_method": method, "total_results": len(tips), "tips": tips}
