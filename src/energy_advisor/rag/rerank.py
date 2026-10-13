"""LLM re-ranking: a chat model scores each candidate passage 0-10 for relevance to the question."""

from typing import Any, Protocol

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from energy_advisor.rag.retrieval import Passage

PASSAGE_PREVIEW_CHARS = 700


class PassageScore(BaseModel):
    index: int = Field(description="Passage number shown in square brackets")
    score: int = Field(description="Relevance 0-10")


class RerankScores(BaseModel):
    scores: list[PassageScore] = Field(description="One entry per passage")


class Scorer(Protocol):
    """Anything with ``invoke(prompt)``: a structured-output chat model or a test fake."""

    def invoke(self, input: Any, /) -> Any: ...


def build_prompt(query: str, passages: list[Passage]) -> str:
    numbered = "\n\n".join(f"[{i}] {p['content'][:PASSAGE_PREVIEW_CHARS]}" for i, p in enumerate(passages))
    return (
        "Rate how useful each passage is for answering the home-energy question. "
        "10 = directly answers it with concrete advice, 0 = unrelated.\n"
        f"QUESTION: {query}\n\nPASSAGES:\n{numbered}\n\n"
        f"Score every passage [0]..[{len(passages) - 1}]."
    )


def apply_scores(passages: list[Passage], scores: dict[int, int], k: int, min_score: int) -> list[Passage]:
    """Order passages by LLM score; drop the irrelevant ones (keep at least one).

    The LLM sometimes skips passages (regression F-06): a skipped passage keeps its fused position
    but ranks below every scored passage.
    """
    for i, p in enumerate(passages):
        p["rerank_score"] = max(0, min(10, int(scores[i]))) if i in scores else None
    ranked = sorted(
        passages,
        key=lambda p: (p["rerank_score"] if p["rerank_score"] is not None else -1, p.get("rrf", 0)),
        reverse=True,
    )
    relevant = [p for p in ranked if p["rerank_score"] is None or p["rerank_score"] >= min_score] or ranked[
        :1
    ]
    return relevant[:k]


class LLMReranker:
    def __init__(self, scorer: Scorer, min_score: int = 3) -> None:
        self._scorer = scorer
        self._min_score = min_score

    @classmethod
    def from_llm(cls, llm: BaseChatModel, min_score: int = 3) -> "LLMReranker":
        return cls(llm.with_structured_output(RerankScores, method="function_calling"), min_score)

    def rerank(self, query: str, passages: list[Passage], k: int) -> list[Passage]:
        if not passages:
            return passages
        result = self._scorer.invoke(build_prompt(query, passages))
        scores = {s.index: s.score for s in result.scores}
        if not scores:
            raise ValueError("Re-ranker returned no scores")
        return apply_scores(passages, scores, k, self._min_score)
