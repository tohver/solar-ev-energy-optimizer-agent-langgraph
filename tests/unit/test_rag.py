"""Knowledge base, index building and retrieval (regressions F-03, F-06, F-11). Offline: fake embeddings."""

from types import SimpleNamespace

import pytest

from energy_advisor.rag.benchmark import rank_of, score
from energy_advisor.rag.ingest import IndexExistsError, build_index, load_documents
from energy_advisor.rag.knowledge_base import IndexNotBuiltError, KnowledgeBase
from energy_advisor.rag.rerank import LLMReranker, apply_scores
from energy_advisor.rag.retrieval import rrf_fuse, tokenize

REQUIRED_TOPICS = ["hvac", "smart_home", "renewable", "seasonal", "storage"]


# ----------------------------------------------------------------------------- documents
def test_knowledge_base_has_at_least_seven_documents(settings):
    assert len(load_documents(settings.paths.knowledge_base)) >= 7


@pytest.mark.parametrize("topic", REQUIRED_TOPICS)
def test_knowledge_base_covers_required_topic(settings, topic):
    assert any(topic in p.name for p in settings.paths.knowledge_base.glob("*.txt"))


def test_missing_documents_folder_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path)


# ----------------------------------------------------------------------------- index
def _sources_and_texts(kb: KnowledgeBase) -> tuple[set[str], list[str]]:
    _, chunks = kb._bm25_index()
    return {c["source"] for c in chunks}, [c["content"] for c in chunks]


def test_f03_every_document_is_indexed(settings, knowledge_base):
    # F-03: the starter indexed only 2 hard-coded files.
    sources, _ = _sources_and_texts(knowledge_base)
    assert sources == {p.name for p in settings.paths.knowledge_base.glob("*.txt")}


def test_f11_rebuild_does_not_duplicate_chunks(settings, knowledge_base, fake_embeddings):
    # F-11: rebuilding used to append a second copy of every chunk.
    result = build_index(
        settings.paths.knowledge_base, settings.paths.vectorstore, fake_embeddings, settings.rag, rebuild=True
    )
    knowledge_base.reset()
    _, texts = _sources_and_texts(knowledge_base)
    assert len(texts) == len(set(texts)) == result.chunks


def test_build_without_rebuild_refuses_to_overwrite(settings, knowledge_base, fake_embeddings):
    with pytest.raises(IndexExistsError):
        build_index(settings.paths.knowledge_base, settings.paths.vectorstore, fake_embeddings, settings.rag)


def test_search_before_build_explains_what_to_do(settings, fake_embeddings):
    kb = KnowledgeBase(settings.paths.vectorstore, settings.rag, lambda: fake_embeddings)
    assert not kb.is_built()
    with pytest.raises(IndexNotBuiltError, match="build-index"):
        kb.search("anything")


# ----------------------------------------------------------------------------- retrieval
def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("EV-charging, 11 kW!") == ["ev", "charging", "11", "kw"]


def test_rrf_rewards_passages_found_by_both_methods():
    a, b, c = ({"content": x, "source": f"{x}.txt"} for x in "abc")
    fused = rrf_fuse([a, b], [c, b], rrf_k=60)
    assert fused[0]["content"] == "b"  # rank 2 in both lists beats rank 1 in only one
    assert fused[0]["vector_rank"] == 2
    assert fused[0]["bm25_rank"] == 2
    assert fused[0]["rrf"] == pytest.approx(2 / 62)


def test_bm25_finds_exact_terms(knowledge_base):
    hits = knowledge_base.bm25_search("pool pump", k=3)
    assert hits
    assert all("pool" in h["content"].lower() or "pump" in h["content"].lower() for h in hits)


def test_search_falls_back_to_fused_order_without_reranker(knowledge_base):
    result = knowledge_base.search("When should I run my pool pump?", max_results=3)
    assert result["retrieval_method"].endswith("re-ranking unavailable")
    assert 1 <= result["total_results"] <= 3
    assert all(t["source"].endswith(".txt") for t in result["tips"])
    assert [t["rank"] for t in result["tips"]] == list(range(1, result["total_results"] + 1))


def test_search_uses_reranker_scores(settings, knowledge_base, fake_embeddings):
    class TopScoreForSecond:
        def invoke(self, _prompt):
            return SimpleNamespace(scores=[SimpleNamespace(index=1, score=9)])

    kb = KnowledgeBase(
        settings.paths.vectorstore,
        settings.rag,
        lambda: fake_embeddings,
        lambda: LLMReranker(TopScoreForSecond()),
    )
    result = kb.search("heat pump", max_results=2)
    assert result["retrieval_method"].endswith("LLM re-ranking")
    assert result["tips"][0]["rerank_score"] == 9
    assert result["tips"][0]["relevance_score"] == "high"


# ----------------------------------------------------------------------------- re-ranker
def test_f06_reranker_tolerates_missing_scores():
    # F-06: the LLM sometimes returned fewer scores than passages, which raised an error.
    passages = [{"content": f"p{i}", "source": "x.txt", "rrf": 1 / (60 + i)} for i in range(4)]
    ranked = apply_scores(passages, {2: 9, 0: 5}, k=4, min_score=3)
    assert [p["content"] for p in ranked[:2]] == ["p2", "p0"]
    assert {p["content"] for p in ranked[2:]} == {"p1", "p3"}


def test_reranker_drops_irrelevant_passages_but_keeps_one():
    passages = [{"content": f"p{i}", "source": "x.txt", "rrf": 0.0} for i in range(3)]
    assert [p["content"] for p in apply_scores(passages, {0: 1, 1: 2, 2: 0}, k=3, min_score=3)] == ["p1"]


def test_reranker_scores_are_clamped():
    passages = [{"content": "p", "source": "x.txt"}]
    assert apply_scores(passages, {0: 42}, k=1, min_score=3)[0]["rerank_score"] == 10


def test_reranker_without_scores_raises():
    class Empty:
        def invoke(self, _prompt):
            return SimpleNamespace(scores=[])

    with pytest.raises(ValueError, match="no scores"):
        LLMReranker(Empty()).rerank("q", [{"content": "p", "source": "x.txt"}], k=1)


# ----------------------------------------------------------------------------- benchmark maths
def test_rank_of_and_scores():
    results = [{"source": "a.txt"}, {"source": "b.txt"}]
    assert rank_of(results, "b.txt") == 2
    assert rank_of(results, "c.txt") is None
    assert score([1, 2, None, 1]) == (0.75, 0.62)
