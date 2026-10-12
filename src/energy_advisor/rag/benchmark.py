"""Compare retrieval variants on labelled queries: hit@k and MRR (mean reciprocal rank)."""

from dataclasses import dataclass
from typing import Any

from energy_advisor.rag.knowledge_base import KnowledgeBase
from energy_advisor.rag.retrieval import Passage

# Each query is labelled with the document that best answers it.
BENCHMARK_QUERIES: list[tuple[str, str]] = [
    ("When should I run my dishwasher with a dynamic tariff?", "tip_device_best_practices.txt"),
    ("How do I keep the house warm during an evening price peak?", "tip_hvac_optimization.txt"),
    ("Can my electric car power the house?", "tip_energy_storage_optimization.txt"),
    ("Should I export my solar power or use it myself?", "tip_renewable_integration.txt"),
    (
        "How much solar power does an 8 kWp system produce in December in Munich?",
        "tip_seasonal_energy_management.txt",
    ),
    ("How do smart plugs reduce standby power?", "tip_smart_home_automation.txt"),
    ("How much CO2 does an 8 kWp solar system avoid?", "tip_renewable_integration.txt"),
    ("Is a home battery worth it with a dynamic tariff?", "tip_energy_storage_optimization.txt"),
    ("Dishwasher eco programme and full loads", "tip_device_best_practices.txt"),
    ("Cheapest time to charge an EV overnight", "tip_ev_charging_and_tariffs.txt"),
    ("What flow temperature should my heat pump use?", "tip_hvac_optimization.txt"),
    ("What does section 14a mean for my wallbox?", "tip_ev_charging_and_tariffs.txt"),
]
METHODS = ("vector", "hybrid", "hybrid_rerank")


@dataclass(frozen=True)
class MethodScore:
    method: str
    hit_at_k: float
    mrr: float


def rank_of(results: list[Passage], expected_source: str) -> int | None:
    """1-based position of the first result from the expected document (None if absent)."""
    return next((i for i, r in enumerate(results, start=1) if r["source"] == expected_source), None)


def score(ranks: list[int | None]) -> tuple[float, float]:
    hit = sum(r is not None for r in ranks) / len(ranks)
    mrr = sum(1 / r for r in ranks if r is not None) / len(ranks)
    return round(hit, 2), round(mrr, 2)


def run_benchmark(
    kb: KnowledgeBase, k: int = 3, queries: list[tuple[str, str]] | None = None
) -> tuple[list[dict[str, Any]], list[MethodScore]]:
    queries = queries or BENCHMARK_QUERIES
    rows: list[dict[str, Any]] = []
    for query, expected in queries:
        rows.append(
            {
                "query": query,
                "expected": expected,
                "vector": rank_of(kb.vector_search(query, k=k), expected),
                "hybrid": rank_of(kb.hybrid_search(query, k=k), expected),
                "hybrid_rerank": rank_of(kb.reranked_search(query, k=k), expected),
            }
        )
    summary = [MethodScore(m, *score([row[m] for row in rows])) for m in METHODS]
    return rows, summary
