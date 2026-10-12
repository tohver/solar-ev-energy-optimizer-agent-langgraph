"""Compare retrieval variants on labelled queries: hit@k and MRR (mean reciprocal rank)."""

from dataclasses import dataclass
from typing import Any

from energy_advisor.rag.knowledge_base import KnowledgeBase
from energy_advisor.rag.retrieval import Passage

# Each query is labelled with the document that best answers it.
BENCHMARK_QUERIES: list[tuple[str, str]] = [
    ("When should I run my pool pump?", "tip_ev_charging_and_tariffs.txt"),
    ("How do I keep my house cool during an evening price spike?", "tip_hvac_optimization.txt"),
    ("Can my electric car power the house during an outage?", "tip_energy_storage_optimization.txt"),
    ("Should I export my solar power or use it myself?", "tip_renewable_integration.txt"),
    ("What thermostat temperature for winter?", "tip_seasonal_energy_management.txt"),
    ("How do smart plugs reduce standby power?", "tip_smart_home_automation.txt"),
    ("How much CO2 does a 6 kW solar system avoid?", "tip_renewable_integration.txt"),
    ("Battery arbitrage payback period", "tip_energy_storage_optimization.txt"),
    ("Dishwasher eco mode and full loads", "tip_device_best_practices.txt"),
    ("Cheapest time to charge an EV overnight", "tip_ev_charging_and_tariffs.txt"),
    ("How often should I replace the HVAC filter?", "tip_hvac_optimization.txt"),
    ("Is there a demand charge during peak hours?", "tip_energy_storage_optimization.txt"),
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
