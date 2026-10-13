"""Evaluation cases: loaded from YAML so new questions need no code changes."""

from importlib import resources
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    id: str
    question: str
    expected_tools: list[str] = Field(default_factory=list)
    expected_response: str


def load_cases(path: Path | None = None) -> list[EvalCase]:
    """Cases from ``path`` or from the bundled ``cases.yaml``."""
    if path is None:
        text = resources.files("energy_advisor.evaluation").joinpath("cases.yaml").read_text(encoding="utf-8")
    else:
        text = path.read_text(encoding="utf-8")
    cases = [EvalCase.model_validate(item) for item in yaml.safe_load(text)]
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Evaluation case ids must be unique")
    return cases
