"""Records: plain data, no behaviour."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from evokit.util import new_id


@dataclass(frozen=True)
class Evaluation:
    """The result of evaluating an individual. No objectives means the evaluation failed."""

    objectives: dict[str, float] = field(default_factory=dict)
    feedback: str = ""
    usage: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"objectives": dict(self.objectives), "feedback": self.feedback, "usage": dict(self.usage)}

    @classmethod
    def from_dict(cls, data: Mapping) -> "Evaluation":
        return cls(dict(data.get("objectives") or {}), data.get("feedback") or "", dict(data.get("usage") or {}))

    @classmethod
    def from_metrics(cls, metrics: Mapping[str, Any], feedback: str = "") -> "Evaluation":
        """From a metrics dict, e.g. a script's JSON output: numbers are objectives; "feedback" and
        "usage" pass through."""
        objectives = {
            k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        usage = merge_usage(metrics.get("usage")) if isinstance(metrics.get("usage"), Mapping) else {}
        text = metrics.get("feedback")
        return cls(objectives, "\n\n".join(t for t in (str(text) if text else "", feedback) if t), usage)


@dataclass(frozen=True)
class Individual:
    id: str
    genome: Any
    parents: tuple[str, ...] = ()
    info: Mapping[str, Any] = field(default_factory=dict)
    evaluation: Evaluation | None = None

    @classmethod
    def new(cls, genome: Any, parents: Sequence[str] = (), info: Mapping[str, Any] | None = None) -> "Individual":
        """A fresh, unevaluated individual."""
        return cls(new_id(), genome, tuple(parents), dict(info or {}))

    def with_evaluation(self, evaluation: Evaluation) -> "Individual":
        return replace(self, evaluation=evaluation)

    def with_info(self, **info: Any) -> "Individual":
        return replace(self, info={**self.info, **info})

    def objective(self, name: str) -> float | None:
        """The value of one objective, or None if it is unevaluated, failed, or has no such objective."""
        return None if self.evaluation is None else self.evaluation.objectives.get(name)


@dataclass(frozen=True)
class Selection:
    """What a strategy picks for one child-to-be."""

    parents: tuple[Individual, ...]
    inspirations: tuple[Individual, ...] = ()


@dataclass(frozen=True)
class Offspring:
    """What an operator returns."""

    children: tuple[Individual, ...]
    usage: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Failure:
    """An operator call that produced nothing."""

    selection: Selection
    reason: str
    usage: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Asked:
    individuals: tuple[Individual, ...]
    failures: tuple[Failure, ...] = ()
    exhausted: bool = False


def merge_usage(*usages: Mapping[str, Any] | None) -> dict[str, float]:
    total: dict[str, float] = {}
    for usage in usages:
        for key, value in (usage or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                total[key] = total.get(key, 0) + value
    return total
