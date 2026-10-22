"""Prompt pieces shared by operators that talk to models and agents."""

from __future__ import annotations

import json
import textwrap

from evokit.records import Individual
from evokit.util import truncate


def describe_individual(individual: Individual, max_feedback: int = 2000) -> str:
    """Its id, evaluation, evaluator feedback, and how it was made (not its genome)."""
    evaluation = individual.evaluation
    lines = [f"Individual {individual.id}"]
    if evaluation is not None:
        lines.append(f"  objectives: {json.dumps(evaluation.objectives)}" if evaluation.objectives else "  evaluation failed")
        if evaluation.feedback:
            lines.append("  feedback:\n" + textwrap.indent(truncate(evaluation.feedback, max_feedback), "    "))
    summary = individual.info.get("summary") or individual.info.get("note")
    if summary:
        lines.append(f"  how it was made: {truncate(str(summary), 500)}")
    return "\n".join(lines)


def join_notes(*parts: str | None) -> str | None:
    text = "\n\n".join(p for p in parts if p)
    return text or None
