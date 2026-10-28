"""Prompt pieces shared by operators that talk to models and agents."""

from __future__ import annotations

import json
import textwrap

from evokit.records import Individual
from evokit.util import truncate


def describe_individual(individual: Individual, max_feedback: int = 2000) -> str:
    """Its id and name, evaluation, evaluator feedback, and how it was made (not its genome)."""
    evaluation = individual.evaluation
    lines = [f"Individual {individual.id}" + (f": {individual.name}" if individual.name else "")]
    if evaluation is not None:
        lines.append(f"  objectives: {json.dumps(evaluation.objectives)}" if evaluation.objectives else "  evaluation failed")
        if evaluation.feedback:
            lines.append("  feedback:\n" + textwrap.indent(truncate(evaluation.feedback, max_feedback), "    "))
    if individual.description:
        lines.append("  how it was made:\n" + textwrap.indent(truncate(individual.description, 500), "    "))
    return "\n".join(lines)


def join_notes(*parts: str | None) -> str | None:
    text = "\n\n".join(p for p in parts if p)
    return text or None


def scores_text(objectives) -> str:
    """Objectives on one line, `score=0.96, lines=55`; "failed" if there are none."""
    return ", ".join(f"{k}={v:.6g}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in (objectives or {}).items()) or "failed"


def brief(individual: Individual | None) -> str:
    """`ID (score=...)`: an individual in a few words."""
    if individual is None:
        return "(none)"
    return f"{individual.id} ({scores_text(individual.evaluation.objectives if individual.evaluation else None)})"


def format_attempts(entries) -> str:
    """`Run.log` entries, one line each: id, scores, kept or rejected, name."""
    return "\n".join(
        f"- {e['id']}  {scores_text(e.get('objectives'))}  {'kept' if e.get('admitted') else 'rejected'}  {e.get('name') or ''}".rstrip()
        for e in entries
    )
