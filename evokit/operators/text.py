"""Variation by one model call: a prompt, a reply, a child. `generate` is the call; `rewrite_text`
uses it for str genomes."""

from __future__ import annotations

import re
import time
from typing import Any, Callable, Sequence

from evokit.errors import OperatorError
from evokit.operators.models import Model
from evokit.operators.prompts import describe_individual
from evokit.records import Individual, merge_usage
from evokit.util import title_and_body, truncate

_PROSE = re.compile(r"^(?:```|<<<<<<< SEARCH)", re.M)

_FENCE = re.compile(r"```[^\n]*\n(.*?)\n```", re.S)


async def generate(
    model: Model,
    prompt: str,
    parents: Sequence[Individual],
    parse: Callable[[str], Any],
    operator: str,
) -> tuple[Individual, dict]:
    """Ask `model`, and make a child of `parents` from its reply: `parse(text)` gives the genome
    (raise ValueError if the reply is unusable), and the prose before the first code block its
    name and description. Raises OperatorError, with the cost, if the call fails, the reply is
    unusable, or the genome did not change."""
    started = time.monotonic()
    try:
        completion = await model.generate(prompt)
    except Exception as exc:
        raise OperatorError(f"the model call failed: {type(exc).__name__}: {exc}", {"seconds": time.monotonic() - started}) from exc
    usage = merge_usage(completion.usage, {"seconds": time.monotonic() - started})
    try:
        genome = parse(completion.text)
    except ValueError as exc:
        raise OperatorError(f"could not use the model's reply: {exc}\n\n{truncate(completion.text, 4000)}", usage) from exc
    if parents and genome == parents[0].genome:
        raise OperatorError("the model's reply did not change the genome", usage)
    info = {"operator": operator, "response": truncate(completion.text, 20_000)}
    name, description = title_and_body(_PROSE.split(completion.text, 1)[0])
    child = Individual.new(genome, [p.id for p in parents], info, name=name, description=truncate(description, 2000))
    return child, usage


async def rewrite_text(
    model: Model, parent: Individual, others: Sequence[Individual] = (), *, instructions: str = ""
) -> tuple[Individual, dict]:
    """For str genomes: show `parent` (and `others`, for inspiration), get back a complete rewrite
    in a fenced block."""
    parts = ["You are the variation operator in an evolutionary search. Propose one new version that scores better."]
    if instructions:
        parts.append(instructions)
    parts.append(describe_individual(parent) + f"\n  text:\n```\n{parent.genome}\n```")
    if others:
        parts.append("Other versions, for inspiration:")
    for other in others:
        parts.append(describe_individual(other) + f"\n  text:\n```\n{other.genome}\n```")
    parts.append("Reply with the complete new text in a single fenced code block.")
    return await generate(model, "\n\n".join(parts), [parent], _text, "text-rewrite")


def _text(reply: str) -> str:
    blocks = _FENCE.findall(reply)
    if blocks:
        return blocks[-1]
    if not reply.strip():
        raise ValueError("empty reply")
    return reply.strip()


__all__ = ["generate", "rewrite_text"]
