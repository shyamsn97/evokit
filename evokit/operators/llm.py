from __future__ import annotations

import re
import time
from abc import abstractmethod
from typing import Any

from evokit.errors import OperatorError
from evokit.operator import Operator
from evokit.operators.models import Model
from evokit.operators.prompts import describe_individual
from evokit.records import Individual, Offspring, Selection, merge_usage
from evokit.util import truncate


class LLMOperator(Operator):
    """One prompt, one reply, one child. Subclasses say how to ask and how to read the reply.

    Raise ValueError from parse_response when the reply is unusable; the model's usage is
    still recorded.
    """

    name = "llm"

    def __init__(self, model: Model, instructions: str | None = None):
        self.model = model
        self.instructions = instructions

    @abstractmethod
    def build_prompt(self, selection: Selection) -> str: ...

    @abstractmethod
    def parse_response(self, text: str, selection: Selection) -> Any:
        """The new genome."""

    async def create(self, selection: Selection, archive) -> Offspring:
        prompt = self.build_prompt(selection)
        started = time.monotonic()
        try:
            completion = await self.model.generate(prompt)
        except Exception as exc:
            raise OperatorError(f"the model call failed: {type(exc).__name__}: {exc}", {"seconds": time.monotonic() - started}) from exc
        usage = merge_usage(completion.usage, {"seconds": time.monotonic() - started})
        try:
            genome = self.parse_response(completion.text, selection)
        except ValueError as exc:
            raise OperatorError(f"could not use the model's reply: {exc}\n\n{truncate(completion.text, 4000)}", usage) from exc
        if selection.parents and genome == selection.parents[0].genome:
            raise OperatorError("the model's reply did not change the genome", usage)
        info = {"operator": self.name, "response": truncate(completion.text, 20_000)}
        return Offspring((Individual.new(genome, [p.id for p in selection.parents], info),), usage)


_FENCE = re.compile(r"```[^\n]*\n(.*?)\n```", re.S)


class TextRewriteOperator(LLMOperator):
    """For str genomes: show the text, get back a complete rewrite in a fenced block."""

    name = "text-rewrite"

    def build_prompt(self, selection: Selection) -> str:
        parts = ["You are the variation operator in an evolutionary search. Propose one new version that scores better."]
        if self.instructions:
            parts.append(self.instructions)
        for individual in selection.parents:
            parts.append(describe_individual(individual) + f"\n  text:\n```\n{individual.genome}\n```")
        if selection.inspirations:
            parts.append("Other versions, for inspiration:")
        for individual in selection.inspirations:
            parts.append(describe_individual(individual) + f"\n  text:\n```\n{individual.genome}\n```")
        parts.append("Reply with the complete new text in a single fenced code block.")
        return "\n\n".join(parts)

    def parse_response(self, text: str, selection: Selection) -> str:
        blocks = _FENCE.findall(text)
        if blocks:
            return blocks[-1]
        if not text.strip():
            raise ValueError("empty reply")
        return text.strip()
