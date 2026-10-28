"""Variation of Files genomes by one model call: a SEARCH/REPLACE diff, or whole new files."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Callable, Sequence

from evokit.files.diff import apply_diff, apply_rewrite
from evokit.files.tree import Files
from evokit.operators.models import Model
from evokit.operators.prompts import describe_individual
from evokit.operators.text import generate
from evokit.records import Individual

if TYPE_CHECKING:
    from evokit.task import Task

DIFF_FORMAT = """Reply with one or more SEARCH/REPLACE blocks. Each block names the file on the line before it:

path/to/file.py
<<<<<<< SEARCH
exact existing lines to replace
=======
new lines
>>>>>>> REPLACE

The SEARCH text must match the current file exactly. Use an empty SEARCH section to create a new file."""

REWRITE_FORMAT = """Reply with the complete new contents of every file you change, each as a fenced code block preceded by its path:

### path/to/file.py
```
full file contents
```"""

_BLOCK = re.compile(r"(^[^\n]*EVOLVE-BLOCK-START[^\n]*\n).*?(^[^\n]*EVOLVE-BLOCK-END)", re.M | re.S)


def frozen_text(text: str) -> str | None:
    """The text with every EVOLVE-BLOCK's contents removed, or None if it has no blocks."""
    return _BLOCK.sub(r"\1\2", text) if _BLOCK.search(text) else None


async def diff(
    model: Model,
    parent: Individual,
    others: Sequence[Individual] = (),
    *,
    task: "Task | None" = None,
    instructions: str | None = None,
) -> tuple[Individual, dict]:
    """One prompt, one SEARCH/REPLACE diff of `parent` (the AlphaEvolve / OpenEvolve / ShinkaEvolve
    baseline); `others` are shown alongside it. Files with EVOLVE-BLOCK markers may only change
    between them, and with `task` only its editable files. `instructions` default to the task's."""
    return await _vary(model, parent, others, task, instructions, DIFF_FORMAT, apply_diff, "diff")


async def rewrite_files(
    model: Model,
    parent: Individual,
    others: Sequence[Individual] = (),
    *,
    task: "Task | None" = None,
    instructions: str | None = None,
) -> tuple[Individual, dict]:
    """One prompt, complete new contents of the changed files of `parent`; checked as `diff` is."""
    return await _vary(model, parent, others, task, instructions, REWRITE_FORMAT, apply_rewrite, "file-rewrite")


async def _vary(model, parent, others, task, instructions, format: str, apply: Callable[[Files, str], Files], operator: str):
    base = parent.genome
    if instructions is None:
        instructions = task.instructions if task is not None else ""
    parts = [
        "You are the variation operator in an evolutionary search over programs. "
        "Propose one new program that scores better on the objective."
    ]
    if task is not None:
        parts.append(f"Editable files: {', '.join(task.editable)}")
    if any(frozen_text(base.text(p)) is not None for p in base.paths()):
        parts.append(
            "Only change code between the EVOLVE-BLOCK-START and EVOLVE-BLOCK-END markers; "
            "keep the markers and everything outside them exactly as they are."
        )
    if instructions:
        parts.append(instructions)
    parts.append(describe_individual(parent))
    parts.append("Current program:\n\n" + base.render())
    if others:
        parts.append(
            "Other programs, for inspiration (they scored well, or differently from the current one). "
            "Borrow ideas from them; your reply edits the current program."
        )
    for other in others:
        parts.append(f"Program of {other.id}:\n\n" + other.genome.render())
    parts.append(format)
    return await generate(model, "\n\n".join(parts), [parent], lambda reply: _check(base, apply(base, reply), task), operator)


def _check(base: Files, files: Files, task) -> Files:
    if task is not None and task.files is not None:
        outside = [p for p in files.paths() if not task.files.is_genome(p)]
        if outside:
            raise ValueError(f"edited files outside the genome: {', '.join(outside)}")
    for path in base.paths():
        before = frozen_text(base.text(path))
        if before is not None and (path not in files or frozen_text(files.text(path)) != before):
            raise ValueError(f"{path} was changed outside its EVOLVE-BLOCK markers")
    return files


__all__ = ["DIFF_FORMAT", "REWRITE_FORMAT", "diff", "frozen_text", "rewrite_files"]
