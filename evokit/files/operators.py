"""LLM operators for Files genomes."""

from __future__ import annotations

import re

from evokit.files.diff import apply_diff, apply_rewrite
from evokit.files.project import Project
from evokit.files.tree import Files
from evokit.operators.llm import LLMOperator
from evokit.operators.models import Model
from evokit.operators.prompts import describe_individual
from evokit.records import Selection

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


class _FilesOperator(LLMOperator):
    format = ""

    def __init__(self, model: Model, project: Project | None = None, instructions: str | None = None):
        super().__init__(model, instructions)
        self.project = project

    def build_prompt(self, selection: Selection) -> str:
        base = selection.parents[0].genome if selection.parents else Files()
        parts = [
            "You are the variation operator in an evolutionary search over programs. "
            "Propose one new program that scores better on the objective."
        ]
        if self.project:
            parts.append(f"Editable files: {', '.join(self.project.editable)}")
        if any(frozen_text(base.text(p)) is not None for p in base.paths()):
            parts.append(
                "Only change code between the EVOLVE-BLOCK-START and EVOLVE-BLOCK-END markers; "
                "keep the markers and everything outside them exactly as they are."
            )
        if self.instructions:
            parts.append(self.instructions)
        parts.extend(describe_individual(i) for i in selection.parents)
        parts.append("Current program:\n\n" + base.render())
        others = [*selection.parents[1:], *selection.inspirations]
        if others:
            parts.append(
                "Other programs, for inspiration (they scored well, or differently from the current one). "
                "Borrow ideas from them; your reply edits the current program."
            )
        for other in others:
            parts.append(f"Program of {other.id}:\n\n" + other.genome.render())
        parts.append(self.format)
        return "\n\n".join(parts)

    def _check(self, base: Files, files: Files) -> Files:
        if self.project:
            outside = [p for p in files.paths() if not self.project.is_genome(p)]
            if outside:
                raise ValueError(f"edited files outside the genome: {', '.join(outside)}")
        for path in base.paths():
            before = frozen_text(base.text(path))
            if before is not None and (path not in files or frozen_text(files.text(path)) != before):
                raise ValueError(f"{path} was changed outside its EVOLVE-BLOCK markers")
        return files


class DiffOperator(_FilesOperator):
    """One prompt, one SEARCH/REPLACE diff (the AlphaEvolve / OpenEvolve / ShinkaEvolve baseline).

    The first parent is edited; other parents and the inspirations are shown alongside it. Files with EVOLVE-BLOCK
    markers may only change between them.
    """

    name = "diff"
    format = DIFF_FORMAT

    def parse_response(self, text: str, selection: Selection) -> Files:
        base = selection.parents[0].genome if selection.parents else Files()
        return self._check(base, apply_diff(base, text))


class FileRewriteOperator(_FilesOperator):
    """One prompt, complete new contents of the changed files."""

    name = "file-rewrite"
    format = REWRITE_FORMAT

    def parse_response(self, text: str, selection: Selection) -> Files:
        base = selection.parents[0].genome if selection.parents else Files()
        return self._check(base, apply_rewrite(base, text))
