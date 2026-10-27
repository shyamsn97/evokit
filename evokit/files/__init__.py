"""Optional: evolving files. A genome is a Files value; the rest of a Project is a fixed environment."""

from evokit.files.commands import FileCommands
from evokit.files.diff import apply_diff, apply_rewrite
from evokit.files.evaluator import ScriptEvaluator, code_lines, parse_metrics, with_lines
from evokit.files.operators import DiffOperator, FileRewriteOperator
from evokit.files.project import DEFAULT_PROTECTED, Project
from evokit.files.tree import DEFAULT_IGNORE, Files

__all__ = [
    "DEFAULT_IGNORE",
    "DEFAULT_PROTECTED",
    "DiffOperator",
    "FileCommands",
    "FileRewriteOperator",
    "Files",
    "Project",
    "ScriptEvaluator",
    "apply_diff",
    "apply_rewrite",
    "code_lines",
    "parse_metrics",
    "with_lines",
]
