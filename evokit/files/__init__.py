"""Optional: evolving files. A genome is a Files value; the rest of a Project is a fixed environment."""

from evokit.files.agents import AgentResult, ClaudeCode, CodingAgent, CommandAgent
from evokit.files.coding_agent import CodingAgentOperator
from evokit.files.commands import FileCommands
from evokit.files.diff import apply_diff, apply_rewrite
from evokit.files.evaluator import ScriptEvaluator, code_lines, parse_metrics, with_lines
from evokit.files.meta_agent import MetaAgentModify, write_history
from evokit.files.operators import DiffOperator, FileRewriteOperator
from evokit.files.project import DEFAULT_PROTECTED, Project
from evokit.files.self_modify import SelfModify, script_agent
from evokit.files.tree import DEFAULT_IGNORE, Files

__all__ = [
    "AgentResult",
    "ClaudeCode",
    "CodingAgent",
    "CodingAgentOperator",
    "CommandAgent",
    "DEFAULT_IGNORE",
    "DEFAULT_PROTECTED",
    "DiffOperator",
    "FileCommands",
    "FileRewriteOperator",
    "Files",
    "MetaAgentModify",
    "Project",
    "ScriptEvaluator",
    "SelfModify",
    "apply_diff",
    "apply_rewrite",
    "code_lines",
    "parse_metrics",
    "script_agent",
    "with_lines",
    "write_history",
]
