"""Variation functions: each makes one child, returning (child, usage) or raising
OperatorError(reason, usage), for `experiment.create`. And the models they call."""

from evokit.operators.files import diff, rewrite_files
from evokit.operators.meta_agent import meta_agent_modify, write_history
from evokit.operators.models import AnthropicModel, Completion, FunctionModel, Model, OpenAIModel
from evokit.operators.self_modify import script_agent, self_modify
from evokit.operators.text import generate, rewrite_text

__all__ = [
    "AnthropicModel",
    "Completion",
    "FunctionModel",
    "Model",
    "OpenAIModel",
    "diff",
    "generate",
    "meta_agent_modify",
    "rewrite_files",
    "rewrite_text",
    "script_agent",
    "self_modify",
    "write_history",
]
