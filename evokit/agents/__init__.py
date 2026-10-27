"""Coding agents, and the operators that use them to edit a Files genome."""

from evokit.agents.base import AgentResult, CodingAgent
from evokit.agents.claude import ClaudeCode
from evokit.agents.command import CommandAgent
from evokit.agents.meta_agent import MetaAgentModify, write_history
from evokit.agents.operator import CodingAgentOperator
from evokit.agents.repl import ReplAgent
from evokit.agents.self_modify import SelfModify, script_agent
from evokit.agents.tools import ToolServer, agent_tools

__all__ = [
    "AgentResult",
    "ClaudeCode",
    "CodingAgent",
    "CodingAgentOperator",
    "CommandAgent",
    "MetaAgentModify",
    "ReplAgent",
    "SelfModify",
    "ToolServer",
    "agent_tools",
    "script_agent",
    "write_history",
]
