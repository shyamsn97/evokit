"""Coding agents: the harness interface, the harnesses EvoKit ships, and how their tools reach them."""

from evokit.agents.claude import NO_SHELL, ClaudeCode
from evokit.agents.command import CommandAgent
from evokit.agents.harness import Agent, Event, Prompt, Session, ToolCall, Turn
from evokit.agents.repl import ReplAgent
from evokit.agents.tools import ShellTools

__all__ = [
    "NO_SHELL",
    "Agent",
    "ClaudeCode",
    "CommandAgent",
    "Event",
    "Prompt",
    "ReplAgent",
    "Session",
    "ShellTools",
    "ToolCall",
    "Turn",
]
