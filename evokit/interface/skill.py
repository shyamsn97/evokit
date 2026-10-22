"""`evokit skill export`: a SKILL.md generated from the same @command registry."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import TYPE_CHECKING

from evokit.interface.commands import component_commands

if TYPE_CHECKING:
    from evokit.interface.setup import Setup

FRONTMATTER = """---
name: evokit
description: Take part in or drive an EvoKit evolutionary search from the shell. Use when asked to evolve or optimize code with evokit, when a .evokit run directory exists, or when told to act as the evokit operator.
---

"""

BODY = """# EvoKit

EvoKit runs evolutionary search on a task: a folder with `TASK.md` (the goal, the rules, the
scoring), `config.yaml`, the evaluator, and the files that evolve. `evokit init NAME` creates one.
A run of a task lives in a directory (default `.evokit`, override with `--dir` or `$EVOKIT_DIR`);
the first command on a new directory needs `--task TASK_DIR`, and creates the run, seeded with
the task's files. Every command prints JSON on stdout, prints errors as JSON on stderr, and
never prompts.

An individual is a genome (here: a version of the project's evolving files) plus its
evaluation: its objectives, and feedback such as a traceback. An evaluation with no objectives
failed. The archive holds the individuals that survive. Start with `evokit status`.

## You are the operator (when the run has no operator)

1. `evokit select-individuals` asks the strategy what to build on: each selection has
   `parents` (edit the first) and `inspirations` (ideas to borrow). You may also pick others
   yourself with `evokit list-individuals`, for example to go back to an older one when stuck.
2. `evokit checkout-individual ID DIR` writes the full project with that individual's files into
   a new directory. Read its evaluation and feedback with `evokit get-individual ID`.
3. Edit the files in `DIR`. Only files matching the genome patterns are kept. Never edit other
   files, especially the evaluator; evaluations of directories with such edits are rejected.
4. `evokit evaluate-directory DIR --parent ID --note "what you changed and why"` evaluates it,
   records it as a new individual, and offers it to the archive. The result has its `id`, its
   evaluation, and `info.admitted`: admitted means kept, otherwise you build on the previous best
   again. A crash is recorded too; read its feedback, fix it, try again.
5. Repeat. Do not stop to ask whether to continue: keep going until you are told to stop. Every
   attempt, kept or not, is in `evokit list-events`.

Several agents can work at the same time, each in its own directory.

## EvoKit is the operator (when the run has an operator)

`evokit run [--workers N] [--max-evaluations N] [--max-cost USD]` runs ask, evaluate and tell
until a limit (counted over the whole run) is reached. Inspect with `status`, `best-individual`,
`list-individuals`, `get-individual ID`, `read-file ID PATH`, `diff-individuals A B`, `list-events`.

## Exit codes

`0` ok, `1` internal error, `2` usage error (read `hint`).
"""


def _usage(fn) -> str:
    from evokit.interface.cli import CLI_NAMES, cli_params, hints
    from evokit.records import Individual

    types = hints(fn)
    parts = []
    for param in cli_params(fn) or ():
        hint = types.get(param.name, "")
        flag = CLI_NAMES.get(param.name, "--" + param.name.replace("_", "-"))
        if param.default is inspect.Parameter.empty:
            if hint is Individual:
                parts.append(f"{param.name.upper()}_ID" if param.name != "individual" else "INDIVIDUAL_ID")
            elif param.name == "directory":
                parts.append("DIR")
            else:
                parts.append(param.name.upper())
        elif "list" in repr(hint):
            parts.append(f"[{flag} {'ID' if param.name == 'parents' else 'VALUE'} ...]")
        elif hint is bool:
            parts.append(f"[{flag}]")
        else:
            parts.append(f"[{flag} {param.name.upper()}]")
    return " ".join(parts)


def command_reference(setup: "Setup | None") -> str:
    from evokit.interface.cli import class_commands, cli_params

    present = None if setup is None else {type(o) for o in setup.objects}
    lines = ["## Command reference", "", "- `evokit init NAME`: create a task folder from the template."]
    for name, _, cls, fn in class_commands("cli"):
        if (present is not None and not any(issubclass(p, cls) for p in present)) or cli_params(fn) is None:
            continue
        doc = (inspect.getdoc(fn) or "").splitlines()
        line = " ".join(x for x in ("evokit", name, _usage(fn)) if x)
        lines.append(f"- `{line}`: {doc[0] if doc else ''}")
    if setup is not None:
        for slot, cmds in component_commands(setup.objects, "cli").items():
            for cmd in cmds:
                line = " ".join(x for x in ("evokit", slot, cmd.name, _usage(cmd.func)) if x)
                lines.append(f"- `{line}`: {cmd.summary}")
    lines.append("- `evokit mcp`: serve these commands as MCP tools.")
    return "\n".join(lines) + "\n"


def run_section(setup: "Setup | None") -> str:
    if setup is None:
        return ""
    from evokit.files.commands import FileCommands

    run = setup.run
    objective = getattr(run.archive, "objective", None)
    if run.operator is None:
        operator = f"you (the strategy, {type(run.strategy).__name__}, selects individuals)"
    else:
        operator = f"EvoKit ({type(run.operator).__name__})"
    lines = ["## This run", "", f"- directory: `{setup.directory}`"]
    if run.task is not None:
        lines.append(f"- task: `{run.task.directory}`")
    lines += [f"- archive: {type(run.archive).__name__}", f"- operator: {operator}"]
    if objective:
        direction = "minimize" if getattr(run.archive, "minimize", False) else "maximize"
        lines.append(f"- objective: {direction} `{objective}`")
    files = setup.find(FileCommands)
    if files is not None:
        lines.append(f"- evolving files: {', '.join(files.project.editable)}")
    if run.task is not None:
        lines += ["", "The task's TASK.md:", ""] + [f"> {line}".rstrip() for line in run.task.instructions.splitlines()]
    return "\n".join(lines) + "\n\n"


def render_skill(setup: "Setup | None", target: str = "claude") -> str:
    body = BODY + "\n" + run_section(setup) + command_reference(setup)
    if target in ("claude", "cursor"):
        return FRONTMATTER + body
    return body


MARK_BEGIN, MARK_END = "<!-- evokit:begin -->", "<!-- evokit:end -->"


def export_skill(setup: "Setup | None", target: str = "claude", root: str | Path = ".") -> Path:
    root = Path(root)
    text = render_skill(setup, target)
    if target == "codex":
        path = root / "AGENTS.md"
        block = f"{MARK_BEGIN}\n{text}{MARK_END}\n"
        existing = path.read_text() if path.exists() else ""
        if MARK_BEGIN in existing and MARK_END in existing:
            head, rest = existing.split(MARK_BEGIN, 1)
            tail = rest.split(MARK_END, 1)[1].lstrip("\n")
            path.write_text(head + block + tail)
        else:
            path.write_text(existing + ("\n" if existing and not existing.endswith("\n") else "") + block)
        return path
    path = root / (".claude" if target == "claude" else ".cursor") / "skills" / "evokit" / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path
