"""The evokit skill: a SKILL.md generated from the same @command registry, shipped with the
package as evokit/skills/evokit/SKILL.md (a test keeps the two equal; regenerate it with
`evokit skill export --stdout > evokit/skills/evokit/SKILL.md`)."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import TYPE_CHECKING

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
failed. Each has a `name` (one line saying what it is) and a `description` (what changed and
why). The archive holds the individuals that survive. Start with `evokit status`.

## Reading a run

These work like their git counterparts, on any run directory:

- `evokit log [ID]`: ID and its ancestors, nearest first, with names and scores; without ID, from
  the last individual the archive admitted. `--all` lists every evaluated individual, newest
  first, rejected ones too. `member` says whether it is still in the archive.
- `evokit show ID [PATH]`: its name, description, evaluation and files; with PATH, that file.
- `evokit diff A [B]`: what changed from A to B; with only A, from its parent to A.
- `evokit grep PATTERN`: individuals whose name, description or files match a regex, rejected
  ones too, so you can see what was already tried and how it scored.

## You are the operator (when the run has no operator)

1. `evokit select-individuals` asks the strategy what to build on: each selection has
   `parents` (edit the first) and `inspirations` (ideas to borrow). You may also pick others
   yourself with `evokit list-individuals`, for example to go back to an older one when stuck.
2. `evokit checkout-individual ID DIR` writes the full project with that individual's files into
   a new directory. Read its evaluation and feedback with `evokit show ID`.
3. Edit the files in `DIR`. Only files matching the genome patterns are kept. Never edit other
   files, especially the evaluator; evaluations of directories with such edits are rejected.
4. `evokit evaluate-directory DIR --parent ID --name "one line" --description "what you changed
   and why"` evaluates it, records it as a new individual, and offers it to the archive. The
   result has its `id`, its evaluation, and `info.admitted`: admitted means kept, otherwise you
   build on the previous best again. A crash is recorded too; read its feedback, fix it, try
   again.
5. Repeat. Do not stop to ask whether to continue: keep going until you are told to stop. Every
   attempt, kept or not, is in `evokit log --all`.

Several agents can work at the same time, each in its own directory.

## EvoKit is the operator (when the run has an operator)

`evokit run [--workers N] [--max-evaluations N] [--max-cost USD]` runs ask, evaluate and tell
until a limit (counted over the whole run) is reached. Follow it with `status`, `log` and
`list-events`.

## You are an operator's agent (EvoKit started you to improve one individual)

Your working directory already holds the individual to improve, and `$EVOKIT_DIR` points at the
run, so the commands under "Reading a run" work as they are. Use them to see what was tried
before and how it scored, before repeating an idea. Do not run `evaluate-directory`, `run` or
`select-individuals`: edit the files in your working directory, and finish your session when you
are done; EvoKit evaluates the result.

## Exit codes

`0` ok, `1` internal error, `2` usage error (read `hint`).
"""


def _usage(fn) -> str:
    from evokit.interface.cli import CLI_NAMES, _is_individual, cli_params, hints, positional

    types = hints(fn)
    parts = []
    for param in cli_params(fn) or ():
        hint = types.get(param.name, "")
        flag = CLI_NAMES.get(param.name, "--" + param.name.replace("_", "-"))
        if positional(fn, param):
            if _is_individual(hint):
                word = "ID" if param.name == "individual" else param.name.upper()
            elif param.name == "directory":
                word = "DIR"
            else:
                word = param.name.upper()
            parts.append(word if param.default is inspect.Parameter.empty else f"[{word}]")
        elif "list" in repr(hint):
            parts.append(f"[{flag} {'ID' if param.name == 'parents' else 'VALUE'} ...]")
        elif hint is bool:
            parts.append(f"[{flag}]")
        else:
            parts.append(f"[{flag} {param.name.upper()}]")
    return " ".join(parts)


def command_reference() -> str:
    """Every built-in command with its usage and the first line of its docstring."""
    from evokit.interface.cli import class_commands, cli_params

    lines = ["## Command reference", "", "- `evokit init NAME`: create a task folder from the template."]
    for name, _, _, fn in class_commands("cli"):
        if cli_params(fn) is None:
            continue
        doc = (inspect.getdoc(fn) or "").splitlines()
        line = " ".join(x for x in ("evokit", name, _usage(fn)) if x)
        lines.append(f"- `{line}`: {doc[0] if doc else ''}")
    lines.append("- `evokit mcp`: serve these commands as MCP tools.")
    lines.append("- `evokit --help` in a run directory also lists the commands its archive, strategy and operator add.")
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


def render_skill(setup: "Setup | None" = None, target: str = "claude") -> str:
    """The skill: the same for every run, except that MCP instructions describe the run they serve."""
    body = BODY + "\n" + run_section(setup) + command_reference()
    if target in ("claude", "cursor"):
        return FRONTMATTER + body
    return body


SKILL_FILE = Path(__file__).resolve().parent.parent / "skills" / "evokit" / "SKILL.md"
MARK_BEGIN, MARK_END = "<!-- evokit:begin -->", "<!-- evokit:end -->"


def install_skill(directory: str | Path, skills: str = ".claude/skills") -> Path:
    """Copy the shipped SKILL.md into `directory`/`skills`/evokit/."""
    path = Path(directory) / skills / "evokit" / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(SKILL_FILE.read_bytes())
    return path


def export_skill(target: str = "claude", root: str | Path = ".") -> Path:
    root = Path(root)
    if target in ("claude", "cursor"):
        return install_skill(root, f".{target}/skills")
    path = root / "AGENTS.md"
    block = f"{MARK_BEGIN}\n{render_skill(None, target)}{MARK_END}\n"
    existing = path.read_text() if path.exists() else ""
    if MARK_BEGIN in existing and MARK_END in existing:
        head, rest = existing.split(MARK_BEGIN, 1)
        tail = rest.split(MARK_END, 1)[1].lstrip("\n")
        path.write_text(head + block + tail)
    else:
        path.write_text(existing + ("\n" if existing and not existing.endswith("\n") else "") + block)
    return path
