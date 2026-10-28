"""The evokit skill: a SKILL.md generated from the same command functions as the CLI, shipped with the
package as evokit/skills/evokit/SKILL.md (a test keeps the two equal; regenerate it with
`evokit skill export --stdout > evokit/skills/evokit/SKILL.md`)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evokit.experiment import Experiment

FRONTMATTER = """---
name: evokit
description: Take part in or drive an EvoKit evolutionary search from the shell. Use when asked to evolve or optimize code with evokit, when a .evokit experiment directory exists, or when told to act as the evokit operator.
---

"""

BODY = """# EvoKit

EvoKit runs evolutionary search on a task: a folder with `TASK.md` (the goal, the rules, the
scoring), `config.yaml`, the evaluator, and the files that evolve. `evokit init NAME` creates one.
An experiment on a task lives in a directory (default `.evokit`, override with `--dir` or
`$EVOKIT_DIR`); the first command on a new directory needs `--task TASK_DIR`, and creates the
experiment, seeded with the task's files. Every command prints JSON on stdout, prints errors as
JSON on stderr, and never prompts.

An individual is a genome (here: a version of the task's evolving files) plus its evaluation:
its objectives, and feedback such as a traceback. An evaluation with no objectives failed. Each
has a `name` (one line saying what it is) and a `description` (what changed and why). The
archive holds the individuals that survive. Start with `evokit status`.

## Reading an experiment

These work like their git counterparts, on any experiment directory:

- `evokit log [ID]`: ID and its ancestors, nearest first, with names and scores; without ID, from
  the last individual the archive admitted. `--all` lists every evaluated individual, newest
  first, rejected ones too. `member` says whether it is still in the archive.
- `evokit show ID [PATH]`: its name, description, evaluation and files; with PATH, that file.
- `evokit diff A [B]`: what changed from A to B; with only A, from its parent to A.
- `evokit grep PATTERN`: individuals whose name, description or files match a regex, rejected
  ones too, so you can see what was already tried and how it scored.

## You are the operator

1. `evokit list-individuals` shows the archive, best first. Build on the best, or go back to an
   older one when stuck.
2. `evokit checkout-individual ID DIR` writes the task with that individual's files into a new
   directory. Read its evaluation and feedback with `evokit show ID`.
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

## EvoKit runs a recipe

`evokit run RECIPE TASK_DIR [--agent claude] [--max-cost USD] [--max-evaluations N]` runs a
published algorithm (avo, autoresearch, dgm, ...) on the task, in `--dir`, continuing it if it
exists. Follow it with `stats`, `log` and `notes`.

## You are an agent EvoKit started

Your working directory already holds the version to improve, and your `evokit` command reads the
experiment (stats, log, show, diff, grep, notes), keeps notes (remember), scores your files
(score) and, when you are told to, submits them (submit). Use them to see what was tried before
and how it scored, before repeating an idea. Finish your turn when you are done.

## Exit codes

`0` ok, `1` internal error, `2` usage error (read `hint`).
"""


def command_reference() -> str:
    """Every command with its usage and the first line of its docstring, from the same functions the
    CLI parses."""
    import functools

    from evokit.agents.tools import tool_summary, usage
    from evokit.interface.commands import COMMANDS, READ

    lines = ["## Command reference", "", "- `evokit init NAME`: create a task folder from the template."]
    for name, fn in sorted({**READ, **COMMANDS}.items()):
        lines.append(f"- `{usage(functools.partial(fn, None), name)}`: {tool_summary(fn)}")
    lines.append("- `evokit run [RECIPE] TASK_DIR [--option value ...]`: run a recipe on the task, in --dir.")
    lines.append("- `evokit mcp`: serve these commands as MCP tools.")
    return "\n".join(lines) + "\n"


def experiment_section(experiment: "Experiment | None") -> str:
    if experiment is None or experiment.archive is None:
        return ""
    lines = ["## This experiment", "", f"- directory: `{experiment.storage.root}`"]
    task = experiment.task
    if task is not None and task.directory is not None:
        lines.append(f"- task: `{task.directory}`")
    lines.append(f"- archive: {type(experiment.archive).__name__}")
    objective = getattr(experiment.archive, "objective", None)
    if objective:
        direction = "minimize" if getattr(experiment.archive, "minimize", False) else "maximize"
        lines.append(f"- objective: {direction} `{objective}`")
    if task is not None:
        lines.append(f"- evolving files: {', '.join(task.editable)}")
        lines += ["", "The task's TASK.md:", ""] + [f"> {line}".rstrip() for line in task.instructions.splitlines()]
    return "\n".join(lines) + "\n\n"


def render_skill(experiment: "Experiment | None" = None, target: str = "claude") -> str:
    """The skill: the same for every experiment, except that MCP instructions describe the one they serve."""
    body = BODY + "\n" + experiment_section(experiment) + command_reference()
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
