---
name: evokit
description: Take part in or drive an EvoKit evolutionary search from the shell. Use when asked to evolve or optimize code with evokit, when a .evokit run directory exists, or when told to act as the evokit operator.
---

# EvoKit

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

## Command reference

- `evokit init NAME`: create a task folder from the template.
- `evokit best-individual`: The archive's best member.
- `evokit checkout-individual ID DIR`: Write the full project (environment plus this individual's genome) into a new directory to edit.
- `evokit diff A [B]`: What changed from A to B, as a unified diff; with only A, from its first parent to A (like `git show`).
- `evokit evaluate-directory DIR [--parent ID ...] [--name NAME] [--description DESCRIPTION]`: Make an individual from a directory's genome files, evaluate it, and offer it to the archive.
- `evokit grep PATTERN [--limit LIMIT]`: Individuals whose name, description or genome matches a regex, rejected ones too.
- `evokit list-events [--limit LIMIT]`: The run's history, oldest first: created, evaluated (and whether admitted), discarded, notes.
- `evokit list-individuals`: The archive's members (best first, for ranked archives).
- `evokit log [ID] [--all] [--limit LIMIT]`: An individual and its ancestors, nearest first, like `git log`.
- `evokit run [--workers WORKERS] [--max-evaluations MAX_EVALUATIONS] [--max-cost MAX_COST] [--max-failures MAX_FAILURES]`: Run ask, evaluate and tell with the run's operator until a limit is reached or nothing can be selected.
- `evokit select-individuals [--n N]`: Ask the strategy what to build on next: each selection has parents and inspirations.
- `evokit show ID [PATH]`: An individual's name, description, parents, evaluation and files; with `path`, that file's text.
- `evokit status`: The archive's size and best individual, and what the run has created, evaluated and spent.
- `evokit mcp`: serve these commands as MCP tools.
- `evokit --help` in a run directory also lists the commands its archive, strategy and operator add.
