---
name: evokit
description: Take part in or drive an EvoKit evolutionary search from the shell. Use when asked to evolve or optimize code with evokit, when a .evokit experiment directory exists, or when told to act as the evokit operator.
---

# EvoKit

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

## Command reference

- `evokit init NAME`: create a task folder from the template.
- `evokit checkout-individual ID DIRECTORY`: Write this individual (with the task's fixed files around it) into a new directory to edit.
- `evokit diff A [B]`: What changed from A to B, as a unified diff; with only A, from its parent to A.
- `evokit evaluate-directory DIRECTORY [--parent PARENT ...] [--name NAME] [--description DESCRIPTION]`: Make an individual from a directory's genome files, evaluate it, and offer it to the archive.
- `evokit grep PATTERN [--limit LIMIT]`: Individuals whose name, description or files match a regex, rejected ones too, so you can see what was already tried and how it scored.
- `evokit list-individuals`: The archive's members (best first, for ranked archives).
- `evokit log [ID] [--all] [--limit LIMIT]`: ID and its ancestors, nearest first, with names and scores, like `git log`; without ID, from the last individual the archive admitted. With --all, every evaluated individual, newest first, rejected ones too; `member` says whether it is still in the archive.
- `evokit notes [KIND] [--limit LIMIT]`: Notes left by earlier agents and by the experiment, newest first: what was learned, what to avoid.
- `evokit show ID [PATH]`: An individual's name, description, parents, evaluation and files; with PATH, that file.
- `evokit stats`: The experiment so far: evaluations, drafts, cost, attempts since the best last improved, and the best.
- `evokit status`: The archive's size and best individual, and what the experiment has created, evaluated and spent.
- `evokit run [RECIPE] TASK_DIR [--option value ...]`: run a recipe on the task, in --dir.
- `evokit mcp`: serve these commands as MCP tools.
