# Reading a run

**Status: implemented.** Replaces the git repository in [git-view.md](git-view.md): evokit has its
own git-style read commands, which work for every genome type and every run. Agents can still use
git in their own workspace.

## What a person or an agent needs

The AVO paper's agent reads its lineage: the versions kept so far, what each changed, and its
score. A person debugging a run needs the same, plus the attempts that were rejected. Four commands
cover it, named and behaving like their git counterparts:

| command | what it shows |
| --- | --- |
| `evokit log [ID] [--all]` | ID and its ancestors, nearest first: name, scores, whether it was admitted and whether it is still in the archive. Without ID, the last admitted individual (git's `HEAD`). With `--all`, every evaluated individual, newest first, rejected ones too. |
| `evokit show ID [PATH]` | name, description, evaluation and files; with PATH, that file |
| `evokit diff A [B]` | what changed from A to B; with only A, from its first parent to A |
| `evokit grep PATTERN` | individuals whose name, description or genome matches a regex, rejected ones too |

```bash
$ evokit log --dir runs/avo-2
[
  {"id": "1f7ec51182", "name": "2.6245! This matches my best score from earlier. ...", "parents": ["375411150b"],
   "objectives": {"score": 2.6245}, "admitted": true, "member": true, "operator": "coding-agent"},
  {"id": "375411150b", ..., "member": false},
  ...
]
$ evokit diff 1f7ec51182 --dir runs/avo-2          # what the last step changed
$ evokit grep 'minimize\(' --dir runs/avo-2        # who tried it, and how it scored
```

(`runs/avo-2` predates names, so its names are the first line of each agent's old summary; new
runs ask the agent for a one-line title.)

They read only the run's storage (its history, the archive's log, and the blobs), so they work on
any run directory, including runs made by a Python recipe that has no `run/task.json`. In Python
they are methods of `HistoryCommands(storage)`:

```python
from evokit.interface.history import HistoryCommands

runs = HistoryCommands(DirectoryStorage("runs/avo-2"))
runs.log()                       # list[dict]
runs.diff(runs.lookup("1f7ec51182"))
```

## A name and a description for every individual

Each individual has a `name` (one line, like a commit subject) and a `description` (what changed
and why, like a commit body). The operator that makes it sets them:

| made by | name | description |
| --- | --- | --- |
| a coding agent | the first line of its reply | the rest of its reply |
| an LLM operator | the first line of its reply before any code | the rest of that text |
| a seed | `seed` | |
| an island migration | `copy of <id> on island <n>` | |
| `evaluate-directory` | `--name` | `--description` |

Runs written before this read `info["summary"]` or `info["note"]` as the name (first line) and the
description (the rest).

## Each genome type says how it differs

`diff` calls `genome.diff(other, a, b)` when the genome type has it. `Files` gives a unified diff of
each changed file. Other genomes fall back to a unified diff of their text: a string as it is,
anything else as indented JSON. `grep` searches a `Files` genome file by file, a string as it is,
and anything else as its JSON.

## The skill

`evokit/skills/evokit/SKILL.md` is generated once from the command registry and shipped with the
package; a test keeps it in sync with the commands. It has a section for each kind of agent: one
that is the operator itself, one that drives `evokit run`, and an operator's agent (inside
`CodingAgentOperator`) that only reads the run. `evokit skill export claude|cursor` copies it into
a project.

`CodingAgentOperator` copies it into each agent's workspace (`.claude/skills/evokit/` for Claude
Code) and sets `EVOKIT_DIR` to the run's directory, when the run is on local disk. `.claude/` and
`.cursor/` are ignored when the genome is read back, so the skill never becomes part of a child.
