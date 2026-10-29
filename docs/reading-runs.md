# Reading runs

Git-style commands that work on any experiment directory:

| Command | Shows |
| --- | --- |
| `evokit log [ID] [--all]` | ID and its ancestors; `--all` lists every evaluated individual, rejected ones too |
| `evokit show ID [PATH]` | an individual, or one of its files |
| `evokit diff A [B]` | what changed from A to B (or from A's parent) |
| `evokit grep PATTERN` | individuals whose name, description or files match |

```bash
evokit log --dir runs/avo-2
evokit diff 1f7ec51182 --dir runs/avo-2
evokit grep 'minimize\(' --dir runs/avo-2
```

In Python they are `Experiment` methods:

```python
experiment = ek.open_experiment("runs/avo-2")
experiment.log()
experiment.diff(experiment.lookup("1f7ec51182"))
```

Agents started by a recipe get the same commands as tools. Each individual has a `name` and a
`description`, set by whatever made it (an agent's `submit`, or an LLM's reply).

`evokit skill export claude|cursor|codex` writes these commands up as a skill for an outside
agent.
