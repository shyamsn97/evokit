# Agents

A coding agent can take part in two ways:

- **Inside the library.** A recipe (or your loop) starts the agent, gives it tools, and records
  everything. EvoKit drives the harness.
- **Outside the library.** Your own agent (Claude Code, Cursor, Codex, ...) is the operator.
  It drives an experiment through the `evokit` CLI or MCP.

## Harnesses

| Harness | What it runs | Tools reach it as |
| --- | --- | --- |
| `ClaudeCode(...)` | headless `claude -p`, resumed with `--resume` | an `evokit` command on its `PATH` |
| `CommandAgent(argv)` | any CLI agent, one process per turn | an `evokit` command on its `PATH` |
| `ReplAgent(model)` | any chat `Model`, writing Python in a persistent REPL | Python functions |
| your own `Agent` | anything | either |

### Claude Code

```python
from evokit.agents import ClaudeCode, NO_SHELL

agent = ClaudeCode(
    model="sonnet",              # default: Claude Code's own default
    max_turns=50,                # per prompt
    timeout="1h",                # per prompt
    auth="login",                # "env" (default) uses ANTHROPIC_API_KEY; "login" uses Claude Code's login
    allowed_tools=NO_SHELL,      # optional: read/edit files and `evokit *` only, so code runs only via score
    max_budget_usd=None,         # also capped per turn by what the experiment's max_cost leaves
    extra_args=[],               # passed straight to `claude`
)
```

EvoKit's system prompt is passed with `--append-system-prompt`. The rest of your Claude Code
setup (settings, MCP servers, `CLAUDE.md`) still applies. Transcripts are saved under
`sessions/` in the experiment.

### Any CLI agent

```python
from evokit.agents import CommandAgent

codex = CommandAgent(["codex", "exec", "{prompt}"], timeout="1h")
cursor = CommandAgent(["cursor-agent", "-p", "{prompt}"])
mine = CommandAgent("python my_agent.py")        # no {prompt}: the prompt goes to stdin
```

`{prompt}` and `{session}` are substituted into the command. If the last stdout line is a JSON
object, its `summary`, `cost_usd`, `tokens` and `session` fields are used. Otherwise stdout is
the reply and the cost is unknown.

### Any model, through a REPL

```python
from evokit.agents import ReplAgent
from evokit.operators import OpenAIModel

agent = ReplAgent(OpenAIModel("gpt-5-mini", price=(0.25, 2.0)), max_steps=50, timeout="1h")
```

The model writes ```` ```repl ```` blocks, which run in a Python process in its directory.
`sh(cmd)` runs shell commands. EvoKit's tools are plain functions there: `score()`,
`submit(name, description)`, `log()`, and so on.

## Inside the library

Every recipe that takes an agent accepts any harness:

```python
await recipes.avo(task, ClaudeCode(auth="login"), dir="experiments/avo-0", max_cost=5)
await recipes.autoresearch(task, CommandAgent(["codex", "exec", "{prompt}"]), dir="experiments/ar-0")
```

To write your own loop, use `experiment.start`:

```python
async with experiment.start(agent, start=parent, submit=True) as directory:
    async for event in agent.prompt("Improve the score; submit what is better."):
        match event:
            case ToolCall(name="submit"):
                print(event.reply)
            case Turn():
                print(event.text, event.usage)
    agent.send("Try a different data structure.")    # reaches it at its next tool call
    turn = await agent.prompt("Name it.", output=Name) # structured: turn.output is a Name
```

Inside the block:

- The agent works in `directory`, a checkout of `start` (default: the best).
- Its tools are `stats`, `log`, `show`, `diff`, `grep`, `notes`, `remember`, `score`, and
  `submit` when `submit=True`. Pass `tools=[fn]` to add your own; a function's name, parameters
  and docstring are what the agent sees.
- A system prompt tells it about the search, the editable files and the task's `TASK.md`.
- Every turn is recorded with its cost. The agent is stopped (`"stop"` in the tool reply, then
  killed after 60 s) once a limit is reached.
- `agent.fresh()` is a separate, read-only conversation, e.g. for a reviewer.
- `resume=session_id` continues an earlier conversation.

`await agent.check()` makes one tiny call to fail fast on a missing binary or bad auth. The
recipes do this for you.

Shell agents call tools as `evokit NAME ARGS...` (for example, `evokit submit "title"
"what changed"`). The command is a shim on the agent's `PATH` that routes back to the session,
so evokit must be importable by the agent's `python`.

## From the CLI

`evokit run` takes `--agent` for any recipe that has an agent:

```bash
evokit run avo packing --dir experiments/0 --max-cost 5                     # Claude Code
evokit run avo packing --dir experiments/0 --agent "codex exec {prompt}"   # any command
evokit run autoresearch packing --dir experiments/1 --agent "cursor-agent -p {prompt}"
```

The examples take `--agent claude` or `--agent repl:MODEL` (ReplAgent on an OpenAI model):

```bash
cd examples/circle_packing && python -m algorithms.avo --dir runs/avo-0 --agent repl:gpt-5-mini
```

## Outside the library: your agent as the operator

Here nothing runs a loop. The agent you are talking to reads the experiment, edits files, and
submits them.

**1. Teach it the commands.**

```bash
evokit skill export claude      # .claude/skills/evokit/SKILL.md
evokit skill export cursor      # .cursor/skills/evokit/SKILL.md
evokit skill export codex       # an AGENTS.md section
```

**2. Let it drive the CLI**, one command at a time:

```bash
evokit status --task packing --dir experiments/0         # creates and seeds the experiment
evokit list-individuals --dir experiments/0
evokit checkout-individual ID work/ --dir experiments/0
# ... the agent edits work/ ...
evokit evaluate-directory work/ --parent ID --name "what changed" --dir experiments/0
evokit log --all --dir experiments/0                     # what was tried, and how it scored
```

Every command prints JSON, never prompts, and reports errors as JSON on stderr. `--dir` defaults
to `.evokit` (or `$EVOKIT_DIR`).

**3. Or serve the same commands over MCP** (`pip install -e ".[mcp]"`):

```bash
claude mcp add evokit -- evokit mcp --dir experiments/0 --task packing
```

In Cursor, `.cursor/mcp.json`:

```json
{"mcpServers": {"evokit": {"command": "evokit", "args": ["mcp", "--dir", "experiments/0", "--task", "packing"]}}}
```

The server releases the experiment after each call, so `evokit run` or another process can
drive it in between.

**An RLM or a script** can use the same commands as plain functions:

```python
from evokit.interface.commands import experiment_tools

tools = experiment_tools(ek.open_experiment("experiments/0", task))
tools["evaluate_directory"]("work/", parents=[parent_id], name="...")
```

## Your own harness

Subclass `Agent` and implement one of these:

- `_run(cwd, prompt, session, env, budget) -> Turn` for one process per turn. Tools reach it
  through `env` (the `evokit` shim on `PATH`), and `session` is the id to resume.
- `session(cwd, tools, resume, env, system) -> Session` for a harness with its own SDK.

Optionally implement `check()`. Set `tool_style = "python"` if tools arrive as functions rather
than a shell command.
