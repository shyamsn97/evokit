# DGM-Hyperagents

[Hyperagents](https://arxiv.org/abs/2603.19461) (Zhang et al., 2026) removes the Darwin Gödel
Machine's assumption that evaluation and self-modification are the same kind of task. In
[DGM](dgm.md):

- the improvement procedure is fixed: diagnose, then implement;
- the procedure only gets better because the agent gets better at coding;
- that only helps if the task is coding too.

A **hyperagent** puts the task agent and a **meta agent** in one editable repository. The
parent's meta agent edits a copy of the whole repository, its own code included. So the way
agents get improved evolves too, on any task. In the paper, the meta agent invented persistent
memory and performance tracking for itself, and those improvements transferred across domains.

In EvoKit this is DGM's loop with one different component: the operator.

```python
# evokit/recipes/hyperagents.py
async def hyperagents(eval_fn, *, storage, archive, fixed_meta_agent=None, **dgm_options):
    operator = MetaAgentModify(storage, fixed_meta_agent=fixed_meta_agent)
    return await dgm(eval_fn, storage=storage, archive=archive, operator=operator, **dgm_options)
```

```python
# evokit/agents/meta_agent.py
class MetaAgentModify(Operator):
    """The genome is one repository: task_agent.py solves the task, meta_agent.py improves the repository.
    The parent's meta agent edits a copy of the whole repository, itself included. With
    `fixed_meta_agent`, that meta agent edits every parent instead (the ablation without
    metacognitive self-modification)."""

    name = "meta-agent"

    def __init__(self, storage, *, entrypoint="meta_agent.py", logs="logs", fixed_meta_agent=None, timeout="30m"):
        self.storage, self.entrypoint, self.logs = storage, entrypoint, logs
        self.fixed_meta_agent, self.timeout = fixed_meta_agent, timeout

    async def create(self, selection, archive) -> Offspring:
        parent = selection.parents[0]
        with tempfile.TemporaryDirectory(prefix="evokit-hyperagent-") as tmp, \
                self.storage.local(f"artifacts/{parent.id}") as parent_artifacts:
            root = Path(tmp)
            parent.genome.write_to(root / "repo")
            write_history(root / "history", archive, parent_artifacts / self.logs)
            meta = root / "repo"
            if self.fixed_meta_agent is not None:
                meta = root / "meta"
                self.fixed_meta_agent.write_to(meta)
            argv = [sys.executable, str(meta / self.entrypoint), "--repo", str(root / "repo"), "--history", str(root / "history")]
            result = await CommandAgent(argv, self.timeout).run("", cwd=root / "repo")
            if result.error:
                raise OperatorError(f"the meta agent failed: {result.error}", usage=result.usage)
            child = Files.from_directory(root / "repo")
        if child == parent.genome:
            raise OperatorError("the meta agent changed nothing", usage=result.usage)
        name, description = title_and_body(result.summary)
        return Offspring((Individual.new(child, [parent.id], {"operator": self.name}, name=name, description=description),), result.usage)


def write_history(directory, archive, parent_logs):
    """What the meta agent can read: every archived agent's lineage and score, and the parent's logs."""
    directory.mkdir()
    with open(directory / "archive.jsonl", "w") as f:
        for member in archive.members():
            f.write(json.dumps({"id": member.id, "parents": member.parents,
                                "score": member.evaluation.objectives.get("score"),
                                "name": member.name, "description": member.description}) + "\n")
    if parent_logs.is_dir():
        shutil.copytree(parent_logs, directory / "parent_logs")
```

The pieces fit together like this:

- **The seed** is a repository with `task_agent.py` and a minimal `meta_agent.py`. In the
  paper, the seed meta agent is an LLM with tools, told to modify any part of the codebase.
- **`eval_fn`** runs `task_agent.py` on the domain's tasks, through
  `script_agent(..., entrypoint="task_agent.py")`. It returns logs under `logs/` (`Evaluation.logs`), where the
  next meta agent will read them.
- **Memory comes for free.** A meta agent can only remember across generations by writing files
  into the repository, for example notes or a score tracker. The repository is the genome, so
  that memory is inherited and selected like everything else.
- **The paper's ablations are arguments:**
  - `fixed_meta_agent=seed_repository`: the seed's meta agent edits every parent;
  - `open_ended=False`: no archive, build greedily on the latest agent.

## How it differs from the official code

Compared with [facebookresearch/Hyperagents](https://github.com/facebookresearch/Hyperagents) at
commit `59a68f6`:

- **A loop, not a system.** The official repository is the whole pipeline: domains with
  train/val/test splits, Docker, an ensemble step, baselines and analysis. This recipe is DGM's loop
  with `MetaAgentModify` as the operator. The caller brings the seed repository, `eval_fn` and the
  archive, and evokit ships no domains and no seed meta agent.
- **Full copies instead of patches.** The official loop applies the chain of ancestor patches to
  the initial repository, runs the meta agent in a git worktree, resets `domains/`, and saves the
  child as a diff. Here the parent's files are written to a temporary directory, and the edited
  directory is the child.
- **No Docker.** The official loop runs the meta agent, the evaluations, parent selection and the
  ensemble in containers. Here the meta agent is a subprocess on the host.
- **Serial generations versus parallel workers.** The official loop makes one child per
  generation, serially, for `--max_generation` generations (10 by default on the command line).
  Here 4 workers create and evaluate children at once, until 80 have been created.
- **A different parent rule.** The official default (`score_child_prop`) averages each candidate's
  validation scores across domains. Its sigmoid has sharpness 10 around the mean of the top
  3 scores, and its weight is multiplied by exp(-(children/8)^3). Here the sigmoid has sharpness 10
  around a fixed 0.5, and the weight is divided by (1 + children). The official loop can also let
  the agents edit the parent-selection code itself (`--edit_select_parent`); here selection is a
  fixed strategy.
- **The meta agent sees different things.** The official loop copies the earlier generations'
  outputs into the container, with validation and test results removed, and writes a README about
  self-improvement. The seed meta agent is Claude Sonnet 4.5 with bash and edit tools, a single
  instruction ("Modify any part of the codebase"), up to 40 tool calls and a 6-hour timeout. Here
  the meta agent gets `history/archive.jsonl` (id, parents, score, name and description of each
  archived agent) and the parent's own logs. It is run as
  `python meta_agent.py --repo ... --history ...` with a 30-minute timeout, and its model, tools
  and prompt are whatever the seed repository's `meta_agent.py` does.
- **Different staged evaluation.** The official loop has per-domain sample counts for the first
  stage (for example 10 for paper review, 3 for Genesis), promotes when every domain scores above 0,
  and scales partial scores for parent selection. Here `staged_eval` is generic, and the caller
  picks the subsets and the threshold.
- **One domain at a time.** The official loop evaluates several domains in parallel, averages them
  for selection, and supports resetting the task or meta agent and transfer from another run's
  archive. Here there is one `eval_fn`.
- **No ensemble and no held-out test.** The official loop can build an ensemble from the archive
  (the best agent per task), evaluate it on train, val and test, and evaluate on held-out test
  splits (`--eval_test`). Neither exists here.
- **Different failure bookkeeping.** The official archive takes every generation and marks bad ones
  `valid_parent: false`, and a failed meta-agent run marks the *parent* invalid. Here only children
  with a score enter the archive, and a failed meta-agent run is an operator error that counts
  toward `max_failures`.
- **Ablations are mapped, not copied.** The official "no self-improvement" ablation runs the meta
  agent from a frozen copy while the genome evolves; here `fixed_meta_agent` does the same job. The
  official DGM baselines swap in `coding_agent.py`; here that is the separate [DGM](dgm.md) recipe.

The same: the genome is one repository holding a task agent and a meta agent that may edit
everything, itself included. Both have an open-ended archive with a latest-only ablation and a
frozen-improver ablation, parent sampling discounted by number of children, and staged evaluation.
In both, the task agent is `task_agent.py`.
