# How a run is stored, and its lineage as git

**Status: superseded** by [reading-runs.md](reading-runs.md): evokit's own `log`, `show`, `diff`
and `grep` commands read a run without git, for every genome type. Git is optional; an agent can
still use it in its own workspace. The research below is why the lineage is the kept versions.

Two questions:

1. A coding agent (and a person) should be able to read the search: the versions kept so far,
   what each one changed, and its score. Git is the natural format. What goes into it?
2. Is our run storage (history, archive log, blobs, artifacts) more complicated than it needs
   to be?

The answer comes from how the published systems store their runs.

## How others store a run

| system | every attempt is recorded in | git holds | logs and large outputs |
| --- | --- | --- | --- |
| AVO paper | (not described) | "each committed version x_i is persisted as a git commit along with its score"; "unsuccessful intermediate attempts ... are not added to the committed lineage" | (not described) |
| [AVO reproduction](https://github.com/gatordevin/avo) | `trajectory.jsonl` (accepted and rejected steps), `rejected/step-NNNN.patch` | the agent's work tree is the repository; only accepted versions are committed, tagged `vN`, plus `.avo/scores.jsonl` | `logs/step-NNNN.log`, outside git |
| [autoresearch](https://github.com/karpathy/autoresearch) | `results.tsv`, "do not commit the results.tsv file, leave it untracked" | the experiment branch: every experiment is committed, and a worse one is `git reset`, so the branch holds only kept versions | `run.log`, untracked |
| [OpenEvolve](https://github.com/codelion/openevolve) | an in-memory program database (code, `parent_id`, metrics, metadata), saved to `checkpoints/checkpoint_<n>/` | no git | in the checkpoint |
| [ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve) | one SQLite database: a `programs` table (code, `parent_id`, metrics, `text_feedback`, `correct`, diff) and an `archive` table of program ids | no git | per-generation results folders |
| [DGM](https://github.com/jennyzzt/dgm) | one folder per attempt: `model_patch.diff` and `metadata.json` (parent, scores); `dgm_metadata.jsonl` lists the archive each generation | git inside the container, only to apply the chain of parent patches | in the attempt's folder |

What they agree on:

- **Every attempt is a record** with its parent, scores, and feedback. It lives in a database or an
  append-only log, never only in git.
- **Where git is used, it holds the kept lineage**, and only the program's files. Rejected
  attempts are reset out of it (autoresearch), or kept as a patch file next to it (the AVO
  reproduction).
- **Logs, transcripts, and evaluation outputs are never committed.**
- **The archive is a list of program ids** (Shinka's `archive` table, DGM's `archive` list), not a
  second copy of the programs.

## How evokit stores a run now

```text
runs/avo-2/
  history/records.jsonl    every created, evaluated, discarded individual, and notes; with costs
  archive/changes.jsonl    the archive's inserts and removals
  blobs/<ab>/<hash>        genome file contents, content-addressed
  artifacts/<id>/          each individual's evaluation outputs, and agent.jsonl (the agent's session)
  run/                     the driver lock, and task.json for runs made from a task folder
  env/, env-scratch/       the task's fixed files, and the persistent agent's workspace
```

This is the same shape as ShinkaEvolve. `history/records.jsonl` is its `programs` table, and
`archive/changes.jsonl` is its `archive` table. The difference is that ours are append-only
files, so a run survives a crash and can live on S3. That part is not weird.

The archive log stores each member in full, which the history also has:

```json
{"inserted": {"id": "8374fa8dae", "genome": {...}, "parents": [], "info": {}, "evaluation": {...}}, "metadata": {}, "time": ...}
{"removed": "8374fa8dae", "time": ...}
```

This stays. Genome contents are in `blobs/` once, so the copy is only ids, scores and feedback
(5 KB of the 852 KB in `runs/avo-2`). It also covers two cases a log of ids cannot:

- an archive's own members that never pass through `tell`, like `IslandArchive`'s copies of the
  seed into each island and its migrants;
- an archive used on its own, through `add`, with no history.

## The lineage as git

Following the systems above, the git repository holds **the kept lineage, and only the genome
files**:

| the run | the repository |
| --- | --- |
| an individual the archive admitted | a commit; its files are the genome, nothing else |
| its parents | the commit's parents, if they are in the repository |
| its id | the tag `<id>`: `git show c98a41a261:main.py` |
| score, summary, feedback | the commit message; score and id as trailers |
| the archive's current members | `refs/archive/<id>` |
| the archive's best | `refs/heads/best`, and `HEAD` |

What is not in it:

- **Rejected and failed attempts.** They stay in the history, like `results.tsv` and
  `trajectory.jsonl`, and are read with `evokit list-events`. An admitted individual that the archive
  later drops (a population or island archive) keeps its commit, and only loses its
  `refs/archive/` ref.
- **The task's fixed files** (`evaluate.py`, `TASK.md`, `config.yaml`), artifacts, transcripts, and
  costs.
- **Score files in the tree.** The score is in the commit message, so `git log` shows it, and the
  tree is exactly the genome.

A commit:

```text
Optimize centers and radii with SLSQP from a 5-5-6-5-5 start

Feedback:
valid packing, sum of radii 2.25933

Evokit-Id: c98a41a261
Score: 2.25933
Operator: coding-agent
```

The subject is the agent's `SUMMARY:` line ([avo.md](avo.md), change 5). Without one, it is the
first line of its summary, or `seed`. `Score:` lists every objective (`score=2.25933,
size=140` for more than one). The prototype built from `runs/avo-2` showed what goes wrong
without that line: subjects like "2.6245! This matches my best score from earlier."

### When it is written: in `tell`, for every run

Every algorithm records its evaluations through `tell`: the recipes, `evokit run`, and the
CLI's `evaluate-directory`. So `tell` is the one place to update the repository, and every
algorithm gets it:

```python
# evokit/loop.py
async def tell(storage, archive, individuals):
    admitted = []
    for individual in individuals:
        ...
        history.record_evaluated(storage, individual, added=added)
        ...
    write_lineage(storage, archive)           # commits for the newly admitted, then the refs
    return admitted
```

Each call writes one commit per newly admitted individual and moves the refs, a few
milliseconds of work. A run whose genomes should not be in git (model checkpoints) turns it
off with `tell(..., lineage=False)`. Recipes take the same keyword and pass it down.

### Where it lives

Storage stays bytes at paths, local or remote. The repository is derived from it and lives on
the local disk, where `git` runs. `Storage` gets one method for that:

```python
# evokit/storage.py
class Storage(ABC):
    @abstractmethod
    def cache(self, name: str) -> Path:
        """A local folder for data derived from the run: rebuildable, never written back."""
```

- `DirectoryStorage("runs/avo-2").cache("lineage.git")` is `runs/avo-2/lineage.git`.
- `MemoryStorage` and a future S3 storage return a folder in the local cache directory, keyed by
  the run. Machines that share a run each build their own copy. The commits are a pure function
  of the history (fixed author, record times), so every copy has the same hashes.

The repository contains a `.gitignore` with `*`, so a project repository that contains the run
directory does not pick it up with `git add .`.

### The Python API

```python
# evokit/lineage.py
@dataclass(frozen=True)
class LineageRepository:
    directory: Path
    added: int                  # commits written by this call
    commits: int                # admitted individuals in the repository
    best: str | None            # the id `best` points at


def write_lineage(storage: Storage, archive: Archive, directory: str | Path | None = None) -> LineageRepository:
    """Bring the lineage repository up to date with the run (default: storage.cache("lineage.git")):
    a commit for each admitted individual not yet in it, then refs/archive/* and `best`.
    Creates it if needed. Calling it again with nothing new writes nothing."""
```

```python
import evokit as ek

repo = ek.write_lineage(storage, archive)                    # what tell calls
repo = ek.write_lineage(storage, archive, "/tmp/avo-2.git")  # an export elsewhere
```

It is written with dulwich (pure Python git), so writing it needs no `git` binary, and dulwich
becomes a dependency of evokit. Reading it needs ordinary `git`. The core loop was prototyped
against `runs/avo-2` (`/tmp/write_git.py`). It wrote 4 commits, wrote 0 when run again, and
gave identical hashes when rebuilt from nothing.

### The CLI

One command, generated from a `RunCommands` method, for runs made before this existed, a
deleted repository, or an export:

```python
# evokit/interface/run.py
@command
def write_lineage(self, directory: str | None = None) -> LineageRepository:
    """Bring this run's lineage repository up to date (or write it to `directory`) and print where it is."""
    return write_lineage(self.storage, self.archive, directory)
```

```bash
evokit write-lineage --dir runs/avo-2
# {"directory": ".../runs/avo-2/lineage.git", "added": 0, "commits": 4, "best": "1f7ec51182"}
```

Reading is plain git:

```bash
git -C runs/avo-2/lineage.git log --oneline --graph --all        # the kept lineage
git -C runs/avo-2/lineage.git log best --format='%h %(trailers:key=Score,valueonly)'
git -C runs/avo-2/lineage.git show best:main.py
git -C runs/avo-2/lineage.git diff c98a41a261 375411150b
git -C runs/avo-2/lineage.git log --all -S 'minimize('           # versions whose code added or removed it
```

### What the agent gets

`CodingAgentOperator` puts two lines in the prompt. It does not write the repository, because
`tell` already has, and the agent only reads it:

```text
The versions kept so far are a git repository at {repository} (read-only):
  git -C {repository} log --all --oneline         git -C {repository} show <id>:main.py
  git -C {repository} diff <id> <id>              git -C {repository} log --all -S '<code>'
```

The agent's workspace stays a plain copy of the parent. A git workspace (a checkout of the
parent's commit) would add `git diff` against the parent. It also brings the questions of what
`git status` shows and what the agent may commit, and none of the systems above needs it.

## What changes, in order

1. **`Storage.cache`** for `DirectoryStorage` and `MemoryStorage`.
2. **`write_lineage`**, called from `tell`, plus `evokit write-lineage`. dulwich becomes a dependency.
3. **The prompt lines** in `CodingAgentOperator`, and the `SUMMARY:` line as the commit subject.

## Tests

- After a small run, the repository has one commit per admitted individual, with their parents.
  Rejected ones have no commit. `refs/archive/*` matches `archive.members()`, and `best` matches
  `archive.best()`.
- An island archive that drops a member keeps its commit and removes its `refs/archive/` ref.
- Calling `write_lineage` again with nothing new writes nothing. After deleting the repository, a
  rebuild has the same hashes.
- A commit's tree has exactly the genome files.
- An enclosing project repository does not see the lineage repository (`git status --porcelain`
  in it is empty).
- `evokit write-lineage --dir RUN` prints the repository, and `git log` on it lists every kept
  individual.
