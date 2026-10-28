"""Staged evaluation: score an agent repository by letting it solve circle-packing tasks."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import statistics
import sys
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path

import evokit as ek
from evokit.operators import script_agent
from evokit.files.evaluator import read_folder
from evokit.records import merge_usage
from evokit.util import truncate

TASK_DIR = Path(__file__).resolve().parent / "task"
REFERENCES = {int(n): v for n, v in json.loads((TASK_DIR / "references.json").read_text()).items() if n.isdigit()}

TASK_PROMPT = """Edit main.py in this directory so that construct_packing() places {n} non-overlapping circles \
inside the unit square with the largest possible sum of radii. It must return (centers, radii) with {n} entries each. \
Check your work with `python evaluate.py`: it prints the score, or why the packing is invalid. Do not edit evaluate.py."""

INSTRUCTIONS = """The benchmark asks you to edit main.py so that it packs N circles (N = 20, 26, 32) in the \
unit square with the largest sum of radii. A score of 1.0 means the best known packings, averaged over tasks."""


@dataclass
class Solve:
    n: int
    valid: bool
    radii_sum: float = 0.0
    problem: str = ""
    main: str = ""
    usage: dict = field(default_factory=dict)

    @property
    def normalized(self) -> float:
        return self.radii_sum / REFERENCES[self.n] if self.valid else 0.0


class PackingBenchmark:
    """An eval function for agents, in three stages.

    1. One N=26 task. An agent that crashes, changes nothing, or leaves an invalid packing has
       lost its editing ability: no objectives, so the archive never admits it.
    2. N in `sizes`; the score is the mean normalized sum of radii.
    3. Only for agents that beat the seed agent's stage 2 score: `repeats` more rounds, averaged.

    Every task runs in a fresh copy with a fresh evaluate.py, so an agent cannot change how it
    is scored. The evaluation's logs hold `logs/` (what self_modify reads) and the best N=26
    packing as `best_main.py`. The seed agent's stage 2 score is read from the experiment in `dir`.
    """

    def __init__(self, dir: str | Path, sizes=(20, 26, 32), repeats: int = 2, timeout: str = "15m"):
        self.dir, self.sizes, self.repeats, self.timeout = Path(dir), tuple(sizes), repeats, timeout

    def seed_score(self) -> float | None:
        if not (self.dir / "history" / "records.jsonl").is_file():
            return None
        return next(
            (i.objective("stage2") for i in ek.open_experiment(self.dir).evaluated() if not i.parents and i.objective("stage2") is not None),
            None,
        )

    async def __call__(self, individual: ek.Individual) -> ek.Evaluation:
        workdir = Path(tempfile.mkdtemp(prefix="dgm-eval-"))
        try:
            evaluation = await self._evaluate(individual, workdir)
            return replace(evaluation, logs=read_folder(workdir))
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    async def _evaluate(self, individual: ek.Individual, workdir: Path) -> ek.Evaluation:
        logs = workdir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        agent_dir = Path(tempfile.mkdtemp(prefix="dgm-agent-"))
        try:
            individual.genome.write_to(agent_dir)
            solves = [await self._solve(agent_dir, 26, logs / "stage1-n26.md")]
            if not solves[0].valid:
                return ek.Evaluation(feedback=f"lost editing ability: {solves[0].problem}", usage=solves[0].usage)
            solves += await self._round(agent_dir, [n for n in self.sizes if n != 26], logs, "stage2")
            stage2 = statistics.mean(s.normalized for s in solves)
            bar = self.seed_score() if individual.parents else None
            if bar is not None and stage2 > bar:
                for k in range(self.repeats):
                    solves += await self._round(agent_dir, self.sizes, logs, f"stage3-{k + 1}")
        finally:
            shutil.rmtree(agent_dir, ignore_errors=True)
        best = max((s for s in solves if s.n == 26 and s.valid), key=lambda s: s.radii_sum)
        (workdir / "best_main.py").write_text(best.main)
        return ek.Evaluation(
            {
                "score": statistics.mean(s.normalized for s in solves),
                "stage2": stage2,
                "best_sum_26": best.radii_sum,
                "solves": len(solves),
            },
            feedback="\n".join(self._describe(s) for s in solves),
            usage=merge_usage(*(s.usage for s in solves)),
        )

    async def _round(self, agent_dir: Path, sizes, logs: Path, label: str) -> list[Solve]:
        return list(await asyncio.gather(*(self._solve(agent_dir, n, logs / f"{label}-n{n}.md") for n in sizes)))

    async def _solve(self, agent_dir: Path, n: int, log: Path) -> Solve:
        work = Path(tempfile.mkdtemp(prefix="dgm-task-"))
        try:
            shutil.copy(TASK_DIR / "main.py", work / "main.py")
            shutil.copy(TASK_DIR / "evaluate.py", work / "evaluate.py")
            prompt = TASK_PROMPT.format(n=n)
            async with script_agent(agent_dir, timeout=self.timeout, env={"PACKING_N": str(n)}).session(work) as session:
                result = await session.prompt(prompt)
            shutil.copy(TASK_DIR / "evaluate.py", work / "evaluate.py")
            main = (work / "main.py").read_text() if (work / "main.py").exists() else ""
            report = await self._score(work, n)
            solve = Solve(n, bool(report.get("valid")), float(report.get("score") or 0), main=main, usage=result.usage)
            if result.error:
                solve.valid, solve.problem = False, f"the agent failed: {result.error}"
            elif main == (TASK_DIR / "main.py").read_text():
                solve.valid, solve.problem = False, "the agent did not change main.py"
            elif not solve.valid:
                solve.problem = f"invalid packing: {report.get('feedback')}"
            log.write_text(
                f"# Task (N={n})\n\n{prompt}\n\n# Agent summary\n\n{result.text}\n\n"
                f"# Agent error\n\n{result.error or 'none'}\n\n# Agent stderr\n\n{truncate(result.trace or '', 8000)}\n\n"
                f"# evaluate.py\n\n{json.dumps(report)}\n\n# main.py\n\n```python\n{main}\n```\n"
            )
            return solve
        finally:
            shutil.rmtree(work, ignore_errors=True)

    async def _score(self, work: Path, n: int) -> dict:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "evaluate.py", cwd=work, env={**os.environ, "PACKING_N": str(n)},
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), 120)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return {"valid": False, "feedback": "evaluate.py took longer than 120 s"}
        try:
            return json.loads(out.decode().strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            return {"valid": False, "feedback": "evaluate.py printed no result"}

    @staticmethod
    def _describe(s: Solve) -> str:
        if not s.valid:
            return f"N={s.n}: {s.problem}"
        return f"N={s.n}: sum of radii {s.radii_sum:.4f} ({s.normalized:.3f} of the best known)"
