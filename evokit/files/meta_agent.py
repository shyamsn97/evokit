"""MetaAgentModify: Hyperagents' operator, where the improvement procedure is part of the genome."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

from evokit.errors import OperatorError
from evokit.files.agents import CommandAgent
from evokit.files.tree import Files
from evokit.operator import Operator
from evokit.records import Individual, Offspring, Selection
from evokit.storage import Storage
from evokit.util import title_and_body, truncate


class MetaAgentModify(Operator):
    """The genome is one repository: task_agent.py solves the task, meta_agent.py improves the repository.

    The parent's meta agent edits a copy of the whole repository, itself included. It is run as
    `python meta_agent.py --repo REPO --history HISTORY`, where HISTORY holds `archive.jsonl`
    (every archived agent's lineage and score) and `parent_logs/` (the parent's evaluation
    logs). With `fixed_meta_agent`, that meta agent edits every parent instead (the ablation
    without metacognitive self-modification).
    """

    name = "meta-agent"

    def __init__(
        self,
        storage: Storage,
        *,
        entrypoint: str = "meta_agent.py",
        logs: str = "logs",
        fixed_meta_agent: Files | None = None,
        objective: str = "score",
        timeout: str | float | None = "30m",
    ):
        self.storage, self.entrypoint, self.logs = storage, entrypoint, logs
        self.fixed_meta_agent, self.objective, self.timeout = fixed_meta_agent, objective, timeout

    async def create(self, selection: Selection, archive) -> Offspring:
        if not selection.parents:
            raise OperatorError("MetaAgentModify needs a parent")
        parent = selection.parents[0]
        root = Path(tempfile.mkdtemp(prefix="evokit-hyperagent-"))
        try:
            parent.genome.write_to(root / "repo")
            with self.storage.local(f"artifacts/{parent.id}") as artifacts:
                write_history(root / "history", archive, artifacts / self.logs, self.objective)
            meta = root / "repo"
            if self.fixed_meta_agent is not None:
                meta = root / "meta"
                self.fixed_meta_agent.write_to(meta)
            argv = [sys.executable, str(meta / self.entrypoint), "--repo", str(root / "repo"), "--history", str(root / "history")]
            result = await CommandAgent(argv, self.timeout).run(root / "repo", "")
            if result.error:
                raise OperatorError(f"the meta agent failed: {result.error}", result.usage)
            child = Files.from_directory(root / "repo")
        finally:
            shutil.rmtree(root, ignore_errors=True)
        if child == parent.genome:
            raise OperatorError("the meta agent changed nothing", result.usage)
        name, description = title_and_body(result.summary)
        child = Individual.new(child, [parent.id], {"operator": self.name}, name=name, description=truncate(description, 2000))
        return Offspring((child,), result.usage)


def write_history(directory: Path, archive, parent_logs: Path, objective: str = "score") -> None:
    """What the meta agent can read: every archived agent's lineage and score, and the parent's logs."""
    directory.mkdir(parents=True)
    with open(directory / "archive.jsonl", "w") as f:
        for member in archive.members():
            f.write(json.dumps({
                "id": member.id,
                "parents": list(member.parents),
                "score": member.objective(objective),
                "name": member.name,
                "description": member.description,
            }) + "\n")
    if parent_logs.is_dir():
        shutil.copytree(parent_logs, directory / "parent_logs")
