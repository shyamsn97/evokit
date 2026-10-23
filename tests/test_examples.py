"""The circle-packing algorithm examples, run with scripted agents and models instead of LLMs."""

import asyncio
import http.server
import json
import re
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from evokit import history
from evokit.files import AgentResult, CodingAgent

CIRCLE_PACKING = Path(__file__).resolve().parents[1] / "examples" / "circle_packing"
sys.path.insert(0, str(CIRCLE_PACKING))


def widen_outer_ring(text: str, step: float = 0.005) -> str:
    """A small change that raises the seed's score (up to a point)."""
    return re.sub(r"0\.5 \+ (0\.\d+) \* math\.cos\(a\), 0\.5 \+ \1 \* math\.sin\(a\)\)\)\n    centers = ",
                  lambda m: m.group(0).replace(m.group(1), f"{float(m.group(1)) + step:.3f}"), text)


class ScriptedAgent(CodingAgent):
    """Scores a draft if it can, widens the outer ring, and reports a cost."""

    def __init__(self):
        self.prompts, self.sessions, self.drafts = [], [], []

    async def run(self, directory, prompt, session=None, env=None):
        self.prompts.append(prompt)
        self.sessions.append(session)
        if env and env.get("EVOKIT_DRAFT_EVAL"):
            proc = await asyncio.to_thread(subprocess.run, [env["EVOKIT_DRAFT_EVAL"]], capture_output=True, text=True)
            self.drafts.append(proc.stdout)
        main = Path(directory) / "main.py"
        main.write_text(widen_outer_ring(main.read_text()))
        return AgentResult("widened the outer ring", usage={"cost_usd": 0.01}, session="session-1")


def test_autoresearch_unattended(tmp_path):
    from algorithms import autoresearch

    storage, archive = asyncio.run(autoresearch.run(str(tmp_path / "run"), agent=ScriptedAgent(), max_evaluations=3))
    best = archive.best()
    assert best.objective("score") > 1.61 and best.objective("lines") == 28
    assert history.usage(storage)["cost_usd"] == pytest.approx(0.03)
    rows = storage.read("results.tsv").decode().splitlines()
    assert len(rows) == 3 and rows[0].split("\t")[2] == "keep"


class CommittingAgent(CodingAgent):
    """AVO's agent, scripted: each session drafts and commits twice, a better version and then a
    tie (one more blank line), and stops when a reply says so. As the supervisor, it names a direction."""

    def __init__(self):
        self.prompts, self.sessions, self.replies, self.supervisor_prompts = [], [], [], []

    async def call(self, script, *args):
        proc = await asyncio.to_thread(subprocess.run, [script, *args], capture_output=True, text=True)
        reply = json.loads(proc.stdout)
        self.replies.append(reply)
        return reply

    async def run(self, directory, prompt, session=None, env=None):
        if "EVOKIT_COMMIT" not in env:
            self.supervisor_prompts.append(prompt)
            return AgentResult("1. Try a hexagonal outer ring.", usage={"cost_usd": 0.01})
        self.prompts.append(prompt)
        self.sessions.append(session)
        main = Path(directory) / "main.py"
        for change in (widen_outer_ring, lambda text: text + "\n"):
            main.write_text(change(main.read_text()))
            draft = await self.call(env["EVOKIT_DRAFT_EVAL"])
            if "stop" in draft:
                break
            if "stop" in await self.call(env["EVOKIT_COMMIT"], "a change", f"draft score {draft['objectives']}"):
                break
        return AgentResult("done for now", usage={"cost_usd": 0.01}, session="session-1")


def test_avo_one_session_commits_and_a_supervisor_redirects(tmp_path):
    from algorithms import avo

    agent = CommittingAgent()
    storage, archive = asyncio.run(avo.run(str(tmp_path / "run"), agent=agent, max_evaluations=11, stall_after=6))
    commits = [r for r in agent.replies if "kept" in r]
    kept = [r["kept"] for r in commits]
    assert kept == [True] * 6 + [False] * 4  # widening helps three times, then regresses; ties are kept
    assert agent.sessions == [None] + ["session-1"] * 4
    assert "evokit log --all" in agent.prompts[0] and "Continue the loop" in agent.prompts[1]
    assert archive.best().objective("score") > 1.61
    (review,) = agent.supervisor_prompts
    assert "last\n6 attempts" in review and "recent drafts (oldest first): score=" in review
    assert "+++ workspace" in review
    assert history.counts(storage)["redirect"] == 1
    delivered = [str(r.get("directions")) for r in agent.replies] + agent.prompts
    assert sum("hexagonal" in d for d in delivered) == 1
    assert agent.replies[-1]["stop"] == "11 evaluations"
    assert history.usage(storage)["draft_evals"] == 10
    assert history.usage(storage)["cost_usd"] == pytest.approx(0.06)
    assert history.individual(storage, commits[0]["id"]).name == "a change"


class MeteredAgent(CodingAgent):
    """Reports its cost as it goes and never finishes on its own: the session commits a tie (a stall,
    so the supervisor starts) and reports $0.40; the supervisor reports $0.70."""

    def __init__(self, on_cost=None):
        self.on_cost, self.replies = on_cost, []

    def with_meter(self, on_cost):
        agent = MeteredAgent(on_cost)
        agent.replies = self.replies
        return agent

    async def run(self, directory, prompt, session=None, env=None):
        if "EVOKIT_COMMIT" in env:
            main = Path(directory) / "main.py"
            main.write_text(main.read_text() + "\n")
            proc = await asyncio.to_thread(subprocess.run, [env["EVOKIT_COMMIT"], "a tie"], capture_output=True, text=True)
            self.replies.append(json.loads(proc.stdout))
            await asyncio.sleep(0.2)
            self.on_cost(0.4)
        else:
            self.on_cost(0.7)
        await asyncio.sleep(600)


def test_avo_session_and_supervisor_share_the_budget(tmp_path, monkeypatch):
    import importlib

    from algorithms import avo

    monkeypatch.setattr(importlib.import_module("evokit.recipes.avo"), "STOP_GRACE", 0.2)
    agent = MeteredAgent()
    storage, _ = asyncio.run(avo.run(str(tmp_path / "run"), agent=agent, max_cost=1.0, stall_after=1))
    assert agent.replies[0]["kept"] and agent.replies[0]["budget_left_usd"] == 1.0
    notes = {r["note"]: r for r in history.records(storage) if r["kind"] == "note"}
    assert notes["stop"]["text"] == "$1 spent"
    assert notes["session"]["usage"]["estimated_cost_usd"] == pytest.approx(0.4)
    assert notes["supervisor-failed"]["usage"]["estimated_cost_usd"] == pytest.approx(0.7)
    assert history.usage(storage)["cost_usd"] == pytest.approx(1.1)


class FakeMessagesAPI(http.server.BaseHTTPRequestHandler):
    """The Anthropic Messages API, scripted: one tool call per task, then a summary."""

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        task = body["messages"][0]["content"]
        if len(body["messages"]) > 1:
            content = [{"type": "text", "text": "done"}]
        elif "proposal.md" in task:
            edit = {"path": "proposal.md", "old": "", "new": "Tell the model to check its work twice."}
            content = [{"type": "tool_use", "id": "t1", "name": "edit_file", "input": edit}]
        elif "Implement this improvement" in task:
            command = "printf '\\nSYSTEM += \"\\\\nCheck your work twice.\"\\n' >> prompts.py"
            content = [{"type": "tool_use", "id": "t1", "name": "bash", "input": {"command": command}}]
        elif "Check your work twice" in body["system"]:
            old = "0.42 * math.cos(a), 0.5 + 0.42 * math.sin(a)"
            edit = {"path": "main.py", "old": old, "new": old.replace("0.42", "0.43")}
            content = [{"type": "tool_use", "id": "t1", "name": "edit_file", "input": edit}]
        else:
            edit = {"path": "main.py", "old": "0.42 * math.cos", "new": "0.43 * math.cos"}
            content = [{"type": "tool_use", "id": "t1", "name": "edit_file", "input": edit}]
        reply = json.dumps({"content": content, "usage": {"input_tokens": 1000, "output_tokens": 100}}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, *args):
        pass


@pytest.fixture
def fake_anthropic(monkeypatch):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeMessagesAPI)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("ANTHROPIC_BASE_URL", f"http://127.0.0.1:{server.server_address[1]}")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    yield
    server.shutdown()


@pytest.mark.parametrize("variant", ["full", "no-self-improve"])
def test_dgm_agents_modify_themselves(tmp_path, fake_anthropic, variant):
    from algorithms.dgm import main as dgm

    directory = tmp_path / "run"
    storage, archive = asyncio.run(dgm.run(str(directory), variant, workers=2, max_children=2))
    seed, *children = sorted(history.evaluated(storage), key=lambda i: len(i.parents))
    assert seed.objective("solves") == 3 and seed.objective("stage2") > 0.6
    assert storage.exists(f"artifacts/{seed.id}/logs/stage1-n26.md")
    assert len(children) == 2 and all(c.parents == (seed.id,) and c.evaluation.objectives for c in children)
    assert all("Check your work twice" in c.genome.text("prompts.py") for c in children)
    assert all("check its work twice" in c.info["proposal"] for c in children)
    # the improved agents beat the seed in stage 2, so stage 3 runs both sizes twice more
    assert all(c.objective("stage2") > seed.objective("stage2") for c in children)
    assert all(c.objective("solves") == 9 for c in children)
    # solves, plus a diagnosis and an implementation per child; two $0.0045 calls each
    assert history.usage(storage)["cost_usd"] == pytest.approx(0.009 * (3 + 2 * 9 + 4))
    dgm.report(storage, archive)
    assert "0.43 * math.sin" in (directory / "best_main.py").read_text()
    assert (directory / "best_agent" / "agent.py").is_file()


def ring_diff_model(seed, cost, prompts):
    """A model that nudges the outer ring of the current program, as a SEARCH/REPLACE diff."""
    import random

    from evokit.operators import Completion, FunctionModel

    rng = random.Random(seed)

    def reply(prompt):
        prompts.append(prompt)
        current = prompt.split("Current program:", 1)[1]
        line = re.search(r"^.*0\.5 \+ (0\.\d+) \* math\.cos\(a\).*$\n    centers = ", current, re.M)
        old = line.group(0).split("\n")[0]
        value = float(line.group(1)) + rng.choice([-0.004, 0.003, 0.006])
        new = old.replace(line.group(1), f"{value:.3f}")
        if rng.random() < 0.3:
            new += "\n        pass" * 16
        return Completion(f"main.py\n<<<<<<< SEARCH\n{old}\n=======\n{new}\n>>>>>>> REPLACE\n", {"cost_usd": cost})

    return FunctionModel(reply)


def test_openevolve_islands_inspirations_and_ensemble(tmp_path):
    from algorithms import openevolve

    prompts = []
    models = {"fast": ring_diff_model(1, 0.001, prompts), "strong": ring_diff_model(2, 0.01, prompts)}
    storage, archive = asyncio.run(openevolve.run(str(tmp_path / "run"), "full", models=models, workers=4, max_evaluations=40))
    best = archive.best()
    assert best.objective("score") > 1.65 and best.objective("lines") is not None
    assert all(archive.island_members(i) for i in range(4))
    assert any(len({key for key in archive.niches() if key[0] == i}) > 1 for i in range(4))
    assert {i.info.get("operator", "seed") for i in history.evaluated(storage)} == {"seed", "fast", "strong"}
    assert sum("Other programs, for inspiration" in p for p in prompts) > len(prompts) / 2
    assert all("EVOLVE-BLOCK" in p for p in prompts)
    print(archive.table(0))


@pytest.mark.parametrize("variant", ["one-island", "no-inspirations", "single-model", "no-grid"])
def test_openevolve_variants_run(tmp_path, variant):
    from algorithms import openevolve

    prompts = []
    models = {"fast": ring_diff_model(1, 0.001, prompts), "strong": ring_diff_model(2, 0.01, prompts)}
    storage, archive = asyncio.run(openevolve.run(str(tmp_path / "run"), variant, models=models, workers=4, max_evaluations=12))
    assert archive.best().objective("score") > 1.61
    islands = sum(1 for i in range(archive.islands) if archive.island_members(i))
    assert islands == (1 if variant == "one-island" else 4)
    if variant == "no-inspirations":
        assert not any("Other programs, for inspiration" in p for p in prompts)
    if variant == "no-grid":
        assert {cell for _, cell in archive.niches()} == {()}


def test_openevolve_rejects_edits_outside_the_block(tmp_path):
    from algorithms import openevolve

    from evokit.operators import Completion, FunctionModel

    outside = Completion("main.py\n<<<<<<< SEARCH\nN = 26\n=======\nN = 25\n>>>>>>> REPLACE\n", {"cost_usd": 0.01})
    model = FunctionModel(lambda p: outside)
    storage, _ = asyncio.run(openevolve.run(str(tmp_path / "run"), "single-model", models={"fast": model}, workers=1, max_cost=0.1))
    failed = [r for r in history.records(storage) if r["kind"] == "note" and r["note"] == "failed"]
    assert len(failed) >= 10 and "outside its EVOLVE-BLOCK" in failed[0]["text"]
    assert history.usage(storage)["cost_usd"] == pytest.approx(0.01 * len(failed))
    assert history.counts(storage)["evaluated"] == 1

