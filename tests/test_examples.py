"""The circle-packing algorithm examples, run with scripted agents and models instead of LLMs."""

import asyncio
import functools
import http.server
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from evokit.history import History
from evokit.agents import AgentResult, CodingAgent


@functools.cache
def history_of(storage) -> History:
    """One History per storage, as a process keeps one per run."""
    return History(storage)


CIRCLE_PACKING = Path(__file__).resolve().parents[1] / "examples" / "circle_packing"
sys.path.insert(0, str(CIRCLE_PACKING))


SEED_SCORE = 0.9598
RING = r"0\.5 \+ (0\.\d+) \* np\.cos\(angle\), 0\.5 \+ \1 \* np\.sin\(angle\)"


def widen_inner_ring(text: str, step: float = 0.03) -> str:
    """A change that raises the seed's score three times, then lowers it."""
    return re.sub(RING, lambda m: m.group(0).replace(m.group(1), f"{float(m.group(1)) + step:.3f}"), text, count=1)


async def evokit(env, *args):
    """Run an agent tool the way a shell agent does, and read its JSON reply."""
    proc = await asyncio.to_thread(subprocess.run, ["evokit", *args], capture_output=True, text=True, env={**os.environ, **env})
    return json.loads(proc.stdout if proc.returncode == 0 else proc.stderr)


class ScriptedAgent(CodingAgent):
    """Widens the inner ring and reports a cost."""

    def __init__(self):
        self.prompts, self.sessions = [], []

    async def _run(self, directory, prompt, session=None, env=None):
        self.prompts.append(prompt)
        self.sessions.append(session)
        main = Path(directory) / "main.py"
        main.write_text(widen_inner_ring(main.read_text()))
        return AgentResult("widened the inner ring", usage={"cost_usd": 0.01}, session="session-1")


def test_autoresearch_unattended(tmp_path):
    from algorithms import autoresearch

    agent = ScriptedAgent()
    storage, archive = asyncio.run(autoresearch.run(str(tmp_path / "run"), agent=agent, max_evaluations=3))
    best = archive.best()
    assert best.objective("score") > SEED_SCORE and best.objective("lines") == 55
    assert history_of(storage).usage()["cost_usd"] == pytest.approx(0.03)
    assert "evokit log" in agent.prompts[0] and "evokit score" not in agent.prompts[0]
    rows = storage.read("results.tsv").decode().splitlines()
    assert len(rows) == 3 and rows[0].split("\t")[2] == "keep"


class SubmittingAgent(CodingAgent):
    """AVO's agent, scripted: each session scores and submits twice, a better version and then a
    tie (one more blank line), and stops when a reply says so. As the supervisor, it names a direction."""

    def __init__(self):
        self.prompts, self.sessions, self.replies, self.supervisor_prompts = [], [], [], []

    async def call(self, env, *args):
        reply = await evokit(env, *args)
        self.replies.append(reply)
        return reply

    async def _run(self, directory, prompt, session=None, env=None):
        if "evokit submit" not in prompt:
            self.supervisor_prompts.append(prompt)
            return AgentResult("1. Try a staggered grid.", usage={"cost_usd": 0.01})
        self.prompts.append(prompt)
        self.sessions.append(session)
        main = Path(directory) / "main.py"
        for change in (widen_inner_ring, lambda text: text + "\n"):
            main.write_text(change(main.read_text()))
            draft = await self.call(env, "score")
            if "stop" in draft:
                break
            if "stop" in await self.call(env, "submit", "a change", f"draft score {draft['objectives']}"):
                break
        return AgentResult("done for now", usage={"cost_usd": 0.01}, session="session-1")


def test_avo_one_session_submits_and_a_supervisor_redirects(tmp_path):
    from algorithms import avo

    agent = SubmittingAgent()
    storage, archive = asyncio.run(avo.run(str(tmp_path / "run"), agent=agent, max_evaluations=11, stall_after=6))
    submitted = [r for r in agent.replies if "kept" in r]
    kept = [r["kept"] for r in submitted]
    assert kept == [True] * 6 + [False] * 4  # widening helps three times, then regresses; ties are kept
    assert agent.sessions == [None] + ["session-1"] * 4
    assert "evokit log [ID] [--all]" in agent.prompts[0] and "Continue the loop" in agent.prompts[1]
    assert archive.best().objective("score") > SEED_SCORE
    (review,) = agent.supervisor_prompts
    assert "last\n6 attempts" in review and "recent drafts (oldest first): score=" in review
    assert "+++ workspace" in review and "evokit show ID" in review and "evokit score" not in review
    assert history_of(storage).counts()["redirect"] == 1
    delivered = [str(r.get("message")) for r in agent.replies] + agent.prompts
    assert sum("staggered grid" in d for d in delivered) == 1
    assert agent.replies[-1]["stop"] == "11 evaluations"
    assert history_of(storage).usage()["draft_evals"] == 10
    assert history_of(storage).usage()["cost_usd"] == pytest.approx(0.06)
    assert history_of(storage).individual(submitted[0]["id"]).name == "a change"


class StallingAgent(CodingAgent):
    """Submits a tie, then never finishes on its own; keeps the budgets it is given."""

    def __init__(self):
        self.budgets, self.replies = [], []

    def with_budget(self, usd):
        self.budgets.append(usd)
        return self

    async def _run(self, directory, prompt, session=None, env=None):
        main = Path(directory) / "main.py"
        main.write_text(main.read_text() + "\n")
        self.replies.append(await evokit(env, "submit", "a tie"))
        await asyncio.sleep(600)


def test_avo_stops_a_session_once_the_run_is_over(tmp_path, monkeypatch):
    import evokit.agents.base
    from algorithms import avo

    monkeypatch.setattr(evokit.agents.base, "STOP_GRACE", 0.2)
    agent = StallingAgent()
    storage, _ = asyncio.run(avo.run(str(tmp_path / "run"), agent=agent, max_cost=1.0, max_evaluations=2, stall_after=None))
    assert agent.replies == [{**agent.replies[0], "kept": True, "stop": "2 evaluations"}]
    assert agent.budgets == [1.0]
    notes = {r["note"]: r for r in history_of(storage).records() if r["kind"] == "note"}
    assert notes["stop"]["text"] == "2 evaluations" and notes["session"]["text"].startswith("1 submitted")


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
    seed, *children = sorted(history_of(storage).evaluated(), key=lambda i: len(i.parents))
    assert seed.objective("solves") == 3 and seed.objective("stage2") > 0.6
    assert storage.exists(f"artifacts/{seed.id}/logs/stage1-n26.md")
    assert len(children) == 2 and all(c.parents == (seed.id,) and c.evaluation.objectives for c in children)
    assert all("Check your work twice" in c.genome.text("prompts.py") for c in children)
    assert all("check its work twice" in c.info["proposal"] for c in children)
    # the improved agents beat the seed in stage 2, so stage 3 runs both sizes twice more
    assert all(c.objective("stage2") > seed.objective("stage2") for c in children)
    assert all(c.objective("solves") == 9 for c in children)
    # solves, plus a diagnosis and an implementation per child; two $0.0045 calls each
    assert history_of(storage).usage()["cost_usd"] == pytest.approx(0.009 * (3 + 2 * 9 + 4))
    dgm.report(storage, archive)
    assert "0.43 * math.sin" in (directory / "best_main.py").read_text()
    assert (directory / "best_agent" / "agent.py").is_file()


def ring_diff_model(seed, cost, prompts):
    """A model that nudges the inner ring of the current program, as a SEARCH/REPLACE diff."""
    import random

    from evokit.operators import Completion, FunctionModel

    rng = random.Random(seed)

    def reply(prompt):
        prompts.append(prompt)
        current = prompt.split("Current program:", 1)[1]
        line = re.search(rf"^.*{RING}.*$", current, re.M)
        old = line.group(0)
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
    assert best.objective("score") > SEED_SCORE + 0.01 and best.objective("lines") is not None
    assert all(archive.island_members(i) for i in range(4))
    assert any(len({key for key in archive.niches() if key[0] == i}) > 1 for i in range(4))
    assert {i.info.get("operator", "seed") for i in history_of(storage).evaluated()} == {"seed", "fast", "strong"}
    assert sum("Other programs, for inspiration" in p for p in prompts) > len(prompts) / 2
    assert all("EVOLVE-BLOCK" in p for p in prompts)
    print(archive.table(0))


@pytest.mark.parametrize("variant", ["one-island", "no-inspirations", "single-model", "no-grid"])
def test_openevolve_variants_run(tmp_path, variant):
    from algorithms import openevolve

    prompts = []
    models = {"fast": ring_diff_model(1, 0.001, prompts), "strong": ring_diff_model(2, 0.01, prompts)}
    storage, archive = asyncio.run(openevolve.run(str(tmp_path / "run"), variant, models=models, workers=4, max_evaluations=12))
    assert archive.best().objective("score") > SEED_SCORE
    islands = sum(1 for i in range(archive.islands) if archive.island_members(i))
    assert islands == (1 if variant == "one-island" else 4)
    if variant == "no-inspirations":
        assert not any("Other programs, for inspiration" in p for p in prompts)
    if variant == "no-grid":
        assert {cell for _, cell in archive.niches()} == {()}


def test_openevolve_rejects_edits_outside_the_block(tmp_path):
    from algorithms import openevolve

    from evokit.operators import Completion, FunctionModel

    edit = "    sum_radii = np.sum(radii)\n=======\n    sum_radii = 3.0\n"
    outside = Completion(f"main.py\n<<<<<<< SEARCH\n{edit}>>>>>>> REPLACE\n", {"cost_usd": 0.01})
    model = FunctionModel(lambda p: outside)
    storage, _ = asyncio.run(openevolve.run(str(tmp_path / "run"), "single-model", models={"fast": model}, workers=1, max_cost=0.1))
    failed = [r for r in history_of(storage).records() if r["kind"] == "note" and r["note"] == "failed"]
    assert len(failed) >= 10 and "outside its EVOLVE-BLOCK" in failed[0]["text"]
    assert history_of(storage).usage()["cost_usd"] == pytest.approx(0.01 * len(failed))
    assert history_of(storage).counts()["evaluated"] == 1

