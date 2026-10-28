"""experiment.start(agent): one conversation in a checkout, with the experiment's tools, recorded turns,
messages, resuming, read-only helpers, and agents working in parallel."""

import asyncio
import json
import sys

import pytest

import evokit as ek
from evokit.agents import CommandAgent
from evokit.files import Project, ScriptEvaluator

# An agent that adds one to genome.json and submits it; prints what `evokit help` lists if asked.
SUBMITTER = (
    "import json, subprocess, sys\n"
    "if 'help' in sys.stdin.read():\n"
    "    print(json.dumps({'summary': subprocess.run(['evokit', 'help'], capture_output=True, text=True).stdout}))\n"
    "    sys.exit()\n"
    "g = json.load(open('genome.json'))\n"
    "json.dump({'x': g['x'] + 1}, open('genome.json', 'w'))\n"
    "print(subprocess.run(['evokit', 'submit', 'plus one'], capture_output=True, text=True).stdout)\n"
)


def files_task(seed, tmp_path):
    project = Project(seed, genome=["main.py"], environment=tmp_path / "env")
    return ek.Task(project.seed_genome(), ScriptEvaluator(project, "python evaluate.py"), files=project)


def counting_task():
    return ek.Task({"x": 0}, lambda i: ek.Evaluation({"score": float(i.genome["x"])}))


def test_one_conversation_across_prompts(seed, tmp_path, fake_agent):
    experiment = files_task(seed, tmp_path).create_experiment()
    agent = CommandAgent([*fake_agent, "{session}"])

    async def go():
        async with experiment.start(agent) as directory:
            first = await agent.prompt("Improve main.py")
            assert (directory / "main.py").read_text() == "X = 1.5\n"
            agent.send("Step back and try something else.")
            second = await agent.prompt("Go on")
        return directory, first, second

    directory, first, second = asyncio.run(go())
    assert not directory.exists()
    assert first.calls == ("score",) and "draft={'score': -1.5}" in first.text and first.session == "sess-1"
    assert "redirect=True" in second.text and "resumed=sess-1" in second.text and second.calls == ()
    assert experiment.usage()["cost_usd"] == pytest.approx(0.02) and len(experiment.drafts()) == 1
    turns = experiment.notes("turn")
    assert [t["session"] for t in turns] == ["sess-1", "sess-1"] and turns[-1]["calls"] == ["score"]


def test_events_and_resuming(seed, tmp_path, fake_agent):
    experiment = files_task(seed, tmp_path).create_experiment()
    agent = CommandAgent([*fake_agent, "{session}"])

    async def go():
        async with experiment.start(agent, resume="sess-7"):
            return [event async for event in agent.prompt("Improve main.py")]

    *calls, turn = asyncio.run(go())
    assert [c.name for c in calls] == ["score"] and calls[0].reply["result"]["objectives"] == {"score": -1.5}
    assert isinstance(turn, ek.Turn) and "resumed=sess-7" in turn.text


def test_another_agent_inside_can_only_read(tmp_path):
    experiment = counting_task().create_experiment()
    agent = CommandAgent([sys.executable, "-c", SUBMITTER])

    async def go():
        async with experiment.start(agent):
            return await agent.fresh().prompt("help")

    tools = asyncio.run(go()).text
    assert "evokit stats" in tools and "evokit remember TEXT" in tools
    assert "evokit score" not in tools and "evokit submit" not in tools
    assert experiment.counts()["turn"] == 1


def test_send_needs_a_started_agent():
    with pytest.raises(ek.UsageError, match="not started"):
        CommandAgent(["true"]).send("hello")


def test_agents_in_parallel_and_stopping_at_the_limit():
    experiment = counting_task().create_experiment(max_evaluations=5)
    agent = CommandAgent([sys.executable, "-c", SUBMITTER])

    async def work():
        async with experiment.start(agent):
            while not experiment.limits.reached():
                await agent.prompt("go")

    async def go():
        await asyncio.gather(work(), work())

    asyncio.run(go())
    assert experiment.evaluations() >= 5 and experiment.best().genome["x"] >= 2
    replies = [json.loads(t["text"]) for t in experiment.notes("turn")]
    assert any(r.get("stop") == "5 evaluations" for r in replies)
