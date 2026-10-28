import asyncio

import pytest

import evokit as ek

pytest.importorskip("rlmflow")
from evokit.recipes.experimental.evorlm import evorlm  # noqa: E402

MUTATE_DOC = "- `await mutate("
INSTRUCTION = "Your instruction, from the agent that started you: "


def toward_three() -> ek.Task:
    return ek.Task(seed=1.0, evaluate=lambda ind: ek.Evaluation({"score": -abs(ind.genome - 3)}))


class Scripted:
    """An rlmflow model that replays `root_turns` for the root and answers `child(instruction, prompt)` for ops."""

    def __init__(self, root_turns, child, prompts):
        self.root_turns, self.child, self.prompts = root_turns, child, prompts

    def chat(self, messages):
        prompt = "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)
        self.prompts.append(prompt)
        if INSTRUCTION in prompt:
            return self.child(prompt.split(INSTRUCTION)[1].split("\n")[0].strip(), prompt)
        turn = sum(m["role"] == "assistant" for m in messages)
        return self.root_turns[min(turn, len(self.root_turns) - 1)]


def scripted(root_turns, child):
    prompts = []
    return Scripted(root_turns, child, prompts), prompts


def block(code: str) -> str:
    return f"```repl\n{code}\n```"


SUBMIT_BEST = block('submit({"candidate_id": population(topk=1)[0]["id"], "title": "best"})')


def test_a_generation_admits_measured_returns_and_verifies_unmeasured_ones(tmp_path):
    def child(instruction, prompt):
        if instruction == "try 2.0":
            return block('print(await evaluate(2.0))\nsubmit({"candidate": 2.0, "title": "move toward 3"})')
        return block('submit({"candidate": 2.5, "title": "never evaluated"})')

    mutate = block(
        'best = population(topk=1)[0]["id"]\n'
        'print(await mutate({"instruction": "try 2.0", "parent_ids": [best]},\n'
        '                   {"instruction": "try 2.5", "parent_ids": [best]}))'
    )
    model, _ = scripted([mutate, SUBMIT_BEST], child)
    best = asyncio.run(evorlm(toward_three(), "scripted", llm_clients={"scripted": model}, dir=tmp_path / "e", generations=1))

    assert best.genome == 2.5 and best.name == "never evaluated"
    e = ek.open_experiment(tmp_path / "e")
    seed = next(i for i in e.evaluated() if i.name == "seed")
    assert best.parents == (seed.id,)
    assert e.counts()["evaluated"] == 3 and e.counts()["draft"] == 2  # the seed and two returns; two measurements
    (generation,) = e.notes("generation")
    assert all(r["admitted"] for r in generation["results"])
    assert (tmp_path / "e" / "rlm" / "generation-0").is_dir()


def test_an_op_recurses_and_hands_up_its_best():
    def child(instruction, prompt):
        if instruction == "split":
            if "[assistant]" not in prompt:
                return block('print(await mutate({"instruction": "try 2.9", "parent_ids": [population()[0]["id"]]}))')
            return block('submit({"candidate_id": population(topk=1)[0]["id"], "title": "my best"})')
        assert MUTATE_DOC not in prompt  # depth 2 is max_depth: a leaf cannot mutate
        return block('await evaluate(2.9)\nsubmit({"candidate": 2.9, "title": "near 3"})')

    mutate = block('print(await mutate({"instruction": "split", "parent_ids": [population()[0]["id"]], "evals": 2, "agents": 2}))')
    model, prompts = scripted([mutate, SUBMIT_BEST], child)
    best = asyncio.run(evorlm(toward_three(), "scripted", llm_clients={"scripted": model}, generations=1))

    assert best.genome == 2.9 and best.name == "near 3" and best.info["agent"].count(".") == 2
    assert any("split" in p and MUTATE_DOC in p for p in prompts)


def test_returns_it_cannot_verify_are_dropped():
    def child(instruction, prompt):
        if instruction == "invisible":
            return block('submit({"candidate_id": "nope", "title": "?"})')
        return block('await evaluate(1.5)\nsubmit({"candidate": 2.0, "title": "no evaluations left to verify it"})')

    mutate = block('print(await mutate({"instruction": "invisible"}, {"instruction": "broke", "evals": 1}))')
    model, prompts = scripted([mutate, SUBMIT_BEST], child)
    best = asyncio.run(evorlm(toward_three(), "scripted", llm_clients={"scripted": model}, generations=1))

    assert best.name == "seed"
    root = [p for p in prompts if INSTRUCTION not in p][-1]
    assert "'nope' is not in your population" in root
    assert "no evaluations left" in root


def test_a_files_genome_is_a_dict_of_its_editable_files(seed):
    from evokit.files import Project, ScriptEvaluator

    project = Project(seed, genome=["main.py"])
    task = ek.Task(project.seed_genome(), ScriptEvaluator(project, "python evaluate.py"), files=project)

    def child(instruction, prompt):
        if "[assistant]" not in prompt:
            return block('print(await evaluate({"data.txt": "x"}))')
        return block(
            'c = candidate(population()[0]["id"])\n'
            'c["main.py"] = c["main.py"].replace("1.0", "2.5")\n'
            'print(await evaluate(c))\n'
            'submit({"candidate": c, "title": "X = 2.5"})'
        )

    mutate = block('print(await mutate({"instruction": "raise X", "parent_ids": [population()[0]["id"]]}))')
    model, prompts = scripted([mutate, SUBMIT_BEST], child)
    best = asyncio.run(evorlm(task, "scripted", llm_clients={"scripted": model}, generations=1))

    assert best.genome.text("main.py") == "X = 2.5\n" and best.objective("score") == -0.5
    assert any("data.txt are not editable" in p for p in prompts)


def test_the_root_stops_mutating_after_its_generations():
    def child(instruction, prompt):
        return block('await evaluate(2.0)\nsubmit({"candidate": 2.0, "title": "2"})')

    mutate = block('print(await mutate({"instruction": "go"}))')
    model, prompts = scripted([mutate, mutate, SUBMIT_BEST], child)
    best = asyncio.run(evorlm(toward_three(), "scripted", llm_clients={"scripted": model}, generations=1))

    assert best.genome == 2.0
    roots = [p for p in prompts if INSTRUCTION not in p]
    assert MUTATE_DOC in roots[0] and MUTATE_DOC not in roots[1]
    assert "you cannot mutate: no generations" in roots[2]  # a binding from an earlier cell is refused


def test_one_evaluation_per_turn():
    def child(instruction, prompt):
        if "[assistant]" not in prompt:
            return block('print(await evaluate(2.0))\nprint(await evaluate(2.5))')
        return block('submit({"candidate": 2.0, "title": "2"})')

    mutate = block('print(await mutate({"instruction": "go", "evals": 3}))')
    model, prompts = scripted([mutate, SUBMIT_BEST], child)
    best = asyncio.run(evorlm(toward_three(), "scripted", llm_clients={"scripted": model}, generations=1))

    assert best.genome == 2.0
    assert any("one evaluate call per turn" in p for p in prompts)


def test_the_roots_own_candidate_reaches_the_archive():
    turns = [block("print(await evaluate(3.0))"), block('submit({"candidate": 3.0, "title": "exact"})')]
    model, _ = scripted(turns, lambda instruction, prompt: "")
    best = asyncio.run(evorlm(toward_three(), "scripted", llm_clients={"scripted": model}, generations=1))

    assert best.genome == 3.0 and best.objective("score") == 0


def test_an_op_gets_eval_budget_and_cannot_recurse_with_one_agent_call():
    def child(instruction, prompt):
        assert "You have 5 evaluations and 0 agent calls left" in prompt and MUTATE_DOC not in prompt
        assert "## Your parents" in prompt and "score -2" in prompt  # the parent's source, up front
        return block('submit({"candidate_id": population()[0]["id"], "title": "kept", "notes": "nothing beat it"})')

    mutate = block('print(await mutate({"instruction": "go", "parent_ids": [population()[0]["id"]]}))')
    model, prompts = scripted([mutate, SUBMIT_BEST], child)
    asyncio.run(evorlm(toward_three(), "scripted", llm_clients={"scripted": model}, generations=1, eval_budget=5))

    root = [p for p in prompts if INSTRUCTION not in p][1]
    assert "'unchanged': True" in root and "nothing beat it" in root and "'evals_used': 0" in root


def test_lineage_through_local_candidates_resolves_in_the_experiment(tmp_path):
    def child(instruction, prompt):
        if instruction == "split":
            turn = prompt.count("[assistant]")
            if turn == 0:
                return block('print(await mutate({"instruction": "a", "parent_ids": [population()[0]["id"]]}))')
            if turn == 1:
                return block('print(await mutate({"instruction": "b", "parent_ids": [population()[0]["id"]]}))')
            return block('submit({"candidate_id": population(topk=1)[0]["id"], "title": "my best"})')
        x = 2.9 if instruction == "a" else 2.95
        return block(f'await evaluate({x})\nsubmit({{"candidate": {x}, "title": "{x}"}})')

    mutate = block('print(await mutate({"instruction": "split", "parent_ids": [population()[0]["id"]], "evals": 3, "agents": 3}))')
    model, _ = scripted([mutate, SUBMIT_BEST], child)
    best = asyncio.run(evorlm(toward_three(), "scripted", llm_clients={"scripted": model}, dir=tmp_path / "e", generations=1))

    assert best.genome == 2.95
    e = ek.open_experiment(tmp_path / "e")
    assert best.parents and all(e.individual(p).name == "seed" for p in best.parents)  # through 2.9, kept only locally
    assert list((tmp_path / "e" / "populations").iterdir())  # every op's population is in the experiment


def test_the_root_cannot_submit_before_its_generations_are_spent(tmp_path):
    mutate = block('print(await mutate({"instruction": "try 2.0", "parent_ids": [population()[0]["id"]]}))')
    child = lambda instruction, prompt: block('await evaluate(2.0)\nsubmit({"candidate": 2.0, "title": "closer"})')
    model, prompts = scripted([mutate, SUBMIT_BEST, mutate, SUBMIT_BEST], child)
    asyncio.run(evorlm(toward_three(), "scripted", llm_clients={"scripted": model}, dir=tmp_path / "e", generations=2))

    assert len(ek.open_experiment(tmp_path / "e").notes("generation")) == 2
    assert any("1 generations are left: run them before you submit" in p.split("[assistant]")[-1] for p in prompts)


def test_a_new_candidate_keeps_the_parents_it_names():
    turns = [
        block('seed = population()[0]["id"]\nprint(await evaluate(3.0))'),
        block('submit({"candidate": 3.0, "title": "exact", "parent_ids": [seed, "nope"]})'),
    ]
    model, _ = scripted(turns, lambda instruction, prompt: "")
    best = asyncio.run(evorlm(toward_three(), "scripted", llm_clients={"scripted": model}, generations=1))

    assert len(best.parents) == 1 and best.parents != ()


def test_an_op_gets_turns_for_its_evaluations():
    seen = []

    def child(instruction, prompt):
        seen.append(prompt.count("[assistant]"))
        if prompt.count("[assistant]") < 20:
            return block('print("thinking")')
        return block('submit({"candidate_id": population()[0]["id"], "title": "kept"})')

    mutate = block('print(await mutate({"instruction": "go", "parent_ids": [population()[0]["id"]], "evals": 10}))')
    model, _ = scripted([mutate, SUBMIT_BEST], child)
    asyncio.run(evorlm(toward_three(), "scripted", llm_clients={"scripted": model}, generations=1, eval_budget=10, max_ops=1))

    assert max(seen) >= 20  # 2 * 10 + 4 turns, more than the root's 7


def test_mutate_is_unavailable_once_the_run_is_over(tmp_path):
    model, prompts = scripted([block('print(await mutate({"instruction": "go"}))'), SUBMIT_BEST], lambda i, p: "")
    asyncio.run(evorlm(toward_three(), "scripted", llm_clients={"scripted": model}, dir=tmp_path / "e", generations=1, max_cost=0))

    assert not ek.open_experiment(tmp_path / "e").notes("generation")
    assert "`mutate` is not available to you" in prompts[0]


def test_every_agent_works_in_its_own_directory(tmp_path):
    where = 'import os\nopen("main.py", "w").write("mine")\nprint("CWD", os.getcwd())'
    child = lambda i, p: block(where) if "CWD" not in p else block('submit({"candidate_id": population()[0]["id"], "title": "kept"})')
    mutate = block('print(await mutate({"instruction": "go", "parent_ids": [population()[0]["id"]]}))')
    model, prompts = scripted([block(where), mutate, SUBMIT_BEST], child)
    asyncio.run(evorlm(toward_three(), "scripted", llm_clients={"scripted": model}, dir=tmp_path / "e", generations=1))

    cwds = {line.split("CWD ")[1].split()[0] for p in prompts for line in p.splitlines() if line.startswith("CWD ")}
    assert len(cwds) == 2 and all(c.startswith(str((tmp_path / "e" / "agents").resolve())) for c in cwds)
