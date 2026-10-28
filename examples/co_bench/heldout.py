"""Score each experiment's best solve.py on its problem's held-out test instances.

    cd examples/co_bench
    python heldout.py runs/bin-packing-autoresearch-0 runs/bin-packing-avo-0

The best is the evaluated solve.py with the highest development score; it is written to
<run>/best_solve.py and scored with the run's own evaluate.py (in <run>/env).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import evokit as ek


def best(experiment) -> tuple[object, float] | None:
    scored = [
        (i, i.objective("score"))
        for i in experiment.evaluated()
        if i.objective("score") is not None
    ]
    return max(scored, key=lambda pair: pair[1]) if scored else None


SCORE_TEST = "import json, sys; from evaluate import score; print(json.dumps(score(open(sys.argv[1]).read(), 'test')))"


def test_run(directory: str | Path) -> dict:
    shown, directory = str(directory), Path(directory).resolve()
    experiment = ek.open_experiment(directory)
    env = directory / "env"
    counts, usage = experiment.counts(), experiment.usage()
    problem = json.loads((env / "problem.json").read_text())["problem"]
    row = {
        "run": shown,
        "problem": problem,
        "evaluations": counts["evaluated"],
        "cost_usd": usage.get("cost_usd", 0.0),
    }
    found = best(experiment)
    if found is None:
        return {**row, "dev": None, "test": None, "feasible": None}
    individual, dev = found
    (directory / "best_solve.py").write_text(individual.genome.text("solve.py"))
    done = subprocess.run(
        ["python", "-c", SCORE_TEST, str(directory / "best_solve.py")],
        cwd=env,
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(done.stdout.strip().splitlines()[-1])
    return {
        **row,
        "best": individual.id,
        "dev": dev,
        "test": result["score"],
        "feasible": result["feasible"],
    }


def number(x: float | None) -> str:
    return f"{x:7.4f}" if x is not None else "      -"


def print_rows(rows: list[dict]) -> None:
    print(
        f"{'run':40s} {'problem':34s} {'evals':>5s} {'cost':>7s} {'dev':>7s} {'test':>7s} {'feasible':>8s}"
    )
    for r in rows:
        print(
            f"{r['run'][-40:]:40s} {r['problem'][:34]:34s} {r['evaluations']:5d} ${r['cost_usd']:6.2f} "
            f"{number(r['dev'])} {number(r['test'])} {number(r['feasible']):>8s}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score runs' best solve.py on the held-out test set"
    )
    parser.add_argument("runs", nargs="+")
    print_rows([test_run(run) for run in parser.parse_args().runs])


if __name__ == "__main__":
    main()
