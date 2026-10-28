"""Run CO-Bench grids on Modal: one container per run, one per evaluation, all runs on the Volume `evokit-runs`.

    modal secret create evokit CLAUDE_CODE_OAUTH_TOKEN=... OPENAI_API_KEY=...       # once
    modal run modal_app.py --problems job-shop-scheduling --algorithms evorlm --max-cost 2
    modal run --detach modal_app.py --problems all --algorithms evorlm,avo,autoresearch,openevolve --seeds 3
    modal volume get evokit-runs co-bench runs/modal                                 # read the runs here

A run is <problem>/<algorithm>-<seed> under co-bench/ on the Volume; starting the same grid again continues
its runs. Claude Code uses the login token in the secret (no ANTHROPIC_API_KEY), with permissions bypassed:
the container is the sandbox. The image holds all 36 problems and their data, built once by setup.py.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MODAL_IMAGE_BUILDER_VERSION", "2025.06")  # the legacy builder's Debian 11 is gone

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RLMKIT = REPO.parent / "rlmkit"
CO_BENCH = "/root/co_bench"
RUNS = "/runs"
CPUS = 8

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "git")
    .run_commands("curl -fsSL https://claude.ai/install.sh | bash")
    .pip_install("numpy", "scipy", "networkx", "openai", "claude-agent-sdk")
    .env({
        "PATH": "/root/.local/bin:/usr/local/bin:/usr/bin:/bin",
        "IS_SANDBOX": "1",  # lets Claude Code bypass permissions as root
        "BASH_DEFAULT_TIMEOUT_MS": "3600000",  # an evorlm mutate cell can run for an hour
        "BASH_MAX_TIMEOUT_MS": "3600000",
        "CO_BENCH_WORKERS": str(CPUS),
    })
    .run_commands(
        "mkdir -p /root/.claude && echo '{\"permissions\": {\"defaultMode\": \"bypassPermissions\"}}' > /root/.claude/settings.json"
    )
    .add_local_file(HERE / "setup.py", f"{CO_BENCH}/setup.py", copy=True)
    .add_local_file(HERE / "evaluate.py", f"{CO_BENCH}/evaluate.py", copy=True)
    .run_commands(f"cd {CO_BENCH} && python setup.py --all")
    .add_local_file(RLMKIT / "pyproject.toml", "/opt/rlmkit/pyproject.toml", copy=True)
    .add_local_file(RLMKIT / "README.md", "/opt/rlmkit/README.md", copy=True)
    .add_local_dir(RLMKIT / "rlmflow", "/opt/rlmkit/rlmflow", copy=True, ignore=["**/__pycache__"])
    .run_commands("pip install '/opt/rlmkit[claude,openai]'")
    .add_local_file(REPO / "pyproject.toml", "/opt/evokit/pyproject.toml", copy=True)
    .add_local_file(REPO / "README.md", "/opt/evokit/README.md", copy=True)
    .add_local_dir(REPO / "evokit", "/opt/evokit/evokit", copy=True, ignore=["**/__pycache__"])
    .run_commands("pip install /opt/evokit")
    .add_local_file(HERE / "run.py", f"{CO_BENCH}/run.py")
    .add_local_file(HERE / "heldout.py", f"{CO_BENCH}/heldout.py")
)
app = modal.App("evokit-co-bench", image=image)
runs = modal.Volume.from_name("evokit-runs", create_if_missing=True)
secret = modal.Secret.from_name("evokit")


def co_bench():
    os.chdir(CO_BENCH)
    sys.path.insert(0, CO_BENCH)


@app.function(cpu=CPUS, memory=8192, timeout=3600)
async def evaluate(problem: str, individual):
    """One evaluation of a candidate, on the problem's development instances."""
    import evokit as ek

    return await ek.Task.from_directory(f"{CO_BENCH}/tasks/{problem}").evaluate(individual)


@app.function(cpu=CPUS, memory=8192, timeout=3600, volumes={RUNS: runs})
def heldout(name: str) -> dict:
    """The run's best solve.py on the held-out test instances."""
    co_bench()
    from heldout import test_run

    runs.reload()
    row = test_run(f"{RUNS}/co-bench/{name}")
    runs.commit()
    return row


@app.function(secrets=[secret], volumes={RUNS: runs}, timeout=24 * 3600, max_containers=8)  # Claude rate limits
async def experiment(problem: str, algorithm: str, name: str, agent: str, model: str,
                     max_evaluations: int, max_cost: float) -> dict:
    """One run of `algorithm` on `problem`, evaluating every candidate in its own container."""
    co_bench()
    from run import run

    async def remote(individual):
        return await evaluate.remote.aio(problem, individual)

    await run(problem, algorithm, f"{RUNS}/co-bench/{name}", agent, model, max_evaluations, max_cost,
              workers=4, evaluate=remote)
    await runs.commit.aio()
    return await heldout.remote.aio(name)


@app.local_entrypoint()
def main(problems: str = "job-shop-scheduling", algorithms: str = "evorlm", seeds: int = 1,
         agent: str = "claude", model: str = "gpt-5-mini", max_evaluations: int = 64,
         max_cost: float = 10.0, max_runs: int = 50):
    from setup import problems as listed, slug

    chosen = [slug(p) for p in listed()] if problems == "all" else problems.split(",")
    grid = [(p, a, f"{p}/{a}-{s}", agent, model, max_evaluations, max_cost)
            for p in chosen for a in algorithms.split(",") for s in range(seeds)]
    if len(grid) > max_runs:
        raise SystemExit(f"{len(grid)} runs is more than --max-runs {max_runs}")
    print(f"{len(grid)} runs, at most ${len(grid) * max_cost:.0f} of model cost")
    for spec, row in zip(grid, experiment.starmap(grid, return_exceptions=True)):
        if isinstance(row, BaseException):
            print(f"{spec[2]}: failed: {row!r}")
        else:
            print(f"{spec[2]}: {row['evaluations']} evaluations, ${row['cost_usd']:.2f}, "
                  f"dev {row['dev']}, test {row['test']}, feasible {row['feasible']}")
