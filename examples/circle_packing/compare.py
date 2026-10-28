"""Run the algorithm examples and their variants, then compare them at equal cost.

    python compare.py run autoresearch avo openevolve:full openevolve:one-island --repeats 3
    python compare.py run dgm:full dgm:greedy --max-cost 20
    python compare.py report [runs] [--external-cost autoresearch-external-0=4.20]

An arm is ALGORITHM or ALGORITHM:VARIANT; its runs go to runs/<algorithm>[-<variant>]-<k>
(autoresearch runs unattended here; an interactive run belongs in runs/autoresearch-external-<k>).
The report reads each experiment's history (evokit.Experiment.records): the best admitted score at
fractions of the dollar budget (per algorithm, since DGM scores agents and the others score
packings), plus final cost, evaluations, and wall time. Interactive runs (an outside
agent as the operator) do not report their model cost; pass it with --external-cost.
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
from pathlib import Path

from evokit.experiment import LOG, Experiment
from evokit.storages import DirectoryStorage

HERE = Path(__file__).resolve().parent
FRACTIONS = (0.25, 0.5, 1.0)
MODULES = {
    "autoresearch": "algorithms.autoresearch",
    "avo": "algorithms.avo",
    "openevolve": "algorithms.openevolve",
    "dgm": "algorithms.dgm.main",
}
DEFAULT_VARIANT = {"openevolve": "full", "dgm": "full"}


def run_arms(arms: list[str], repeats: int, out: Path, max_cost: float | None) -> None:
    for arm in arms:
        algorithm, _, variant = arm.partition(":")
        if algorithm not in MODULES:
            raise SystemExit(f"unknown algorithm {algorithm!r}; choose from {', '.join(MODULES)}")
        variant = variant or DEFAULT_VARIANT.get(algorithm, "")
        if variant and algorithm not in DEFAULT_VARIANT:
            raise SystemExit(f"{algorithm} has no variants")
        for k in range(repeats):
            path = out / (f"{algorithm}-{variant}-{k}" if variant else f"{algorithm}-{k}")
            command = [sys.executable, "-m", MODULES[algorithm], "--dir", str(path)]
            if variant:
                command += ["--variant", variant]
            if max_cost is not None:
                command += ["--max-cost", str(max_cost)]
            print(f"{path.name}: {' '.join(command[1:])}", flush=True)
            subprocess.run(command, cwd=HERE, check=False)


def curve(run_dir: Path, extra_cost: float | None) -> dict:
    records = Experiment(DirectoryStorage(run_dir)).records()
    start = records[0]["time"] if records else 0.0
    best, points, kept, told, cost = None, [], 0, 0, 0.0
    for record in records:
        cost += (record.get("usage") or {}).get("cost_usd", 0.0)
        if record["kind"] == "evaluated":
            told += 1
            score = ((record["individual"].get("evaluation") or {}).get("objectives") or {}).get("score")
            if record["added"]:
                kept += 1
                if score is not None and (best is None or score > best):
                    best = score
        points.append({"best": best, "cost": cost, "evals": told, "seconds": record["time"] - start})
    final = points[-1] if points else {"cost": 0.0, "evals": 0, "seconds": 0.0}
    cost = extra_cost if extra_cost is not None else final["cost"]
    return {"points": points, "best": best, "cost": cost, "evals": final["evals"], "seconds": final["seconds"], "kept": kept, "told": told}


def best_at(points: list[dict], cost: float) -> float | None:
    eligible = [p["best"] for p in points if p["cost"] <= cost + 1e-12 and p["best"] is not None]
    return max(eligible) if eligible else None


def report(runs: Path, external_costs: dict[str, float]) -> None:
    groups: dict[str, dict[str, list]] = {}
    for run_dir in sorted(p for p in runs.iterdir() if (p / LOG).exists()):
        arm = run_dir.name.rsplit("-", 1)[0]
        algorithm = arm.split("-", 1)[0]
        groups.setdefault(algorithm, {}).setdefault(arm, []).append((run_dir.name, curve(run_dir, external_costs.get(run_dir.name))))
    for algorithm, arms in groups.items():
        budget = max((c["cost"] for entries in arms.values() for _, c in entries), default=0.0)
        print(f"\n{algorithm} (budget ${budget:.2f})")
        header = ["run", "best", "cost $", "evals", "kept/told", "minutes"] + [f"best@{int(f * 100)}%$" for f in FRACTIONS]
        print(" | ".join(header))
        for arm, entries in arms.items():
            for name, c in entries:
                at = [best_at(c["points"], f * budget) if budget else None for f in FRACTIONS]
                cells = [name, _fmt(c["best"]), f"{c['cost']:.2f}", str(int(c["evals"])), f"{c['kept']}/{c['told']}", f"{c['seconds'] / 60:.1f}"]
                print(" | ".join(cells + [_fmt(x) for x in at]))
            bests = [c["best"] for _, c in entries if c["best"] is not None]
            if bests:
                spread = statistics.stdev(bests) if len(bests) > 1 else 0.0
                print(f"{arm}: mean best {statistics.mean(bests):.4f} +/- {spread:.4f} over {len(bests)} runs")


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run")
    run_p.add_argument("arms", nargs="+", help="ALGORITHM or ALGORITHM:VARIANT")
    run_p.add_argument("--repeats", type=int, default=3)
    run_p.add_argument("--max-cost", type=float, default=None, help="US dollars per run (default: each example's own)")
    run_p.add_argument("--out", type=Path, default=HERE / "runs")
    rep = sub.add_parser("report")
    rep.add_argument("runs", nargs="?", type=Path, default=HERE / "runs")
    rep.add_argument("--external-cost", action="append", default=[], help="RUN=USD for interactive runs")
    args = parser.parse_args()
    if args.cmd == "run":
        run_arms(args.arms, args.repeats, args.out, args.max_cost)
    else:
        costs = dict((k, float(v)) for k, v in (item.split("=", 1) for item in args.external_cost))
        report(args.runs, costs)


if __name__ == "__main__":
    main()
