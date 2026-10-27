"""Score solve.py on this CO-Bench problem's development instances. Protected (part of the environment).

The problem's config.py (from the CO-Bench dataset) defines load_data, eval_func, norm_score and
get_dev; this follows CO-Bench's own Evaluator: every instance runs `solve(**instance)` in a fresh
process with a time limit, `eval_func(**instance, **solution)` checks and scores the solution, and
norm_score turns it into a ratio to the best known solution. An exception, an infeasible solution
or a timeout scores 0. The score is the mean over data files of the mean over their instances.

    python evaluate.py            # prints {"score": ..., "feasible": ..., "feedback": ...}
"""

import concurrent.futures
import fcntl
import importlib.util
import json
import multiprocessing as mp
import os
import signal
import sys
from pathlib import Path

TIMEOUT = 10.0  # seconds per instance, as in CO-Bench
FEEDBACK_LENGTH = 64


def problem() -> dict:
    return json.loads(Path(__file__).with_name("problem.json").read_text())


def load_config(data: Path):
    spec = importlib.util.spec_from_file_location("config", data / "config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _child(source: str, config_path: str, instance: dict, conn) -> None:
    os.setpgrp()
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    try:
        config = load_config(Path(config_path).parent)
        namespace: dict = {}
        exec(source, namespace)
        if "solve" not in namespace:
            raise ValueError("The source code does not define a 'solve' function.")
        solution = namespace["solve"](**instance)
        solution = {str(k): v for k, v in solution.items()}
        conn.send(config.eval_func(**instance, **solution))
    except Exception as exc:
        conn.send(f"Exception: {exc}")


def run_instance(source: str, data: Path, instance: dict, timeout: float):
    receive, send = mp.Pipe(duplex=False)
    process = mp.get_context("spawn").Process(
        target=_child, args=(source, str(data / "config.py"), instance, send)
    )
    process.start()
    send.close()
    try:
        if receive.poll(timeout + 1):
            return receive.recv()
        return f"Timeout ({timeout:g}s)"
    except EOFError:
        return "No result"
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        process.join()


def numbers(scores: list) -> list[float]:
    """Scores with every error, timeout or skip (a string) counted as 0."""
    return [
        s if isinstance(s, (int, float)) and not isinstance(s, bool) else 0.0
        for s in scores
    ]


def selected(count: int, case: str, dev: dict | None, split: str) -> list[int]:
    """The instance indices of one data file in `split` (CO-Bench's filter_dev / filter_test)."""
    if dev is None:
        return list(range(count))
    if case not in dev:
        return [] if split == "dev" else list(range(count))
    chosen = set(dev[case] or [0])
    return [i for i in range(count) if (i in chosen) == (split == "dev")]


def score(
    source: str,
    split: str = "dev",
    timeout: float = TIMEOUT,
    workers: int | None = None,
) -> dict:
    data = Path(problem()["data"])
    config = load_config(data)
    dev = config.get_dev() if hasattr(config, "get_dev") else None
    cases = sorted(
        p.name
        for p in data.iterdir()
        if p.is_file() and p.suffix != ".py" and not p.name.startswith(".")
    )
    jobs, results, wanted = [], {}, {}
    for case in cases:
        instances = config.load_data(str(data / case))
        indices = selected(len(instances), case, dev, split)
        if indices:
            results[case], wanted[case] = ["skipped"] * len(instances), set(indices)
            jobs += [(case, i, instances[i]) for i in indices]
    with open(
        data.parent / ".evaluate.lock", "w"
    ) as lock:  # one evaluation at a time across all problems, as CO-Bench's FileLock
        fcntl.flock(lock, fcntl.LOCK_EX)
        with concurrent.futures.ThreadPoolExecutor(workers or os.cpu_count()) as pool:
            futures = {
                pool.submit(run_instance, source, data, instance, timeout): (case, i)
                for case, i, instance in jobs
            }
            for future in concurrent.futures.as_completed(futures):
                case, i = futures[future]
                results[case][i] = future.result()
    normed = {case: (list(scores), None) for case, scores in results.items()}
    if hasattr(config, "norm_score"):
        normed = config.norm_score(normed)
    per_case = {
        case: [s for i, s in enumerate(scores) if i in wanted[case]]
        for case, (scores, _) in normed.items()
    }
    per_case = {case: scores for case, scores in per_case.items() if scores}
    total = sum(sum(numbers(s)) / len(s) for s in per_case.values()) / max(
        len(per_case), 1
    )
    flat = [x for s in per_case.values() for x in numbers(s)]
    lines = [
        f"{case} -> Scores: {[s if isinstance(s, str) else f'{float(s):.3f}' for s in scores][:FEEDBACK_LENGTH]}"
        for case, scores in per_case.items()
    ][:FEEDBACK_LENGTH]
    return {
        "score": total,
        "feasible": sum(x > 0 for x in flat) / max(len(flat), 1),
        "instances": len(flat),
        "feedback": "\n".join(lines) + f"\nAvg Score {total}",
    }


def main() -> None:
    path = Path("solve.py")
    if not path.is_file():
        sys.exit("solve.py is missing")
    source = path.read_text()
    try:
        compile(source, "solve.py", "exec")
    except SyntaxError as exc:
        sys.exit(f"solve.py does not compile: {exc}")
    print(json.dumps(score(source)))


if __name__ == "__main__":
    main()
