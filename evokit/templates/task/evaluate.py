"""The evaluator: scores one individual. Part of the fixed environment; it never evolves.

The contract (what evokit's ScriptEvaluator expects):

- It runs as config.yaml's `evaluate` command, in a fresh copy of the task folder where the
  genome files are the individual's.
- It prints its metrics as the last stdout line, a JSON object: numbers become objectives
  (`objective` in config.yaml names the one to optimize), "feedback" is text the operator reads
  next time. `name: number` lines also work.
- It exits non-zero to reject the program: a failed evaluation, with stderr as feedback.
- $EVOKIT_WORKDIR, when set, is the individual's artifact folder: files written there are kept.
- `timeout` in config.yaml bounds it.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

TARGET = 3.0


def load(path: str = "main.py"):
    spec = importlib.util.spec_from_file_location("main", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    try:
        value = float(load().solve())
    except Exception as exc:
        print(f"main.py failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
    if os.environ.get("EVOKIT_WORKDIR"):
        Path(os.environ["EVOKIT_WORKDIR"], "solve.txt").write_text(f"{value}\n")
    print(json.dumps({"score": -abs(value - TARGET), "feedback": f"solve() returned {value}; the target is {TARGET}"}))


if __name__ == "__main__":
    main()
