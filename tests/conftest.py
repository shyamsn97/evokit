import sys
import textwrap
from pathlib import Path

import pytest

EVALUATE = textwrap.dedent(
    """
    import importlib.util, json, sys
    spec = importlib.util.spec_from_file_location("main", "main.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    if getattr(m, "BROKEN", False):
        print("main.py sets BROKEN", file=sys.stderr)
        sys.exit(1)
    if m.X > 3.0:
        print(f"incorrect: X = {m.X} is above 3", file=sys.stderr)
        sys.exit(1)
    print(json.dumps({"score": m.X - 3.0}))
    """
)

# A fake coding agent: bumps X by 0.5, uses a draft evaluation, reports usage as JSON.
FAKE_AGENT = textwrap.dedent(
    """
    import json, os, re, subprocess, sys
    prompt = sys.stdin.read()
    text = open("main.py").read()
    x = float(re.search(r"X = ([\\d.\\-]+)", text).group(1))
    open("main.py", "w").write(f"X = {x + 0.5}\\n")
    draft = None
    if os.environ.get("EVOKIT_DRAFT_EVAL"):
        draft = json.loads(subprocess.run([os.environ["EVOKIT_DRAFT_EVAL"]], capture_output=True, text=True).stdout)
    if os.environ.get("TAMPER"):
        open("evaluate.py", "a").write("# tampered\\n")
    redirect = "Step back" in prompt
    resumed = sys.argv[1] if len(sys.argv) > 1 else ""
    print(json.dumps({"summary": f"x -> {x + 0.5}; draft={draft and draft.get('objectives')}; redirect={redirect}; resumed={resumed}",
                      "cost_usd": 0.01, "tokens": 100, "session": resumed or "sess-1"}))
    """
)


@pytest.fixture
def seed(tmp_path: Path) -> Path:
    directory = tmp_path / "seed"
    directory.mkdir()
    (directory / "main.py").write_text("X = 1.0\n")
    (directory / "evaluate.py").write_text(EVALUATE)
    (directory / "data.txt").write_text("environment file\n")
    return directory


@pytest.fixture
def fake_agent(tmp_path: Path) -> list[str]:
    script = tmp_path / "fake_agent.py"
    script.write_text(FAKE_AGENT)
    return [sys.executable, str(script)]
