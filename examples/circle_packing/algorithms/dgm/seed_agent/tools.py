import subprocess
from pathlib import Path

MAX_OUTPUT = 10_000

TOOLS = [
    {
        "name": "bash",
        "description": "Run a shell command in the working directory. Returns the exit code and output.",
        "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
    },
    {
        "name": "view_file",
        "description": "Show a file with line numbers, optionally only lines start to end.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "start": {"type": "integer"}, "end": {"type": "integer"}},
            "required": ["path"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace the text `old`, which must appear exactly once, with `new`. "
        "With an empty `old`, write `new` as the whole file.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}},
            "required": ["path", "old", "new"],
        },
    },
]


def clip(text):
    if len(text) <= MAX_OUTPUT:
        return text
    return text[: MAX_OUTPUT // 2] + "\n... (clipped) ...\n" + text[-MAX_OUTPUT // 2 :]


def bash(root, command):
    try:
        proc = subprocess.run(command, shell=True, cwd=root, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return "error: the command took longer than 300 seconds and was stopped"
    return clip(f"exit code {proc.returncode}\n{proc.stdout}{proc.stderr}")


def view_file(root, path, start=None, end=None):
    lines = (Path(root) / path).read_text().splitlines()
    start = start or 1
    return clip("\n".join(f"{i:5}  {line}" for i, line in enumerate(lines[start - 1 : end], start)))


def edit_file(root, path, old, new):
    target = Path(root) / path
    if not old:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(new)
        return f"wrote {path}"
    text = target.read_text()
    count = text.count(old)
    if count != 1:
        return f"error: the old text appears {count} times in {path}; it must appear exactly once"
    target.write_text(text.replace(old, new))
    return f"edited {path}"


def run_tool(root, name, arguments):
    try:
        return {"bash": bash, "view_file": view_file, "edit_file": edit_file}[name](root, **arguments)
    except Exception as exc:
        return f"error: {type(exc).__name__}: {exc}"
