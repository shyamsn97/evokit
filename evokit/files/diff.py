"""Apply model output to Files: SEARCH/REPLACE diffs and full-file rewrites."""

from __future__ import annotations

import re

from evokit.errors import DiffError
from evokit.files.tree import Files

_BLOCK = re.compile(
    r"(?:^(?P<path>[^\n<>=`]*?)[ \t]*\n)?(?:```[^\n]*\n)?<<<<<<< SEARCH[ \t]*\n(?P<search>.*?)^=======[ \t]*\n(?P<replace>.*?)^>>>>>>> REPLACE[ \t]*$",
    re.S | re.M,
)


def apply_diff(files: Files, text: str) -> Files:
    """Apply SEARCH/REPLACE blocks (AlphaEvolve/OpenEvolve style).

    Each block may be preceded by a line naming the file. When there is exactly one file, the
    name may be omitted. An empty SEARCH section creates or overwrites the file.
    """
    blocks = list(_BLOCK.finditer(text))
    if not blocks:
        raise DiffError("no SEARCH/REPLACE blocks found")
    changed: dict[str, str] = {}
    paths = files.paths()
    for block in blocks:
        path = _clean_path(block.group("path"))
        if not path:
            if len(paths) != 1:
                raise DiffError("SEARCH/REPLACE block without a file name, and there are several files")
            path = paths[0]
        search, replace = block.group("search"), block.group("replace")
        if path in changed:
            current = changed[path]
        elif path in files:
            current = files.text(path)
        else:
            current = ""
        if not search.strip():
            changed[path] = replace
            continue
        if search in current:
            changed[path] = current.replace(search, replace, 1)
            continue
        loose = _loose_replace(current, search, replace)
        if loose is None:
            raise DiffError(f"SEARCH text not found in {path}:\n{search[:500]}")
        changed[path] = loose
    return files.with_files(changed)


def _clean_path(raw: str | None) -> str:
    if not raw:
        return ""
    path = raw.strip().strip("`*#: ").strip()
    if path.lower().startswith(("file", "path")) and ":" in path:
        path = path.split(":", 1)[1].strip()
    return path if path and " " not in path else ""


def _loose_replace(current: str, search: str, replace: str) -> str | None:
    """Match ignoring trailing whitespace on each line."""
    cur_lines = current.splitlines(keepends=True)
    want = [line.rstrip() for line in search.splitlines()]
    n = len(want)
    for i in range(len(cur_lines) - n + 1):
        if [line.rstrip() for line in cur_lines[i : i + n]] == want:
            return "".join(cur_lines[:i]) + replace + "".join(cur_lines[i + n :])
    return None


_FENCE = re.compile(r"(?:^|\n)(?:#+\s*|\*\*)?`?(?P<path>[\w./\-]+\.\w+)`?(?:\*\*)?:?\s*\n```[^\n]*\n(?P<body>.*?)\n```", re.S)
_BARE_FENCE = re.compile(r"```[^\n]*\n(?P<body>.*?)\n```", re.S)


def apply_rewrite(files: Files, text: str) -> Files:
    """Apply full-file rewrites: fenced code blocks preceded by a file name.

    With exactly one file, one bare fenced block replaces it.
    """
    changed = {m.group("path"): m.group("body") + "\n" for m in _FENCE.finditer(text)}
    if not changed:
        paths = files.paths()
        bare = _BARE_FENCE.findall(text)
        if len(paths) == 1 and bare:
            changed = {paths[0]: bare[-1] + "\n"}
        else:
            raise DiffError("no fenced code blocks with file names found")
    return files.with_files(changed)
