"""The REPL that ReplAgent's code blocks run in. Standard library only, so that any Python can run
this file (`ReplProcess` passes its source to `python -c`), e.g. the task's Python with numpy.

- `Repl` is the interpreter: one namespace; `run(code)` returns what the block printed. With
  EVOKIT_TOOLS set (by `ToolServer`), the namespace also has the agent's tools as functions.
- `serve()` runs a `Repl` over stdin and stdout, one JSON line per block.
"""

import ast
import io
import json
import linecache
import os
import socket
import subprocess
import traceback
from contextlib import redirect_stderr, redirect_stdout


def sh(command, timeout=None):
    """Run a shell command in the working directory, print its output, and return its exit code."""
    result = subprocess.run(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout
    )
    print(result.stdout, end="")
    if result.returncode:
        print(f"[exit code {result.returncode}]")
    return result.returncode


def tools(spec):
    """Functions calling the tools a `ToolServer` serves, from its EVOKIT_TOOLS: {"port", "tools"}."""
    spec = json.loads(spec)

    def tool(name, signature):
        def call(*args, **kwargs):
            with socket.create_connection(("127.0.0.1", spec["port"])) as s:
                s.sendall((json.dumps({"tool": name, "args": args, "kwargs": kwargs}) + "\n").encode())
                reply = json.loads(b"".join(iter(lambda: s.recv(65536), b"")).decode() or "{}")
            for key in ("message", "stop"):
                if key in reply:
                    print(f"[{key}] {reply[key]}")
            if "error" in reply:
                raise RuntimeError(reply["error"] + (f" ({reply['hint']})" if reply.get("hint") else ""))
            return reply.get("result")

        call.__name__, call.__doc__ = name, f"{name}{signature}"
        return call

    return {name: tool(name, signature) for name, signature in spec["tools"].items()}


class Repl:
    """A Python namespace that runs blocks of code and returns what they print, plus the value of a
    last-line expression; an exception prints its traceback."""

    def __init__(self):
        self.namespace = {"__name__": "__main__", "__builtins__": __builtins__, "sh": sh}
        if os.environ.get("EVOKIT_TOOLS"):
            self.namespace.update(tools(os.environ["EVOKIT_TOOLS"]))
        self.blocks = 0

    def run(self, code):
        self.blocks += 1
        name = f"<block {self.blocks}>"
        linecache.cache[name] = (len(code), None, code.splitlines(True), name)
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(output):
            try:
                self._run(code, name)
            except BaseException as error:
                trace = error.__traceback__
                while trace and trace.tb_frame.f_code.co_filename != name:
                    trace = trace.tb_next
                traceback.print_exception(type(error), error, None if isinstance(error, SyntaxError) else trace)
        return output.getvalue()

    def _run(self, code, name):
        tree = ast.parse(code, name)
        last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        exec(compile(tree, name, "exec"), self.namespace)
        if last is not None:
            value = eval(compile(ast.Expression(last.value), name, "eval"), self.namespace)
            if value is not None:
                print(repr(value))


def serve():
    """Run each {"code": ...} line read from stdin in one `Repl`, replying {"output": ...} on stdout."""
    requests, replies = os.fdopen(os.dup(0)), os.fdopen(os.dup(1), "w")
    # Blocks must not read the requests or write into the replies: input() sees an empty stdin, and
    # output that bypasses sys.stdout (os.system, C extensions) goes to stderr, which is discarded.
    os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
    os.dup2(2, 1)
    repl = Repl()
    for line in requests:
        replies.write(json.dumps({"output": repl.run(json.loads(line)["code"])}) + "\n")
        replies.flush()


if __name__ == "__main__":
    serve()
