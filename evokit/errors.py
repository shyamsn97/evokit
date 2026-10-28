"""Errors with stable exit codes for the CLI and other agent-facing surfaces."""

from __future__ import annotations


class EvoError(Exception):
    code = 1
    kind = "error"

    def __init__(self, message: str, hint: str | None = None):
        super().__init__(message)
        self.message = message
        self.hint = hint

    def to_dict(self) -> dict:
        return {"error": self.message, "code": self.kind, "exit_code": self.code, "hint": self.hint}


class UsageError(EvoError):
    code = 2
    kind = "usage"


class EnvironmentModified(UsageError):
    kind = "environment_modified"

    def __init__(self, paths: list[str], editable: list[str]):
        self.paths = paths
        super().__init__(
            f"modified files outside the genome: {', '.join(paths)}",
            hint=f"Only these files evolve: {', '.join(editable)}. Revert the other changes and try again.",
        )


class RunBusy(UsageError):
    """Another process holds the run's lock (Storage.lock), e.g. another loop is driving it."""

    kind = "run_busy"


class OperatorError(Exception):
    """A variation tried and produced nothing, e.g. the LLM's diff didn't apply. `experiment.create`
    records it with its cost, and stores `logs` (file name -> contents) under failed/<id>/."""

    def __init__(self, reason: str, usage: dict | None = None, logs: dict[str, bytes] | None = None):
        super().__init__(reason)
        self.reason = reason
        self.usage = usage or {}
        self.logs = logs or {}


class DiffError(ValueError):
    pass
