"""HistoryCommands: the History's log, show, diff and grep as CLI and MCP commands."""

from __future__ import annotations

import functools

from evokit.archive import Archive
from evokit.history import History, diff_genomes, genome_text
from evokit.interface.commands import command
from evokit.storage import Storage


def _command(method, **options):
    @functools.wraps(method)
    def call(self, *args, **kwargs):
        return method(self, *args, **kwargs)

    return command(call, **options)


class HistoryCommands(History):
    """log, show, diff, grep. `archive` is the run's Archive, or the name of its log (default "archive")."""

    def __init__(self, storage: Storage, archive: Archive | str = "archive"):
        super().__init__(storage, archive if isinstance(archive, str) else archive.log.split("/")[0])

    log = _command(History.log)
    show = _command(History.show, positional=("path",))
    diff = _command(History.diff)
    grep = _command(History.grep)


__all__ = ["HistoryCommands", "diff_genomes", "genome_text"]
