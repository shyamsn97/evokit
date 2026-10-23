"""The history of a run: what was created, evaluated, discarded and noted, with what it cost.

`ask` and `tell` write it; a recipe only writes notes and discards. Everything is one JSON line
per record in `history/records.jsonl`. Totals are kept in memory per storage, filled from the
log on first use, so checking a budget every iteration doesn't re-read the log.
"""

from __future__ import annotations

import logging
import threading
import time
import weakref
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from evokit.records import Individual, merge_usage
from evokit.storage import Storage, append_json, decode, encode, read_json_lines
from evokit.util import truncate

log = logging.getLogger("evokit")

LOG = "history/records.jsonl"


@dataclass
class _State:
    usage: Counter = field(default_factory=Counter)
    counts: Counter = field(default_factory=Counter)
    individuals: dict[str, Individual] = field(default_factory=dict)
    evaluated: list[str] = field(default_factory=list)
    admitted: list[str] = field(default_factory=list)
    pending: dict[str, None] = field(default_factory=dict)       # insertion-ordered set
    claimed: set[str] = field(default_factory=set)               # pending, but handed out by this process
    notes: dict[str, str] = field(default_factory=dict)          # the last text of each note kind
    failures_in_a_row: int = 0                                   # "failed" notes since the last creation
    lock: threading.RLock = field(default_factory=threading.RLock)


_states: "weakref.WeakKeyDictionary[Storage, _State]" = weakref.WeakKeyDictionary()
_states_lock = threading.Lock()


def _state(storage: Storage) -> _State:
    with _states_lock:
        state = _states.get(storage)
        if state is None:
            state = _states[storage] = _State()
            for record in read_json_lines(storage, LOG):
                _apply(state, record, storage)
    return state


def _apply(state: _State, record: dict, storage: Storage) -> None:
    kind = record["kind"]
    state.usage.update(merge_usage(record.get("usage")))
    if kind == "created":
        for data in record["children"]:
            child = decode(data, storage)
            state.individuals[child.id] = child
            state.pending[child.id] = None
        state.counts["created"] += len(record["children"])
        state.failures_in_a_row = 0
    elif kind == "evaluated":
        individual = decode(record["individual"], storage)
        state.individuals[individual.id] = individual
        state.pending.pop(individual.id, None)
        state.claimed.discard(individual.id)
        state.evaluated.append(individual.id)
        state.counts["evaluated"] += 1
        if record["added"]:
            state.admitted.append(individual.id)
            state.counts["added"] += 1
    elif kind == "discarded":
        state.pending.pop(record["id"], None)
        state.claimed.discard(record["id"])
        state.counts["discarded"] += 1
    elif kind == "note":
        state.notes[record["note"]] = record.get("text", "")
        state.counts[record["note"]] += 1
        if record["note"] == "failed":
            state.failures_in_a_row += 1


def _write(storage: Storage, record: dict) -> None:
    record = {**record, "time": time.time()}
    state = _state(storage)
    with state.lock:
        append_json(storage, LOG, record)
        _apply(state, record, storage)


# writing: ask, tell, and recipes that replace them


def record_created(storage: Storage, children: Iterable[Individual], parents: Iterable[str] = (), usage=None) -> None:
    """Children an operator made. They stay pending until evaluated or discarded; this process holds them."""
    children = list(children)
    _write(storage, {
        "kind": "created",
        "children": [encode(c, storage) for c in children],
        "parents": list(parents),
        "usage": merge_usage(usage),
    })
    _state(storage).claimed.update(c.id for c in children)


def record_evaluated(storage: Storage, individual: Individual, *, added: bool) -> None:
    """An evaluated individual, and whether the archive admitted it. Its cost is its evaluation's usage."""
    usage = individual.evaluation.usage if individual.evaluation is not None else {}
    _write(storage, {"kind": "evaluated", "individual": encode(individual, storage), "added": added, "usage": usage})


def claim_pending(storage: Storage, n: int) -> list[Individual]:
    """Up to n children that were created but never evaluated (e.g. a crash), not held by this process."""
    state = _state(storage)
    with state.lock:
        free = [i for i in state.pending if i not in state.claimed][:n]
        state.claimed.update(free)
        return [state.individuals[i] for i in free]


def note(storage: Storage, kind: str, text: str = "", usage=None) -> None:
    """Anything else worth keeping: a failed operator call, a redirect, a summary, an LLM judge's cost."""
    if text:
        log.info("%s: %s", kind, truncate(text, 500))
    _write(storage, {"kind": "note", "note": kind, "text": text, "usage": merge_usage(usage)})


def discard(storage: Storage, individual: Individual, reason: str) -> None:
    """A child that was created but will not be evaluated."""
    _write(storage, {"kind": "discarded", "id": individual.id, "reason": reason})


# reading


def usage(storage: Storage) -> Counter:
    """cost_usd, tokens, ...: the sum over everything recorded."""
    return Counter(_state(storage).usage)


def counts(storage: Storage) -> Counter:
    """"created", "evaluated", "added", "discarded", and each note kind."""
    return Counter(_state(storage).counts)


def evaluated(storage: Storage) -> list[Individual]:
    """Everything evaluated so far, in order, with their evaluations."""
    state = _state(storage)
    return [state.individuals[i] for i in state.evaluated]


def admitted(storage: Storage) -> list[Individual]:
    """The ones the archive admitted at the time."""
    state = _state(storage)
    return [state.individuals[i] for i in state.admitted]


def pending(storage: Storage) -> list[Individual]:
    """Created, but not yet evaluated or discarded."""
    state = _state(storage)
    return [state.individuals[i] for i in state.pending]


def individual(storage: Storage, id: str) -> Individual:
    """Raises KeyError if the history has no such individual."""
    return _state(storage).individuals[id]


def failures_in_a_row(storage: Storage) -> int:
    """Operator failures ("failed" notes) since the last child was created."""
    return _state(storage).failures_in_a_row


def last_note(storage: Storage, kind: str) -> str | None:
    return _state(storage).notes.get(kind)


def records(storage: Storage) -> list[dict]:
    """Every record, oldest first, as stored: for tools like compare.py."""
    return read_json_lines(storage, LOG)
