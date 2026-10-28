"""Strategies: how `archive.sample(n, strategy)` picks.

A strategy is a plain function `strategy(candidates, n, archive) -> individuals`. `candidates` are
what the archive offers, best first if it ranks (an IslandArchive offers one island at a time);
`archive.rng` is its random generator. Any strategy works with any archive.

    parent, *others = archive.sample(3)                           # the archive's own default
    parents = archive.sample(10, strategy=tournament(3))
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Sequence

from evokit.records import Individual

if TYPE_CHECKING:
    from evokit.archive import Archive

Strategy = Callable[[Sequence[Individual], int, "Archive"], list[Individual]]


def best(candidates: Sequence[Individual], n: int, archive: "Archive") -> list[Individual]:
    """The first `n` candidates: the best ones, in an archive that ranks."""
    return list(candidates[:n])


def uniform(candidates: Sequence[Individual], n: int, archive: "Archive") -> list[Individual]:
    """`n` different candidates at random (fewer if there are fewer)."""
    return archive.rng.sample(list(candidates), min(n, len(candidates)))


def tournament(size: int = 3) -> Strategy:
    """Each pick is the best of `size` candidates drawn at random; picks may repeat."""

    def pick(candidates: Sequence[Individual], n: int, archive: "Archive") -> list[Individual]:
        rank = {c.id: i for i, c in enumerate(candidates)}
        draw = lambda: min(archive.rng.sample(list(candidates), min(size, len(candidates))), key=lambda c: rank[c.id])  # noqa: E731
        return [draw() for _ in range(n)]

    return pick


__all__ = ["Strategy", "best", "tournament", "uniform"]
