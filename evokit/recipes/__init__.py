"""Published algorithms, each a plain loop over evokit.ask, evokit.evaluate and evokit.tell.

Each takes what is specific to your problem (the project, the eval function, the agent or
models) plus a storage and an archive the caller has already seeded, and returns the best
individual. To change one, copy its loop.
"""

from evokit.recipes.autoresearch import autoresearch
from evokit.recipes.avo import avo
from evokit.recipes.dgm import dgm, staged_eval
from evokit.recipes.ga import ga
from evokit.recipes.hyperagents import hyperagents
from evokit.recipes.openevolve import openevolve
from evokit.recipes.shinka import ModelBandit, ShinkaParents, is_novel, shinka

__all__ = [
    "ModelBandit",
    "ShinkaParents",
    "autoresearch",
    "avo",
    "dgm",
    "ga",
    "hyperagents",
    "is_novel",
    "openevolve",
    "shinka",
    "staged_eval",
]
