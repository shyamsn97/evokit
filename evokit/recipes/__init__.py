"""Published algorithms, each a plain function: `recipe(task, agent or models, *, dir=None,
archive=None, <limits>, <options>)`. Each creates its experiment on the task (in `dir`, continued
if it exists), runs its loop, and returns the best individual. To change one, copy its loop.
"""

from evokit.recipes.autoresearch import autoresearch
from evokit.recipes.avo import avo
from evokit.recipes.dgm import dgm, proportional, staged_eval
from evokit.recipes.ga import ga
from evokit.recipes.hyperagents import hyperagents
from evokit.recipes.openevolve import islands, openevolve
from evokit.recipes.shinka import ModelBandit, is_novel, parents, shinka

__all__ = [
    "ModelBandit",
    "autoresearch",
    "avo",
    "dgm",
    "ga",
    "hyperagents",
    "is_novel",
    "islands",
    "openevolve",
    "parents",
    "proportional",
    "shinka",
    "staged_eval",
]
