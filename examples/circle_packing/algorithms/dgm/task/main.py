"""Pack N circles in the unit square to maximize the sum of their radii (N from $PACKING_N).

`construct_packing()` must return (centers, radii) with N entries each; `evaluate.py`
checks validity and scores it.
"""

import math
import os

N = int(os.environ.get("PACKING_N", "26"))


def construct_packing():
    # A center circle, an inner ring of 8, and an outer ring with the rest.
    centers = [(0.5, 0.5)]
    for i in range(8):
        a = 2 * math.pi * i / 8
        centers.append((0.5 + 0.22 * math.cos(a), 0.5 + 0.22 * math.sin(a)))
    for i in range(N - 9):
        a = 2 * math.pi * i / (N - 9)
        centers.append((0.5 + 0.42 * math.cos(a), 0.5 + 0.42 * math.sin(a)))
    centers = [(min(max(x, 0.01), 0.99), min(max(y, 0.01), 0.99)) for x, y in centers]
    return centers, compute_max_radii(centers)


def compute_max_radii(centers):
    """Largest radii that keep circles inside the square and non-overlapping (greedy scaling)."""
    radii = [min(x, y, 1 - x, 1 - y) for x, y in centers]
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            d = math.dist(centers[i], centers[j])
            if radii[i] + radii[j] > d:
                scale = d / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale
    return radii
