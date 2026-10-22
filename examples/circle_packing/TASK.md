# Goal

Place 26 non-overlapping circles inside the unit square to maximize the sum of their radii.
The best known sum is about 2.635.

# Rules

- Edit only `main.py`. `evaluate.py`, `config.yaml` and this file are fixed.
- `construct_packing()` in `main.py` must return `(centers, radii)`: 26 `(x, y)` centers and 26 radii.

# Scoring

`evaluate.py` checks that every circle lies inside the square and that no two overlap (tolerance
1e-9), then reports `score`, the sum of the radii. An invalid packing, or a `construct_packing()`
that raises, fails its evaluation; the feedback names the circles at fault.

# Hints

- The seed is a center circle with rings of 8 and 17, shrunk until nothing overlaps.
- Optimizing the centers numerically (and the radii given the centers) does far better than a fixed pattern.
