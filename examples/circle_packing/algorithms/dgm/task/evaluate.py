"""Score main.py for N circles ($PACKING_N): the sum of radii if the packing is valid.

Prints JSON {score, valid, feedback}; exits 1 for an invalid packing.
"""

import importlib.util
import json
import math
import os
import sys

N = int(os.environ.get("PACKING_N", "26"))
TOL = 1e-9


def fail(message):
    print(json.dumps({"score": 0.0, "valid": False, "feedback": message}))
    sys.exit(1)


def main():
    spec = importlib.util.spec_from_file_location("main", "main.py")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        centers, radii = module.construct_packing()
        centers, radii = [tuple(map(float, c)) for c in centers], [float(r) for r in radii]
    except Exception as exc:
        fail(f"construct_packing() raised {type(exc).__name__}: {exc}")
    if len(centers) != N or len(radii) != N:
        fail(f"expected {N} circles, got {len(centers)} centers and {len(radii)} radii")
    problems = []
    for i, ((x, y), r) in enumerate(zip(centers, radii)):
        if not all(map(math.isfinite, (x, y, r))) or r < 0:
            problems.append(f"circle {i} has invalid values")
        elif x - r < -TOL or y - r < -TOL or x + r > 1 + TOL or y + r > 1 + TOL:
            problems.append(f"circle {i} leaves the square")
    for i in range(N):
        for j in range(i + 1, N):
            if math.dist(centers[i], centers[j]) < radii[i] + radii[j] - TOL:
                problems.append(f"circles {i} and {j} overlap")
    if problems:
        fail("; ".join(problems[:10]) + (f" (+{len(problems) - 10} more)" if len(problems) > 10 else ""))
    total = sum(radii)
    print(json.dumps({"score": total, "valid": True, "feedback": f"valid packing of {N} circles, sum of radii {total:.6f}"}))


if __name__ == "__main__":
    main()
