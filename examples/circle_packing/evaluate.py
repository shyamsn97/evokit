"""Score main.py as ShinkaEvolve's examples/circle_packing/evaluate.py does (SakanaAI/ShinkaEvolve at
8adc053): call `run_packing()`, check its output with their `adapted_validate_packing` (copied
unchanged: strict, atol=0), and report the reported sum of radii as `score`. Protected (part of the
environment)."""

import importlib.util
import json
import sys
from typing import Optional, Tuple

import numpy as np


def adapted_validate_packing(
    run_output: Tuple[np.ndarray, np.ndarray, float],
    atol=0.0,
) -> Tuple[bool, Optional[str]]:
    """
    Validates circle packing results based on the output of 'run_packing'.

    Args:
        run_output: Tuple (centers, radii, reported_sum) from run_packing.

    Returns:
        (is_valid: bool, error_message: Optional[str])
    """
    centers, radii, reported_sum = run_output
    msg = "The circles are placed correctly. There are no overlaps or any circles outside the unit square."
    if not isinstance(centers, np.ndarray):
        centers = np.array(centers)
    if not isinstance(radii, np.ndarray):
        radii = np.array(radii)

    n_expected = 26
    if centers.shape != (n_expected, 2):
        msg = (
            f"Centers shape incorrect. Expected ({n_expected}, 2), got {centers.shape}"
        )
        return False, msg
    if radii.shape != (n_expected,):
        msg = f"Radii shape incorrect. Expected ({n_expected},), got {radii.shape}"
        return False, msg

    if (
        not np.all(np.isfinite(centers))
        or not np.all(np.isfinite(radii))
        or not np.isfinite(reported_sum)
    ):
        msg = "Non-finite values found in centers, radii, or reported_sum."
        return False, msg

    if np.any(radii < 0):
        negative_indices = np.where(radii < 0)[0]
        msg = f"Negative radii found for circles at indices: {negative_indices}"
        return False, msg

    if not np.isclose(np.sum(radii), reported_sum, atol=atol):
        msg = (
            f"Sum of radii ({np.sum(radii):.6f}) does not match "
            f"reported ({reported_sum:.6f})"
        )
        return False, msg

    for i in range(n_expected):
        x, y = centers[i]
        r = radii[i]
        is_outside = (
            x - r < -atol or x + r > 1 + atol or y - r < -atol or y + r > 1 + atol
        )
        if is_outside:
            msg = (
                f"Circle {i} (x={x:.4f}, y={y:.4f}, r={r:.4f}) is outside unit square."
            )
            return False, msg

    for i in range(n_expected):
        for j in range(i + 1, n_expected):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - atol:
                msg = (
                    f"Circles {i} & {j} overlap. Dist: {dist:.4f}, "
                    f"Sum Radii: {(radii[i] + radii[j]):.4f}"
                )
                return False, msg
    return True, msg


def fail(message):
    print(json.dumps({"score": 0.0, "feedback": message}))
    sys.exit(1)


def main():
    spec = importlib.util.spec_from_file_location("main", "main.py")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        output = module.run_packing()
        valid, message = adapted_validate_packing(output)
    except Exception as exc:
        fail(f"run_packing() failed: {type(exc).__name__}: {exc}")
    if not valid:
        fail(message)
    score = float(output[2])
    print(json.dumps({"score": score, "feedback": f"{message} Sum of radii: {score:.6f}"}))


if __name__ == "__main__":
    main()
