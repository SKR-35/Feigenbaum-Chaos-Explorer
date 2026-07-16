"""Numerical routines for bifurcations, orbits and Feigenbaum estimates."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .maps import MapSpec

FEIGENBAUM_DELTA = 4.669201609102990
FEIGENBAUM_ALPHA = -2.502907875095893


@dataclass(frozen=True)
class ScalingEstimate:
    level: int
    period: int
    parameter: float
    delta: float | None
    distance: float | None
    alpha: float | None


def iterate_orbit(spec: MapSpec, parameter: float, x0: float, iterations: int) -> np.ndarray:
    values = np.empty(iterations + 1, dtype=float)
    values[0] = x0
    for i in range(iterations):
        values[i + 1] = spec.function(values[i], parameter)
    return values


def bifurcation_points(
    spec: MapSpec,
    p_min: float,
    p_max: float,
    width: int = 1400,
    burn_in: int = 900,
    samples: int = 220,
) -> tuple[np.ndarray, np.ndarray]:
    parameters = np.linspace(p_min, p_max, width)
    x = np.full(width, spec.critical_point, dtype=float)
    for _ in range(burn_in):
        x = spec.function(x, parameters)
    xs = np.empty((samples, width), dtype=float)
    for i in range(samples):
        x = spec.function(x, parameters)
        xs[i] = x
    return np.tile(parameters, samples), xs.ravel()


def lyapunov_curve(
    spec: MapSpec,
    p_min: float,
    p_max: float,
    width: int = 1400,
    burn_in: int = 600,
    samples: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    parameters = np.linspace(p_min, p_max, width)
    x = np.full(width, spec.critical_point + 1e-9, dtype=float)
    for _ in range(burn_in):
        x = spec.function(x, parameters)
    total = np.zeros(width)
    for _ in range(samples):
        derivative = np.abs(spec.derivative(x, parameters))
        total += np.log(np.maximum(derivative, 1e-15))
        x = spec.function(x, parameters)
    return parameters, total / samples


def _critical_return(spec: MapSpec, parameter: float, period: int) -> float:
    x = spec.critical_point
    for _ in range(period):
        x = float(spec.function(x, parameter))
    return x - spec.critical_point


def _bisect(spec: MapSpec, period: int, left: float, right: float, tol: float = 1e-13) -> float:
    f_left = _critical_return(spec, left, period)
    f_right = _critical_return(spec, right, period)
    if f_left == 0:
        return left
    if f_right == 0:
        return right
    if f_left * f_right > 0:
        raise ValueError("Root is not bracketed")
    for _ in range(100):
        middle = (left + right) / 2.0
        f_mid = _critical_return(spec, middle, period)
        if abs(f_mid) < tol or right - left < tol:
            return middle
        if f_left * f_mid <= 0:
            right, f_right = middle, f_mid
        else:
            left, f_left = middle, f_mid
    return (left + right) / 2.0


def _next_superstable_root(spec: MapSpec, period: int, start: float, end: float) -> float:
    # Ignore the previous lower-period roots by beginning just above start and
    # select the first sign change. A dense scan is reliable at the modest
    # levels used by the interactive application.
    grid = np.linspace(start + max(1e-10, (end - start) * 1e-7), end, 30000)
    previous_x = grid[0]
    previous_y = _critical_return(spec, previous_x, period)
    for current_x in grid[1:]:
        current_y = _critical_return(spec, float(current_x), period)
        if math.isfinite(previous_y) and math.isfinite(current_y) and previous_y * current_y < 0:
            return _bisect(spec, period, float(previous_x), float(current_x))
        previous_x, previous_y = current_x, current_y
    raise RuntimeError(f"Could not bracket period-{period} superstable root")


def superstable_parameters(spec: MapSpec, levels: int = 7) -> list[float]:
    """Return period 2, 4, ..., 2**levels superstable parameters."""
    if spec.name == "Logistic map":
        lower, upper = 3.0, 3.58
    elif spec.name == "Sine map":
        lower, upper = 0.70, 0.87
    else:
        lower, upper = 0.70, 1.405

    roots: list[float] = []
    start = lower
    for level in range(1, levels + 1):
        period = 2**level
        root = _next_superstable_root(spec, period, start, upper)
        roots.append(root)
        start = root
    return roots


def _nearest_critical_distance(spec: MapSpec, parameter: float, period: int) -> float:
    orbit = iterate_orbit(spec, parameter, spec.critical_point, period)
    distances = np.abs(orbit[1:period] - spec.critical_point)
    distances = distances[distances > 1e-13]
    return float(distances.min())


def scaling_estimates(spec: MapSpec, levels: int = 7) -> list[ScalingEstimate]:
    roots = superstable_parameters(spec, levels)
    distances = [_nearest_critical_distance(spec, p, 2 ** (i + 1)) for i, p in enumerate(roots)]
    rows: list[ScalingEstimate] = []
    for i, parameter in enumerate(roots):
        delta = None
        if i >= 2:
            delta = (roots[i - 1] - roots[i - 2]) / (roots[i] - roots[i - 1])
        alpha = None
        if i >= 1 and distances[i] != 0:
            alpha = -distances[i - 1] / distances[i]
        rows.append(ScalingEstimate(i + 1, 2 ** (i + 1), parameter, delta, distances[i], alpha))
    return rows
