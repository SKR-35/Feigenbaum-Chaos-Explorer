"""One-dimensional maps used by the Feigenbaum explorer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class MapSpec:
    name: str
    parameter_symbol: str
    parameter_min: float
    parameter_max: float
    critical_point: float
    function: Callable[[Array | float, Array | float], Array | float]
    derivative: Callable[[Array | float, Array | float], Array | float]
    default_range: tuple[float, float]


def logistic(x, r):
    return r * x * (1.0 - x)


def logistic_derivative(x, r):
    return r * (1.0 - 2.0 * x)


def sine_map(x, r):
    return r * np.sin(np.pi * x)


def sine_derivative(x, r):
    return r * np.pi * np.cos(np.pi * x)


def quadratic(x, a):
    return 1.0 - a * x * x


def quadratic_derivative(x, a):
    return -2.0 * a * x


MAPS: dict[str, MapSpec] = {
    "Logistic map": MapSpec(
        "Logistic map", "r", 0.0, 4.0, 0.5,
        logistic, logistic_derivative, (2.8, 4.0),
    ),
    "Sine map": MapSpec(
        "Sine map", "r", 0.0, 1.0, 0.5,
        sine_map, sine_derivative, (0.65, 1.0),
    ),
    "Quadratic map": MapSpec(
        "Quadratic map", "a", 0.0, 2.0, 0.0,
        quadratic, quadratic_derivative, (0.5, 2.0),
    ),
}
