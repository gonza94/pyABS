"""
TO DO:
    Add resistive wall wake
    Add other precomputed wakes?
    non precomputed wakes?
"""

# src/pyabs/core/wake.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from pyabs.core.kernels import(WakeConfig, build_arbwake_matrix)

Array = Any

WakeKind = Literal["theta"]

@dataclass
class WakeScale:

    w: float

@dataclass
class ThetaWake:
    W0: float

def build_theta_wake_matrix(basis: Any, model: ThetaWake, scale: WakeScale)->np.ndarray:
    U_mn = build_arbwake_matrix(basis=basis, config=WakeConfig(kind="theta"))

    return model.W0 * U_mn


def build_wake_matrix(basis: Any, model: ThetaWake, scale: WakeScale)->np.ndarray:
    if isinstance(model, ThetaWake):
        return build_theta_wake_matrix(basis=basis, model=model, scale=scale)

    raise TypeError(f"Unsupported wake model: {type(model)!r}")
