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

from pyabs.core.kernels import(WakeConfig, build_arbwake_matrix, build_twoparam_wake_matrix)

Array = Any

WakeKind = Literal["theta", "resisitve-wall", "sine", "exponential"]

@dataclass
class WakeScale:
    w: float

@dataclass
class ThetaWake:
    W0: float

def build_theta_wake_matrix(basis: Any, model: ThetaWake, scale: WakeScale)->np.ndarray:
    U_nm = build_arbwake_matrix(basis=basis, config=WakeConfig(kind="theta"))
    return model.W0*scale.w*U_nm

@dataclass
class ResistiveWallWake:
    RW: float

def build_rw_wake_matrix(basis: Any, model: ResistiveWallWake, scale: WakeScale)->np.ndarray:
    U_nm = build_arbwake_matrix(basis=basis, config=WakeConfig(kind="resistive-wall"))
    return model.RW*scale.w*U_nm

@dataclass
class SineWake:
    A: float
    omega: float

def build_sine_wake_matrix(basis: Any, model: SineWake, scale: WakeScale)->np.ndarray:
    U_nm = build_twoparam_wake_matrix(basis=basis, config=WakeConfig(kind="sine"), param=model.omega)
    return model.A*scale.w*U_nm

@dataclass
class ExpWake:
    A: float
    alpha: float

def build_exp_wake_matrix(basis: Any, model: ExpWake, scale: WakeScale)->np.ndarray:
    U_nm = build_twoparam_wake_matrix(basis=basis, config=WakeConfig(kind="exponential"), param=model.alpha)
    return model.A*scale.w*U_nm

def build_wake_matrix(basis: Any, model: Any, scale: WakeScale)->np.ndarray:
    if isinstance(model, ThetaWake):
        return build_theta_wake_matrix(basis=basis, model=model, scale=scale)
    if isinstance(model, ResistiveWallWake):
        return build_rw_wake_matrix(basis=basis, model=model, scale=scale)
    if isinstance(model, SineWake):
        return build_sine_wake_matrix(basis=basis, model=model, scale=scale)
    if isinstance(model, ExpWake):
        return build_exp_wake_matrix(basis=basis, model=model, scale=scale)

    raise TypeError(f"Unsupported wake model: {type(model)!r}")
