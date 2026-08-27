# src/pyabs/core/wakes.py

"""
TO DO:
    Add Resistive Wall Wake

FIX ME:
    Verify and replace theta wake matrix element calculation in theta_wake_scalar(m, n)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

Array = Any
WakeKind = Literal["theta"]

@dataclass
class WakeConfig:
    """
    """
    kind: WakeKind
    equal_tol: float = 1e-14



def theta_wake_scalar(n: int, m: int) -> float:
    if m == 0 and n == 0:
        return 0.5
    if (m==n):
        return 0.0
    if m==-n and n!=0:
        return 0.0
    if m==0:
        return -(1.0 - (-1.0)**n)/((np.pi*n)**2)

    return (1.0/(2.0*m*(np.pi**2)))*((1.0-(-1.0)**(m+n))/(m+n) + (1-(-1)**(m-n))/(m-n))   #-1 + (-1)**(m+n)/((np.pi)**2)*(m**2 - n**2)

def wake_scalar(kind: WakeKind, n: int, m: int):
    if kind == "theta":
        return theta_wake_scalar(n, m)

    raise ValueError(f"Unknown wake: {kind}")

def build_arbwake_matrix(basis: Any, config: WakeConfig)->np.ndarray:
    """
    Build the wake matrix Unm
    """
    U_nm = np.zeros((basis.n_modes, basis.n_modes), dtype=float)

    basis_n = np.asarray(basis.n)
    
    for i in range(basis.n_modes):
        n = int(basis_n[i])

        for j in range(basis.n_modes):
            m = int(basis_n[j])

            U_nm[i][j] = wake_scalar(config.kind, n, m)

    return U_nm
