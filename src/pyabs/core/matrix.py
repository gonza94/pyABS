"""
TO DO:
    Add Chromaticity
"""

import numpy as np
from typing import Any
from dataclasses import dataclass

from pyabs.core.basis import free_fourier_matrix
from pyabs.core.wake import (WakeScale, ThetaWake, build_wake_matrix)

Array = Any

@dataclass
class ABSMatrixParts:
    """
    The ABS Operator.

    Eigenvalue problem with no wake:
    Adot = -iH*A
    H = -i[(n - q/2)*KroneckerDelta_nm + (q/2)*KroneckerDelta_(-n)m]
    """
    fourier: np.ndarray
    wake: np.ndarray
    operator: np.ndarray


def build_fourier_operator(basis: Any) -> np.ndarray:
    """
    the no-wake fourier operator
    """
    return np.asarray(free_fourier_matrix(np, basis, dtype=complex))
    


def build_abs_matrix(basis: Any,
                     *,
                     wake_model: WakeModel | None=None,
                     scale: WakeScale | None=None) -> np.ndarray:
    parts = build_abs_matrix_parts(basis, wake_model=wake_model, scale=scale)
    return parts.operator

def build_abs_matrix_parts(basis: Any,
                           *,
                           wake_model: WakeModel | None=None,
                           scale: WakeScale | None=None) -> ABSMatrixParts:
    A = build_fourier_operator(basis)
    I = np.eye(basis.n_modes, dtype = complex)
    P = np.fliplr(I)
    U = np.zeros_like(A, dtype=complex)
    q=basis.q

    if wake_model is not None:
        if scale is None:
            raise ValueError("Scale error")
        U = build_wake_matrix(basis=basis, model=wake_model, scale=scale)


    return ABSMatrixParts(fourier=A, wake=U, operator= (-1j)*(A - (q/2)*I + (q/2)*P + U))
