"""
TO DO:
    Add Chromaticity
"""

#import numpy as np
from typing import Any
from dataclasses import dataclass
from pyabs.core.backend import xp

from pyabs.core.basis import free_fourier_matrix
from pyabs.core.wake import (WakeScale, ThetaWake, ResistiveWallWake, SineWake, ExpWake, build_wake_matrix)
from pyabs.core.chromaticity import (ChromaticScale, build_chromaticity_matrix)

Array = Any
WakeModel = ResistiveWallWake

@dataclass
class ABSMatrixParts:
    """
    The ABS Operator.

    Eigenvalue problem with no wake:
    Adot = -iH*A
    H = -i[(n - q/2)*KroneckerDelta_nm + (q/2)*KroneckerDelta_(-n)m]
    """
    fourier: Array
    wake: Array
    chromaticity: Array
    operator: Array


def build_fourier_operator(basis: Any) -> Array:
    """
    the no-wake fourier operator
    """
    return xp.asarray(free_fourier_matrix(xp, basis, dtype=complex))
    


def build_abs_matrix(basis: Any,
                     *,
                     wake_model: WakeModel | None=None,
                     scale: WakeScale | None=None,
                     xi: ChromaticScale | None=None) -> np.ndarray:
    parts = build_abs_matrix_parts(basis, wake_model=wake_model, scale=scale, xi=xi)
    return parts.operator

def build_abs_matrix_parts(basis: Any,
                           *,
                           wake_model: WakeModel | None=None,
                           scale: WakeScale | None=None,
                           xi: ChromaticScale) -> ABSMatrixParts:
    A = build_fourier_operator(basis)
    I = xp.eye(basis.n_modes, dtype = complex)
    P = xp.fliplr(I)
    U = xp.zeros_like(A, dtype=complex)
    Xi =xp.zeros_like(A, dtype=complex)
    q=basis.q

    if wake_model is not None:
        if scale is None:
            raise ValueError("Scale error")
        U = build_wake_matrix(basis=basis, model=wake_model, scale=scale)

    if xi is not None:
        Xi = build_chromaticity_matrix(basis=basis,scale=xi)

    return ABSMatrixParts(fourier=A, wake=U, chromaticity=Xi, operator= (A - (q/2)*I + (q/2)*P - U - Xi)) #operator = (-1j)*(the rest sometimes)
