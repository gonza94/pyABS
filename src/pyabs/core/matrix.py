import numpy as np
from typing import Any
from dataclasses import dataclass

from pyabs.core.basis import free_fourier_matrix

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
    operator: np.ndarray


def build_fourier_operator(basis: Any) -> np.ndarray:
    """
    the no-wake fourier operator
    """
    return np.asarray(free_fourier_matrix(np, basis, dtype=complex))
    


def build_abs_matrix(basis: Any) -> np.ndarray:
    parts = build_abs_matrix_parts(basis)
    return parts.operator

def build_abs_matrix_parts(basis: Any) -> ABSMatrixParts:
    A = build_fourier_operator(basis)
    I = np.eye(basis.n_modes, dtype = complex)
    P = np.fliplr(I)
    q=basis.q

    return ABSMatrixParts(fourier=A, operator= (-1j)*(A - (q/2)*I + (q/2)*P))
