# src/pynht/core/basis.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Literal

ArrayModule = Any
Array = Any


""" All with no wake at this point in time """

@dataclass
class ABSBasisConfig:
    '''
    Configuration of the ABS config.

    Parameters
    ----------
    q:
        Space charge parameter
    n_max:
        Total number of modes. Size of truncation of fourier sums
    '''
    n_max: int
    q: float
    def __post_init__(self) -> None:
        if self.n_max < 0:
            raise ValueError("n_max must be non-negative.")


@dataclass
class ABSBasis:
    '''Flattened ABS Basis

    Attributes
    ----------
    '''
    n: Array
    q: float
    n_modes: int
    def index(self, n_index: int) -> int:
        return n_index #no flattening needed in this situation


def make_n_values(xp: ArrayModule, n_max: int) -> Array:
    return xp.arange(-n_max, n_max+1, dtype = int)

def build_abs_basis(xp: ArrayModule, config: ABSBasisConfig) -> ABSBasis:
    n_values = make_n_values(xp=xp, n_max=config.n_max)
    n = n_values.reshape(-1)
    q = config.q
    n_modes = int(n_values.shape[0])

    return ABSBasis(n=n, q=q, n_modes=n_modes)


def free_fourier_matrix(xp:ArrayModule, basis: ABSBasis, dtype: Any = complex) -> Array:
    '''
    Construct free fourier matrix 
    A0 = n*KroneckerDelta_nm
    '''
    return xp.diag(basis.n.astype(dtype))
