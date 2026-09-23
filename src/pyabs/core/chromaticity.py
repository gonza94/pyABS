# src/pyabs/core/chromaticity.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

Array = Any

@dataclass
class ChromaticScale:
    xi: float


def chrom_scalar(n: int, m: int) -> float:
    p = n-m
    if p%2 == 0:
        return 0
    else:
        return 2/(p*(np.pi)**2)


def build_chromaticity_matrix(basis: Any, scale: ChromaticScale)->np.ndarray:
    Xi_nm = np.zeros((basis.n_modes, basis.n_modes), dtype=float)

    basis_n = np.asarray(basis.n)

    for i in range(basis.n_modes):
        n = int(basis_n[i])
        for j in range(basis.n_modes):
            m = int(basis_n[j])
            Xi_nm[i][j] = chrom_scalar(n, m)

    return Xi_nm
