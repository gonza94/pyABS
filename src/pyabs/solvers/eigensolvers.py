from __future__ import annotations

from typing import Any
import numpy as np
from dataclasses import dataclass

Array = Any


"""
TO DO
~~sort eigenvectors~~
~~solve for antheta~~
solve for alpha
solve for azimuthal
"""




@dataclass
class Solution:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray | None = None
    antheta: np.ndarray | None = None
    azimuthal: np.ndarray | None = None
    total_solution : np.ndarray | None = None 

@dataclass
class EigenSolution:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray | None = None


@dataclass
class EigenScanResult:
    modes: np.ndarray
    eigenvalues: np.ndarray



def validate_square_matrix(matrix:Array) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=complex)

    if matrix.ndim != 2:
        raise ValueError("Matrix must be 2D.")

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square.")

    return matrix


def solve_eigenvalues(matrix: Array) -> np.ndarray:
    matrix = validate_square_matrix(matrix)
    return np.linalg.eigvals(matrix)


#return sorted eigensystem
def solve_eigensystem(matrix: Array, *, vectors: bool=True) -> EigenSolution:
    matrix = validate_square_matrix(matrix)

    if vectors:
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        sindx = np.argsort(eigenvalues.imag)
        eigenvalues, eigenvectors = eigenvalues[sindx], eigenvectors[:,sindx]
        return EigenSolution(eigenvalues=eigenvalues,
                             eigenvectors=eigenvectors,)

    eigenvalues = np.linalg.eigvals(matrix)
    sindx = np.argsort(eigenvalues.imag)
    eigenvalues = eigenvalues[sindx]
    return EigenSolution(eigenvalues=eigenvalues,
                         eigenvectors=None,)

def solve_antheta(eigsys: EigenSolution):
    antheta = lambda theta : np.sum(eigsys.eigenvectors*np.exp((eigsys.eigenvalues)*theta), axis=1)
    return Solution(eigenvalues=eigsys.eigenvalues, eigenvectors=eigsys.eigenvectors, antheta=antheta)

def scan_eigenvalues_over_sc(basis: Any, *, q: Array) -> EigenScanResult:
    sc = np.asarray(sc, dtype=complex)

    eigenvalues = np.empty((sc.shape[0], basis.n_modes), dtype=complex)

    for i, sc in enumerate(sc):
        matrix = build_abs_matrix(basis, q=complex(sc))
        eigenvalues[i] = solve_eigenvalues(matrix)
    return EigenScanResult(sc = sc, eigenvalues=eigenvalues)
