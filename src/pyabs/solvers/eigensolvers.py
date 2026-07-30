from __future__ import annotations

from typing import Any
import numpy as np


class EigenSolution:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray | None = None


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


def scan_eigenvalues_over_sc(basis: Any, *, q: Array) -> EigenScanResult:
    sc = np.asarray(sc, dtype=complex)

    eigenvalues = np.empty((sc.shape[0], basis.n_modes), dtype=complex)

    for i, sc in enumerate(sc):
        matrix = build_abs_matrix(basis, q=complex(sc))
        eigenvalues[i] = solve_eigenvalues(matrix)
    return EigenScanResult(sc = sc, eigenvalues=eigenvalues)
