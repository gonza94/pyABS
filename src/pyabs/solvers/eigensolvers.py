from __future__ import annotations

from typing import Any
import numpy as np
from dataclasses import dataclass
from pyabs.core.matrix import build_abs_matrix, WakeModel
from pyabs.core.wake import ThetaWake, WakeScale, ResistiveWallWake

Array = Any

@dataclass
class EigenSolution:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray | None = None

'''
@dataclass
class SpaceChargeScanResult:
    sc_parameters: np.array
    eigenvalues: np.ndarray
'''
@dataclass
class WakeScanResult:
    wake_parameters: np.array
    eigenvalues: np.ndarray
    threshold: float | None = None

    def determine_threshold(self, *, verbose=False):
        eigs = self.eigenvalues
        for i, w in enumerate(self.wake_parameters):
            for j in range(len(eigs[0])-1):
                if((np.abs(eigs[i][j+1].real - eigs[i][j].real) < 1e-8)):
                    self.threshold = w
                    if(verbose):
                        upper = int(eigs[0][j+1].real)
                        lower = int(eigs[0][j].real)
                        print(f"Modes {upper} and {lower} couple at wth = {w}")
                    return w
                    continue
            self.threshold=0




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
        sindx = np.argsort(eigenvalues)
        eigenvalues, eigenvectors = eigenvalues[sindx], eigenvectors[:,sindx]
        return EigenSolution(eigenvalues=eigenvalues,
                             eigenvectors=eigenvectors,)

    eigenvalues = np.linalg.eigvals(matrix)
    sindx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sindx]
    return EigenSolution(eigenvalues=eigenvalues,
                         eigenvectors=None,)

def scan_over_w(basis: Any, wrange: Array, *, 
                determine_threshold=False,
                wake_model: WakeModel| None = None) -> WakeScanResult:
    result = np.empty((wrange.shape[0], basis.n_modes), dtype=complex)
    for i, w in enumerate(wrange):
        matrix = build_abs_matrix(basis, wake_model=wake_model, scale=WakeScale(w=w))
        result[i] = np.array(solve_eigenvalues(matrix))

    eigenscan = sort_by_continuity(result)
    result = WakeScanResult(wake_parameters=wrange, eigenvalues=eigenscan)

    if determine_threshold:
        wth = result.determine_threshold(verbose = True)
        return WakeScanResult(wake_parameters=wrange, eigenvalues=eigenscan, threshold=wth)
    return result


#i dont work yet. needs to be fixed though.
'''
def scan_over_q(basis: Any, qrange: Array,*, 
                wake_model: WakeModel | None = None, 
                scale: WakeScale | None = None) -> SpaceChargeScanResult:
    result=np.empty((qrange.shape[0], basis.n_modes), dtype=complex)
    for i, q in enumerate(qrange):
        update_q(basis, newq=q)
        matrix = build_abs_matrix(basis, wake_model=wake_model, scale=scale)
        result[i] = np.array(solve_eigenvalues(matrix))
    eigenscan = sort_by_continuity(result)

    return SpaceChargeScanResult(sc_parameters=qrange, eigenvalues=eigenscan)
'''
def sort_by_continuity(eigenscan):
    '''
    Python has a hard time with eigenvalue crossing. This takes eigenvalues in a parameter scan and sorts them by continuity.
    '''
    sorted_scan = eigenscan.copy()
    n_steps, n_modes = eigenscan.shape
    for i in range(1, n_steps):
        prev = sorted_scan[i - 1]
        curr = sorted_scan[i].copy()
        used = set()
        new_order = np.zeros(n_modes, dtype=int)
        for j in range(n_modes):
            # find the closest unused eigenvalue in curr to prev[j]
            dists = np.abs(curr - prev[j])
            dists[list(used)] = np.inf
            best = np.argmin(dists)
            new_order[j] = best
            used.add(best)
        sorted_scan[i] = curr[new_order]
    return sorted_scan
