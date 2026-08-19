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
    alpha: np.ndarray | None  = None
    azimuthal: np.ndarray | None = None

    def compute_x(self, theta: float, psi: float):
        an = self.antheta(theta)
        az = self.azimuthal(psi)
        soln = 0
        for n, coeff in enumerate(self.eigenvalues):
            term = an[n]*az[n]
            soln = soln + term
        return soln

    def compute_xplus(self, theta: float, s: float):
        #an = self.antheta(theta)
        #az = self.azimuthal(psi)
        return self.compute_x(theta=theta, psi=(-np.pi*s))

    def compute_xminus(self, theta: float, s: float):
        return self.compute_x(theta=theta, psi=(np.pi*s))

    def compute_centroid(self, psi: float, *, theta: float = 0.0):
        soln = 0
        for n, coeff in enumerate(self.eigenvalues):
            term = (self.compute_x(theta, psi) + self.compute_x(theta, -psi))/2
            soln = soln + term
        return soln

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

def solve_system(matrix: Array, *, A0: complex = 1.0) -> Solution:
    eigsys = solve_eigensystem(matrix)
    alpha = solve_alpha(eigsys, A0=A0)
    antheta = solve_antheta(eigsys, alpha=alpha)
    azimuth=solve_azimuthal(eigsys)
    return Solution(eigenvalues=eigsys.eigenvalues, eigenvectors=eigsys.eigenvectors, 
                    antheta=antheta, alpha=alpha, azimuthal=azimuth)


#def solve_antheta(eigsys: EigenSolution):
#    antheta = lambda theta : np.sum(eigsys.eigenvectors*np.exp((eigsys.eigenvalues)*theta), axis=1)
#    return antheta

def solve_alpha(eigsys: EigenSolution, *, A0: complex = 1.0):
    size = int((int(eigsys.eigenvalues.shape[0])-1)/2)
    target = np.zeros(eigsys.eigenvalues.shape[0])
    target[size] = A0
    alpha = np.linalg.solve(eigsys.eigenvectors, target)
    return alpha

def solve_antheta(eigsys: EigenSolution, alpha: np.npdarray):
    term = lambda theta: eigsys.eigenvectors*np.exp((eigsys.eigenvalues)*theta)
    antheta = lambda theta: np.sum(alpha*term(theta), axis=1)
    return antheta

def solve_azimuthal(eigsys: EigenSolution):
    n_max = (int(eigsys.eigenvalues.shape[0]) - 1)/2
    nvec= np.arange(-n_max, n_max+1, dtype=int)
    azimuth = lambda psi: np.exp((1j)*nvec*psi)
    return azimuth

def scan_over_w(basis: Any, *, w: Array) -> EigenScanResult:
    print("I don't do anything yet")
    return EigenScanResult(sc = sc, eigenvalues=eigenvalues)



def scan_eigenvalues_over_sc(basis: Any, *, q: Array) -> EigenScanResult:
    sc = np.asarray(sc, dtype=complex)

    eigenvalues = np.empty((sc.shape[0], basis.n_modes), dtype=complex)

    for i, sc in enumerate(sc):
        matrix = build_abs_matrix(basis, q=complex(sc))
        eigenvalues[i] = solve_eigenvalues(matrix)
    return EigenScanResult(sc = sc, eigenvalues=eigenvalues)
