from __future__ import annotations

from typing import Any
import numpy as np
from dataclasses import dataclass
#from pyabs.core.basis import ABSBasis, update_q
from pyabs.core.matrix import build_abs_matrix, WakeModel
from pyabs.core.wake import ThetaWake, WakeScale, ResistiveWallWake

Array = Any

@dataclass
class Solution:
    n_modes: int
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

    def compute_xl(self, theta: float, psi: float, l: int)->complex:
        n_modes = self.n_modes
        an_theta = solve_an(self.eigenvectors, self.eigenvalues)
        an = an_theta(theta) #self.antheta(theta)
        az = self.azimuthal(psi)
        return np.dot(an[l+n_modes],az[l+n_modes])

    def compute_xplus(self, theta: float, s: float):
        return self.compute_x(theta=theta, psi=(-1*np.pi*s))

    def compute_xminus(self, theta: float, s: float):
        return self.compute_x(theta=theta, psi=(np.pi*s))

    def compute_kmax(self):
        modes, ks = self.compute_amplification_factors()
        kmax = np.max(ks)
        return kmax

    def compute_amplification_factors(self):
        modes = np.arange(-(self.n_modes), (self.n_modes)+1, dtype=int)
        ks = np.empty_like(modes, dtype=float)
        for i, mode in enumerate(modes):
            xl0 = self.compute_xl(0,0,mode)
            xlpi= self.compute_xl(0,np.pi,mode)
            ks[i] = np.abs(xlpi[i]/xl0[i])
        return modes, ks

    #not done but it compiles
    def compute_centroid(self, theta: float, psi: float):
        an = self.antheta(theta)
        n_max = (int(self.eigenvalues.shape[0] - 1)/2)
        nvec = np.arange(-n_max, n_max+1, dtype=int)
        cosine_basis = np.cos(nvec*psi)
        return an*cosine_basis

    ''' we want
    def compute_centroid
    def compute_centroid_per_mode
    '''

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
                if((np.abs(eigs[i][j+1].real - eigs[i][j].real) < 1e-10)):
                    self.threshold = w
                    if(verbose):
                        upper = int(eigs[0][j+1].real)
                        lower = int(eigs[0][j].real)
                        print(f"Modes {upper} and {lower} couple at wth = {w}")
                    return w




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

def solve_system(matrix: Array, *, A0: complex = 1.0) -> Solution:
    eigsys = solve_eigensystem(matrix)
    n_modes = int((int(eigsys.eigenvalues.shape[0])-1)/2)
    alpha = solve_alpha(eigsys, A0=A0)
    antheta = solve_antheta(eigsys, alpha=alpha)
    azimuth=solve_azimuthal(eigsys)
    return Solution(n_modes = n_modes, eigenvalues=eigsys.eigenvalues,
                    eigenvectors=eigsys.eigenvectors, antheta=antheta,
                    alpha=alpha, azimuthal=azimuth)

def solve_an(eigenvectors, eigenvalues):
    an = lambda theta : eigenvectors*np.exp(((-1j)*eigenvalues)*theta)
    return an

def solve_alpha(eigsys: EigenSolution, *, A0: complex = 1.0):
    size = int((int(eigsys.eigenvalues.shape[0])-1)/2)
    target = np.zeros(eigsys.eigenvalues.shape[0])
    target[size] = A0
    alpha = np.linalg.solve(eigsys.eigenvectors, target)
    return alpha

def solve_antheta(eigsys: EigenSolution, alpha: np.npdarray):
    term = lambda theta: eigsys.eigenvectors*np.exp((-1j)*(eigsys.eigenvalues)*theta)
    antheta = lambda theta: np.sum(alpha*term(theta), axis=1)
    return antheta

def solve_azimuthal(eigsys: EigenSolution):
    n_max = (int(eigsys.eigenvalues.shape[0]) - 1)/2
    nvec= np.arange(-n_max, n_max+1, dtype=int)
    azimuth = lambda psi: np.exp((1j)*nvec*psi)
    return azimuth

def scan_over_w(basis: Any, wrange: Array, *, 
                determine_threshold=False) -> WakeScanResult:
    result = np.empty((wrange.shape[0], basis.n_modes), dtype=complex)
    for i, w in enumerate(wrange):
        rw_wake= ResistiveWallWake(RW=w)
        matrix = build_abs_matrix(basis, wake_model=rw_wake, scale=w)
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
