# This is a script made to separate the eigensolutions from the full system solutions

from __future__ import annotations
from typing import Any
import numpy as np
from dataclasses import dataclass
from pyabs.core.matrix import build_abs_matrix, WakeModel
from pyabs.core.wake import ThetaWake, ResistiveWallWake, WakeScale
from pyabs.solvers.eigensolvers import EigenSolution, solve_eigensystem

Array = Any
''' The Solution Class which encompasses the purpose of this file'''
@dataclass
class Solution:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    n_modes: int | None = None
    n_max: int | None = None
    an: np.ndarray| None = None
    an_norm: np.ndarray | None = None
    alpha: np.ndarray | None = None
    azimuthal: np.ndarray | None = None

    def compute_x(self, theta: float, psi: float):
        an = self.an_norm(theta)
        az = self.azimuthal(psi)
        result = 0
        for l in range(self.n_modes):
            term = an[l]*az[l]
            result = result + term
        return result

    def compute_xl(self, theta: float, psi: float, *, mode: int = None)->complex:
        an = self.an(theta)
        az = self.azimuthal(psi)
        xl = np.dot(az, an)
        if mode is not None:
            return xl[mode+self.n_max]
        return xl

    def compute_xplus(self, theta: float, s: float):
        return self.compute_x(theta=theta, psi=(s*np.pi))

    def compute_xminus(self, theta: float, s:float):
        return self.compute_x(theta=theta, psi=(-1*s*np.pi))

    def compute_amplification_factors(self, *, mode: int = None):
        modes = np.arange(-self.n_max, self.n_max+1, dtype=int)
        xl0 = self.compute_xl(0.0, 0.0)
        xlpi= self.compute_xl(0.0, np.pi)
        ks = np.abs(xlpi/xl0)
        if mode is not None:
            return ks[mode+self.n_max]
        return ks, modes

    def compute_xlcentroid(self, theta: float, psi: float):
        nvec = np.arange(-self.n_max, self.n_max+1, dtype=int)
        cos_basis = lambda psi: np.cos(nvec*psi)
        return np.dot(cos_basis(psi), self.an(theta))

    def compute_centroidstrobe(self, psi: float, *, ns: int=20, modes: Array = None):
        result = np.empty((self.n_modes, ns))
        if modes is not None:
            n_max = int((len(modes)-1)/2)
        else:
            n_max = self.n_max
            modes = np.arange(-n_max, n_max+1, dtype=int)
        for i, mode in enumerate(modes):
            for j in range(ns):
                val = self.compute_xlcentroid(theta=0.0, psi=psi)[mode+n_max]*np.exp(-2*1j*np.pi*j/ns)
                result[i][j] = np.real(val)
        return result

    def __post_init__(self) -> None:
        self.n_modes = int(self.eigenvalues.shape[0])
        self.n_max = int((self.n_modes-1)/2)




''' Solving the system'''
def solve_system(matrix: Array, *, A0: complex = 1.0) -> Solution:
    eigsys = solve_eigensystem(matrix)
    an = solve_an(eigsys)
    alpha = solve_alpha(eigsys, A0=A0)
    an_norm = solve_annorm(eigsys, alpha=alpha)
    azimuthal=solve_azimuthal(eigsys)
    return Solution(eigenvalues = eigsys.eigenvalues,
                    eigenvectors=eigsys.eigenvectors,
                    an=an, alpha=alpha,
                    an_norm=an_norm,
                    azimuthal=azimuthal)

def solve_an(eigsys: EigenSolution):
    an = lambda theta: np.exp((eigsys.eigenvalues*theta))*eigsys.eigenvectors
    return an

def solve_alpha(eigsys: EigenSolution, *, A0: complex = 1.0):
    size = eigsys.eigenvalues.shape[0]
    initialcond = np.zeros(size)
    initialcond[int((size-1)/2)]=A0
    alpha = np.linalg.solve(eigsys.eigenvectors, initialcond)
    return alpha

def solve_annorm(eigsys: EigenSolution, alpha: np.ndarray):
    term = lambda theta: eigsys.eigenvectors*np.exp((-1j)*(eigsys.eigenvalues)*theta)
    an_norm = lambda theta: np.sum(alpha*term(theta), axis=1)
    return an_norm

def solve_azimuthal(eigsys: Eigensolution):
    size = eigsys.eigenvalues.shape[0]
    n_max= int((size-1)/2)
    nvec = np.arange(-n_max, n_max+1, dtype=int)
    azimuth = lambda psi: np.exp((1j)*nvec*psi)
    return azimuth


