# src/pyabs/core/wakes.py

"""
TO DO:
    Add Resistive Wall Wake

FIX ME:
    Verify and replace theta wake matrix element calculation in theta_wake_scalar(m, n)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.special import fresnel

Array = Any
WakeKind = Literal["theta", "resistive-wall", "sine", "exponential"]

@dataclass
class WakeConfig:
    """
    """
    kind: WakeKind
    equal_tol: float = 1e-14

def theta_wake_scalar(n: int, m: int) -> float:
    if m == 0 and n == 0:
        return 0.5
    if (m==n):
        return 0.0
    if m==-n and n!=0:
        return 0.0
    if m==0:
        return -(1.0 - (-1.0)**n)/((np.pi*n)**2)

    return (1.0/(2.0*m*(np.pi**2)))*((1.0-(-1.0)**(m+n))/(m+n) + (1-(-1)**(m-n))/(m-n))   #-1 + (-1)**(m+n)/((np.pi)**2)*(m**2 - n**2)

def rw_wake_scalar(n: int, m: int) -> float:
    if n == 0 and m == 0 :
        return 1.33333333333
    if np.abs(n) == np.abs(m) and (n != 0 and m != 0):
        return np.sqrt(2/np.abs(n))*(0.5*fresnel(np.sqrt(2*np.abs(n)))[1]-(fresnel(np.sqrt(2*np.abs(n)))[0])/4*np.pi*np.abs(n))
    if n**2 != m**2:
        return (((-1)**(n+m))*np.sqrt(2*np.abs(m))*fresnel(np.sqrt(2*np.abs(m)))[0] - np.sqrt(2*np.abs(n))*fresnel(np.sqrt(2*np.abs(n)))[0])/(np.pi*(n**2 - m**2))

def sine_wake_scalar(n: int, m: int, omega: float) -> float:
    #check for special case (integer multiple of pi)
    if np.isclose(np.remainder(omega, np.pi), 0.0) or np.isclose(np.remainder(omega, np.pi), np.pi):
        P = int(np.round(omega/np.pi))
        if np.abs(m) == P and np.abs(n) != np.abs(m):
            return (-1)**(np.abs(m)+np.abs(n))*np.abs(m)/(2*np.pi*(n**2 - m**2))
        if np.abs(n) == P and np.abs(m) != np.abs(n):
            return (-1)*np.abs(n)/(2*np.pi*(n**2-m**2))
        if (np.abs(m) == np.abs(n) == P != 0):
            return (-1)/(8*np.pi*np.abs(m))
        if n==m or n==-m:
            return P/(2*np.pi*(P**2-m**2))
        if n==m==0:
            return P/(np.pi*(P**2-m**2))
        return 0.0
    else:
        if n==m==0:
            return (omega/(omega**2 - m**2*(np.pi**2)))*(1-((-1)**n * omega*np.sin(omega))/(omega**2 - n**2 * np.pi**2))
        if n==m or n==-m:
            return (omega/(omega**2 - m**2*(np.pi**2)))*(1/2-((-1)**n * omega*np.sin(omega))/(omega**2 - n**2 * np.pi**2))
        return (omega/(omega**2 - m**2*(np.pi**2)))*((-1)**n * omega*np.sin(omega))/(omega**2 - n**2 * np.pi**2)

def exp_wake_scalar(n: int, m: int, alpha: float)-> float:
    E_n = (alpha*((-1)**n)*np.exp(alpha)-1)/(alpha**2 + (n*np.pi)**2)
    if np.abs(n) == np.abs(m):
        return (-0.5*alpha + alpha*E_n)/(alpha**2 + (m*np.pi)**2)
    if (m%2) == (n%2): #same parity
        return (alpha*E_n)/(alpha**2 + (m*np.pi)**2)
    if (m%2) != (n%2): #opposity parity
        S_nm = (2*m)/(np.pi*(m**2 - n**2))
        return (m*np.pi*S_nm + alpha*E_n)/(alpha**2 + (m*np.pi)**2)

def wake_scalar(kind: WakeKind, n: int, m: int, *, param: float = None):
    if kind == "theta":
        return theta_wake_scalar(n, m)
    if kind == "resistive-wall":
        return rw_wake_scalar(n, m)
    if kind == "sine":
        return sine_wake_scalar(n, m, param)
    if kind == "exponential":
        return exp_wake_scalar(n, m, param)

    raise ValueError(f"Unknown wake: {kind}")

def build_arbwake_matrix(basis: Any, config: WakeConfig)->np.ndarray:
    """
    Build the wake matrix Unm
    """
    U_nm = np.zeros((basis.n_modes, basis.n_modes), dtype=float)

    basis_n = np.asarray(basis.n)
    
    for i in range(basis.n_modes):
        n = int(basis_n[i])

        for j in range(basis.n_modes):
            m = int(basis_n[j])
            U_nm[i][j] = wake_scalar(config.kind, n, m)

    return U_nm


def build_twoparam_wake_matrix(basis: Any, config: WakeConfig, param: float)->np.ndarray:
    """
    Build frequency dependent wake matrices
    """
    U_nm = np.zeros((basis.n_modes, basis.n_modes), dtype=float)
    basis_n = np.asarray(basis.n)

    for i in range(basis.n_modes):
        n = int(basis_n[i])
        for j in range(basis.n_modes):
            m=int(basis_n[j])
            U_nm[i][j] = wake_scalar(config.kind, n, m, param=param)

    return U_nm
