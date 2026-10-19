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

#import numpy as np
from pyabs.core.backend import xp
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
        return -(1.0 - (-1.0)**n)/((xp.pi*n)**2)

    return (1.0/(2.0*m*(xp.pi**2)))*((1.0-(-1.0)**(m+n))/(m+n) + (1-(-1)**(m-n))/(m-n))   #-1 + (-1)**(m+n)/((xp.pi)**2)*(m**2 - n**2)

def rw_wake_scalar(n: int, m: int) -> float:
    if n == 0 and m == 0 :
        return 1.33333333333
    if xp.abs(n) == xp.abs(m) and (n != 0 and m != 0):
        return xp.sqrt(2/xp.abs(n))*(0.5*fresnel(xp.sqrt(2*xp.abs(n)))[1]-(fresnel(xp.sqrt(2*xp.abs(n)))[0])/4*xp.pi*xp.abs(n))
    if n**2 != m**2:
        return (((-1)**(n+m))*xp.sqrt(2*xp.abs(m))*fresnel(xp.sqrt(2*xp.abs(m)))[0] - xp.sqrt(2*xp.abs(n))*fresnel(xp.sqrt(2*xp.abs(n)))[0])/(xp.pi*(n**2 - m**2))

def sine_wake_scalar(n: int, m: int, omega: float) -> float:
    #check for special case (integer multiple of pi)
    if xp.isclose(xp.remainder(omega, xp.pi), 0.0) or xp.isclose(xp.remainder(omega, xp.pi), xp.pi):
        P = int(xp.round(omega/xp.pi))
        if xp.abs(m) == P and xp.abs(n) != xp.abs(m):
            return (-1)**(xp.abs(m)+xp.abs(n))*xp.abs(m)/(2*xp.pi*(n**2 - m**2))
        if xp.abs(n) == P and xp.abs(m) != xp.abs(n):
            return (-1)*xp.abs(n)/(2*xp.pi*(n**2-m**2))
        if (xp.abs(m) == xp.abs(n) == P != 0):
            return (-1)/(8*xp.pi*xp.abs(m))
        if n==m or n==-m:
            return P/(2*xp.pi*(P**2-m**2))
        if n==m==0:
            return P/(xp.pi*(P**2-m**2))
        return 0.0
    else:
        if n==m==0:
            return (omega/(omega**2 - m**2*(xp.pi**2)))*(1-((-1)**n * omega*xp.sin(omega))/(omega**2 - n**2 * xp.pi**2))
        if n==m or n==-m:
            return (omega/(omega**2 - m**2*(xp.pi**2)))*(1/2-((-1)**n * omega*xp.sin(omega))/(omega**2 - n**2 * xp.pi**2))
        return (omega/(omega**2 - m**2*(xp.pi**2)))*((-1)**n * omega*xp.sin(omega))/(omega**2 - n**2 * xp.pi**2)

def exp_wake_scalar(n: int, m: int, alpha: float)-> float:
    E_n = (alpha*((-1)**n)*xp.exp(alpha)-1)/(alpha**2 + (n*xp.pi)**2)
    if xp.abs(n) == xp.abs(m):
        return (-0.5*alpha + alpha*E_n)/(alpha**2 + (m*xp.pi)**2)
    if (m%2) == (n%2): #same parity
        return (alpha*E_n)/(alpha**2 + (m*xp.pi)**2)
    if (m%2) != (n%2): #opposity parity
        S_nm = (2*m)/(xp.pi*(m**2 - n**2))
        return (m*xp.pi*S_nm + alpha*E_n)/(alpha**2 + (m*xp.pi)**2)

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

def build_arbwake_matrix(basis: Any, config: WakeConfig)->Array:
    """
    Build the wake matrix Unm
    """
    U_nm = xp.zeros((basis.n_modes, basis.n_modes), dtype=float)

    basis_n = xp.asarray(basis.n)
    
    for i in range(basis.n_modes):
        n = int(basis_n[i])

        for j in range(basis.n_modes):
            m = int(basis_n[j])
            U_nm[i][j] = wake_scalar(config.kind, n, m)

    return U_nm


def build_twoparam_wake_matrix(basis: Any, config: WakeConfig, param: float)->Array:
    """
    Build frequency dependent wake matrices
    """
    U_nm = xp.zeros((basis.n_modes, basis.n_modes), dtype=float)
    basis_n = xp.asarray(basis.n)

    for i in range(basis.n_modes):
        n = int(basis_n[i])
        for j in range(basis.n_modes):
            m=int(basis_n[j])
            U_nm[i][j] = wake_scalar(config.kind, n, m, param=param)

    return U_nm
