"""Util functions for the banking model."""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import sys


def probR(R, zval, a, b, sigmaeps):
    """Probability of success of a project."""

    x = (a*zval - b*R) / sigmaeps
    p = norm.cdf(x)

    return p


def expectation(R, rL, z, z_grid, TP, Nz, a, b, sigmaeps):
    """Expectation future returns of a project."""

    expect = 0
    for zp in range(Nz):
        zval = z_grid[zp]
        TPval = TP[z, zp]

        expect += TPval * probR(R, zval, a, b, sigmaeps) * (zval*R - rL)

    return expect


def LoanDemand(rL, z, z_grid, TP, Nz, omega_min, omega_max, a, b, sigmaeps):
    """Total loan demand."""

    R0 = 0.01
    myfun = lambda R: -1.0 * expectation(R, rL, z, z_grid, TP, Nz, a, b, sigmaeps)
    cons = ({'type': 'ineq', 'fun': lambda R:  R})
    opt = minimize(myfun, R0, method='SLSQP', constraints=cons)
    optR = opt.x
    optv = -1.0 * opt.fun

    if optv > omega_max:
        loan = 1.0
    elif optv < omega_min:
        loan = 0.0
    else:
        loan = (optv - omega_min) / (omega_max - omega_min)

    return loan, optR


def DepositSupply(rD, theta_min, theta_max):
    """Total deposit supply."""

    if rD > theta_max:
        dep = 1.0
    elif rD < theta_min:
        dep = 0.0
    else:
        dep = (rD - theta_min) / (theta_max - theta_min)

    return dep


def Profit(R, zval, rL, loan, dep, rD, rM, lam, delta, a, b, sigmaeps):
    """Profit of the bank at the end of the period."""

    pR = probR(R, zval, a, b, sigmaeps)

    prof = ((pR*(1+rL) + (1-pR)*(1-lam))*loan) + (1+rM)*delta*dep - ((1+rD)*dep)

    return prof


# Functions for simulation
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx



