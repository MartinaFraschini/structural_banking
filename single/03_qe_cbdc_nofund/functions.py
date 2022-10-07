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


def DepositSupply(rD, rC, theta_min, theta_max, gamma_min, gamma_max):
    """Total deposit supply."""

    if rD > theta_max:
        dep_t = 1.0
    elif rD < theta_min:
        dep_t = 0.000001
    else:
        dep_t = (rD - theta_min) / (theta_max - theta_min)

    if (rD - rC) > gamma_max:
        dep_g = 1.0
    elif (rD - rC) < gamma_min:
        dep_g = 0.000001
    else:
        dep_g = (rD - rC - gamma_min) / (gamma_max - gamma_min)

    dep = dep_t * dep_g

    return dep


def CBDCSupply(rD, rC, theta_min, theta_max, gamma_min, gamma_max):
    """Total CBDC supply."""

    N = 10000
    sum_cbdc = 0
    for i in range(N):
        gamma = np.random.uniform(gamma_min, gamma_max)
        theta = np.random.uniform(theta_min, theta_max)
        if (gamma > rD - rC) and (theta < gamma + rC):
            sum_cbdc += 1
    
    cbdc = sum_cbdc / float(N)
    
    return cbdc


def TransferCBDC(rD, theta_min, theta_max, dep):
    """CBDC transfer from bank deposits to CBDC."""

    if rD > theta_max:
        dep_old = 1.0
    elif rD < theta_min:
        dep_old = 0.0
    else:
        dep_old = (rD - theta_min) / (theta_max - theta_min)

    transfer = dep_old - dep

    return transfer


def Profit(R, zval, rL, loan, dep, res, rD, rM, lam, a, b, sigmaeps):
    """Profit of the bank at the end of the period."""

    pR = probR(R, zval, a, b, sigmaeps)
    Prob = pR*(1+rL) + (1-pR)*(1-lam)
    prof = Prob*loan + (1+rM)*res - (1+rD)*dep

    return prof


# Functions for simulation
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx



