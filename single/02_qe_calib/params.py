"""Model and calibration parameters."""

import os
import sys
import numpy as np
from tauchen import *

#---------------------------------------------------------
# Model parameters (to calibrate with moments)
sigmaeps = 0.17
a = 4.3
b = 25.6
omega_max = 0.295 #entrepreneurs
theta_max = 0.046 #households

#---------------------------------------------------------
# Moments
mom_roe = 2.72
mom_deffreq = 2.33
mom_borrret = 10.02
mom_intmarg = 1.58
mom_lev = 9.32

#---------------------------------------------------------
# Other model parameters (pinned down with data)
beta = .95
delta = 0.244
cap_req = 0.07
lam = 0.276
omega_min = 0.0
theta_min = 0.0
rM = 0.008

#---------------------------------------------------------
# Shocks - grid
Nz = 5
rho = 0.614
sigmau = 0.00874
logz_grid, TP = approx_markov(rho, sigmau, m=3, n=Nz)
z_grid = np.exp(logz_grid)

# rL and rD grids
NrL = 25
rL_min = 0.001
rL_max = 0.05
rL_grid = np.linspace(rL_min, rL_max, num=NrL)
NrD = 25
rD_min = 0.001
rD_max = 0.05
rD_grid = np.linspace(rD_min, rD_max, num=NrD)
#---------------------------------------------------------
# Iteration parameters
tol = 1e-4
max_iterations = 10000

# Simulation parameters
Tsim = 30000
Tstart = 10000
