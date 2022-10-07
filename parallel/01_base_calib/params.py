"""Model and calibration parameters."""

import os
import sys
import numpy as np
from tauchen import *

#---------------------------------------------------------
# Model parameters (to calibrate with moments)
sigmaeps = 0.11
# a = 3.3
b = 24.0
omega_max = 0.26 #entrepreneurs
theta_max = 0.021 #households

# Set parameter from array
# sigmaeps = float(sys.argv[1])
a = float(sys.argv[1])
# b = float(sys.argv[1])
# omega_max = float(sys.argv[1])
# theta_max = float(sys.argv[1])

#---------------------------------------------------------
# Moments
mom_roe = 14.18
mom_deffreq = 2.23
mom_borrret = 7.94
mom_intmarg = 1.61
mom_lev = 9.17

#---------------------------------------------------------
# Other model parameters (pinned down with data)
beta = .95
delta = 0.029
cap_req = 0.08
lam = 0.302
omega_min = 0.0
theta_min = 0.0
rM = 0.053

#---------------------------------------------------------
# Shocks - grid
Nz = 5
rho = 0.844
sigmau = 0.00718
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
