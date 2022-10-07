"""Model and calibration parameters."""

import os
import sys
import numpy as np
from tauchen import *

#---------------------------------------------------------
# Model parameters (to calibrate with moments)
sigmaeps = 0.09
a = 2.6
# b = 26.0
omega_max = 0.315 #entrepreneurs
theta_max = 0.022 #households

# Set parameter from array
# sigmaeps = float(sys.argv[1])
# a = float(sys.argv[1])
b = float(sys.argv[1])
# omega_max = float(sys.argv[1])
# theta_max = float(sys.argv[1])

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
delta = 0.125
cap_req = 0.08
lam = 0.302
omega_min = 0.0
theta_min = 0.0

#---------------------------------------------------------
# Shock grid
Nz = 5
rho = 0.844
sigmau = 0.00718
logz_grid, TP = approx_markov(rho, sigmau, m=3, n=Nz)
z_grid = np.exp(logz_grid)

# rL and rD grids
NrL = 25
rL_min = 0.001
rL_max = 0.1
rL_grid = np.linspace(rL_min, rL_max, num=NrL)
NrD = 25
rD_min = 0.001
rD_max = 0.05
rD_grid = np.linspace(rD_min, rD_max, num=NrD)

# Q and rM grid
NQ = 20
Q_min = 0.0
Q_max = 1.0
Q_grid = np.linspace(Q_min, Q_max, num=NQ)
rM_min = 0.008
rM_max = 0.053
rM_grid = rM_min * np.ones(NQ)
rM_grid[0] = rM_max
#---------------------------------------------------------
# Iteration parameters
tol = 1e-4
max_iterations = 10000

# Simulation parameters
Tsim = 30000
Tstart = 10000
