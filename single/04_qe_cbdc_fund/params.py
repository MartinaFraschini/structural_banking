"""Model and calibration parameters."""

import os
import sys
import numpy as np
from tauchen import *

#---------------------------------------------------------
# Set interest rates on CBDC
rC = 0.03
# Set interest rates CB funding
rF_type = 'rM'		# 'rM', 'rD', 'rC', number
# Set preference for CBDC
gamma_num = 0			# 0, 0.5, 1

#---------------------------------------------------------
# Model parameters (to calibrate with moments)
sigmaeps = 0.17
a = 4.3
b = 25.6
omega_max = 0.295 #entrepreneurs
theta_max = 0.046 #households

#---------------------------------------------------------
# Other model parameters (pinned down with data)
beta = .95
delta = 0.244       # exogenous reserves instead of liquidity requirement
cap_req = 0.07
lam = 0.276
omega_min = 0.0
theta_min = 0.0
rM = 0.008
phi = 0.0           # min liquidity buffer

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
# CBDC preference
gamma_min = 0.0
gamma_max = (gamma_num + 0.000001) * theta_max

#---------------------------------------------------------
# Iteration parameters
tol = 1e-4
max_iterations = 10000

# Simulation parameters
Tsim = 30000
Tstart = 10000
