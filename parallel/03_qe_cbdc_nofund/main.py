"""
Model calibration with standard policy. 

Value function iteration and simulation.

We consider:
2 choice variables: rL(t+1), rD(t+1)
4 state variables: rL(t), rD(t), z(t-1), z(t)

Author: Martina Fraschini
Conda environment: env_cbdc
Python version: 3.6
"""

# Import packages
import os
import sys
import numpy as np
from datetime import datetime
# from matplotlib import pyplot as plt

# Import parameters and functions
from params import *
from functions import *

# Make output directory
output_path = './res/gamma-' + str(int(gamma_num*100))
output_name = output_path + '/' + str(1000+int(rC*1000)) + '.npy'

# Start time
time_start = datetime.now()

# ######################################################################################################################
# ####################                           VALUE FUNCTION ITERATION                           ####################
# ######################################################################################################################

# Initialize value function
V_init = np.zeros((NrL, NrD, Nz, Nz))                   # 4 state variables: rL(t), rD(t), z(t-1), z(t)

# Useful variables
L_mat = np.zeros((NrL, Nz))                             # depends on rL(t), z(t-1)
R_mat = np.zeros((NrL, Nz))                             # depends on rL(t), z(t-1)
dep_mat = np.zeros(NrD)                                 # depends on rD(t)
res_mat = np.zeros(NrD)                                 # depends on rD(t)
cbdc_mat = np.zeros(NrD)                                # depends on rD(t)
tran_mat = np.zeros(NrD)                                # depends on rD(t)
eq_mat = np.zeros((NrL, NrD, Nz))                       # depends on rL(t+1), rD(t+1), z(t)
profit_mat = np.zeros((NrL, NrD, Nz, Nz))               # depends on rL(t), rD(t), z(t-1), z(t)
okdep = 1

for rL_ind in range(NrL):
    rL = rL_grid[rL_ind]
    for zm_ind in range(Nz):
        Lopt, Ropt = LoanDemand(rL, zm_ind, z_grid, TP, Nz, omega_min, omega_max, a, b, sigmaeps)
        L_mat[rL_ind, zm_ind] = Lopt
        R_mat[rL_ind, zm_ind] = Ropt
        for rD_ind in range(NrD):
            rD = rD_grid[rD_ind]
            if okdep:
                dep_mat[rD_ind] = DepositSupply(rD, rC, theta_min, theta_max, gamma_min, gamma_max)
                cbdc_mat[rD_ind] = CBDCSupply(rD, rC, theta_min, theta_max, gamma_min, gamma_max)
                tran_mat[rD_ind] = TransferCBDC(rD, theta_min, theta_max, dep_mat[rD_ind])
                if delta*dep_mat[rD_ind] - tran_mat[rD_ind] > phi:
                    res_mat[rD_ind] = delta*dep_mat[rD_ind] - tran_mat[rD_ind]
                else:
                    res_mat[rD_ind] = phi
            eq_mat[rL_ind, rD_ind, zm_ind] = Lopt + res_mat[rD_ind] - dep_mat[rD_ind]
            for z_ind in range(Nz):
                profit_mat[rL_ind, rD_ind, zm_ind, z_ind] = Profit(Ropt, z_grid[z_ind], rL, Lopt, dep_mat[rD_ind], 
                                                                   res_mat[rD_ind], rD, rM, lam, a, b, sigmaeps)
        okdep = 0

# Define the value function for each iteration
def vf_update(V_old, beta, profit_mat, eq_mat, L_mat, TP, cap_req):
    V_new = np.zeros_like(V_old)
    rD_policy = np.zeros_like(V_old)
    rL_policy = np.zeros_like(V_old)

    for rL_i in range(NrL):                     # rL(t)
        for rD_i in range(NrD):                 # rD(t)
            for zm_i in range(Nz):              # z(t-1)
                for z_i in range(Nz):           # z(t)
                    # among each value of rL and rD at t+1, I choose the one that maximizes the VF
                    # V_action = np.zeros((NrL, NrD))

                    # Profit
                    profval = profit_mat[rL_i, rD_i, zm_i, z_i]
                    prof = profval * np.ones((NrL, NrD))
                    # Equity
                    eq = eq_mat[:, :, z_i]
                    # Dividends
                    prof[prof < 0] = 0
                    div = prof - eq
                    # Assets
                    asset = L_mat[:, z_i].copy()
                    # Capital requirement
                    asset[asset == 0] = -1
                    cap = eq / asset[:, None]

                    # Expectation
                    expect = np.zeros((NrL, NrD))
                    for zp_i in range(Nz):
                        expect += TP[z_i, zp_i] * V_old[:, :, z_i, zp_i]

                    # Value function for all possible values of rLp
                    V_action = div + beta * expect

                    # Punishment for negative equity
                    V_action[eq <= 0] = -999

                    # Punishment for capital requirement not met
                    V_action[cap <= cap_req] = -999

                    # Find optimum
                    opt_ind = np.unravel_index(np.argmax(V_action, axis=None), V_action.shape)
                    V_new[rL_i, rD_i, zm_i, z_i] = V_action[opt_ind]
                    rL_policy[rL_i, rD_i, zm_i, z_i] = rL_grid[opt_ind[0]]
                    rD_policy[rL_i, rD_i, zm_i, z_i] = rD_grid[opt_ind[1]]

    return V_new, rL_policy, rD_policy


# Iterate until convergence
V_old = V_init.copy()

error = np.zeros(max_iterations)
for iteration in range(max_iterations):
    V_new, policy_rL, policy_rD = vf_update(V_old, beta, profit_mat, eq_mat, L_mat, TP, cap_req)
    error[iteration] = np.max(abs(V_new - V_old))

    V_old = V_new.copy()

    if error[iteration] < tol:
        time_diff = datetime.now() - time_start
        print('Converged after iteration {}'.format(iteration + 1))
        print('Running time: {}'.format(time_diff))

        break

if iteration == (max_iterations - 1):
    print('Not converged')
    sys.exit()

# ######################################################################################################################
# ####################                                  SIMULATION                                  ####################
# ######################################################################################################################

# Shock simulation
rand_num = np.random.rand(Tsim, 1)

# Initialization of the time series
shock_ind = np.zeros(Tsim, dtype=int)
shock = np.zeros(Tsim)
int_rates_loan_ind = np.zeros(Tsim, dtype=int)
int_rates_loan = np.zeros(Tsim)
int_rates_dep_ind = np.zeros(Tsim, dtype=int)
int_rates_dep = np.zeros(Tsim)
risk = np.zeros(Tsim)
prob_risk = np.zeros(Tsim)
loans = np.zeros(Tsim)
deposits = np.zeros(Tsim)
reserves = np.zeros(Tsim)
cbdc = np.zeros(Tsim)
transfer = np.zeros(Tsim)
CB_assets = np.zeros(Tsim)
CB_fund = np.zeros(Tsim)
profits = np.zeros(Tsim)
equity = np.zeros(Tsim)
dividends = np.zeros(Tsim)
borr_ret = np.zeros(Tsim)
eq_iss = np.zeros(Tsim)
int_margin = np.zeros(Tsim)
def_freq = np.zeros(Tsim)
ROE = np.zeros(Tsim)
leverage = np.zeros(Tsim)

# Cumulative transition probability for each shock
cumTP = TP.copy()
for n in range(Nz):
    cumTP[n, :] = np.cumsum(TP[n, :])

# Initial values
shock_ind[0] = int(Nz / 2)
int_rates_loan_ind[0] = int(NrL / 2)
int_rates_dep_ind[0] = int(NrD / 2)
equity[0] = 1
loans[0] = 1
# (I don't need the other initial values because I'm going to remove the first Tstart values from every series)

# Construction of time series
for t in range(1, Tsim-1):
    shock_ind[t] = next(i for i, x in enumerate(cumTP[shock_ind[t-1], :]) if x > rand_num[t])
    shock[t] = z_grid[shock_ind[t]]

    risk[t-1] = R_mat[int_rates_loan_ind[t-1], shock_ind[t-1]]
    prob_risk[t] = probR(risk[t-1], shock[t], a, b, sigmaeps)
    profits[t] = profit_mat[int_rates_loan_ind[t-1], int_rates_dep_ind[t-1], shock_ind[t-1], shock_ind[t]]

    int_rates_loan[t] = policy_rL[int_rates_loan_ind[t-1], int_rates_dep_ind[t-1], shock_ind[t-1], shock_ind[t]]
    int_rates_loan_ind[t] = np.where(rL_grid == int_rates_loan[t])[0][0]
    int_rates_dep[t] = policy_rD[int_rates_loan_ind[t-1], int_rates_dep_ind[t-1], shock_ind[t-1], shock_ind[t]]
    int_rates_dep_ind[t] = np.where(rD_grid == int_rates_dep[t])[0][0]

    loans[t] = L_mat[int_rates_loan_ind[t], shock_ind[t]]
    deposits[t] = dep_mat[int_rates_dep_ind[t]]
    reserves[t] = res_mat[int_rates_dep_ind[t]]
    cbdc[t] = cbdc_mat[int_rates_dep_ind[t]]
    transfer[t] = tran_mat[int_rates_dep_ind[t]]
    CB_assets[t] = reserves[t] + cbdc[t]
    equity[t] = loans[t] + reserves[t] - deposits[t]
    if profits[t] > 0:
        dividends[t] = profits[t] - equity[t]
    else:
        dividends[t] = - equity[t]

    borr_ret[t] = prob_risk[t] * (risk[t-1] * shock[t] - int_rates_loan[t-1])
    if dividends[t] < 0:
        eq_iss[t] = dividends[t] / equity[t]
    int_margin[t] = int_rates_loan[t-1] - int_rates_dep[t-1]
    def_freq[t] = 1 - prob_risk[t]
    ROE[t] = (profits[t] - equity[t-1]) / equity[t-1]
    leverage[t] = equity[t] / (loans[t-1] + reserves[t-1])

    

# Time series selection (I remove the first Tstart values from every series)
shock_ind = shock_ind[Tstart:-1]
shock = shock[Tstart:-1]
int_rates_loan_ind = int_rates_loan_ind[Tstart:-1]
int_rates_loan = int_rates_loan[Tstart:-1]
int_rates_dep_ind = int_rates_dep_ind[Tstart:-1]
int_rates_dep = int_rates_dep[Tstart:-1]
risk = risk[Tstart:-1]
prob_risk = prob_risk[Tstart:-1]
loans = loans[Tstart:-1]
deposits = deposits[Tstart:-1]
reserves = reserves[Tstart:-1]
cbdc = cbdc[Tstart:-1]
transfer = transfer[Tstart:-1]
CB_assets = CB_assets[Tstart:-1]
CB_fund = CB_fund[Tstart:-1]
profits = profits[Tstart:-1]
equity = equity[Tstart:-1]
dividends = dividends[Tstart:-1]
borr_ret = borr_ret[Tstart:-1]
eq_iss = eq_iss[Tstart:-1]
int_margin = int_margin[Tstart:-1]
def_freq = def_freq[Tstart:-1]
ROE = ROE[Tstart:-1]
leverage = leverage[Tstart:-1]


# Create moments database
moments = np.array([int_rates_dep.mean()*100, int_rates_loan.mean()*100, deposits.mean()*100, cbdc.mean()*100,
                    leverage.mean()*100, loans.mean()*100, ROE.mean()*100, equity.mean()*100, dividends.mean()*100,
                    profits.mean()*100, reserves.mean()*100, transfer.mean()*100, CB_fund.mean()*100, 
                    CB_assets.mean()*100, borr_ret.mean()*100, eq_iss.mean()*100, int_margin.mean()*100, 
                    def_freq.mean()*100])


np.save(output_name, moments)



