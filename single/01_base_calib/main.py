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
import pandas as pd
from datetime import datetime
# from matplotlib import pyplot as plt

# Import parameters and functions
from params import *
from functions import *

# Make output directories
output_path = './calibration'
if not os.path.exists(output_path):
    os.mkdir(output_path)
output_path = output_path + '/'

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
                dep_mat[rD_ind] = DepositSupply(rD, theta_min, theta_max)
            eq_mat[rL_ind, rD_ind, zm_ind] = Lopt - (1-delta)*dep_mat[rD_ind]
            for z_ind in range(Nz):
                profit_mat[rL_ind, rD_ind, zm_ind, z_ind] = Profit(Ropt, z_grid[z_ind], rL, Lopt, dep_mat[rD_ind], rD, 
                                                                   rM, lam, delta, a, b, sigmaeps)
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

        np.save(output_path+'VF.npy', V_new)
        np.save(output_path+'rLpolicy.npy', policy_rL)
        np.save(output_path+'rDpolicy.npy', policy_rD)

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

# ??Initial values
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
    reserves[t] = delta * deposits[t]
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
    leverage[t] = equity[t] / (loans[t-1] + delta*deposits[t-1])

    

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
profits = profits[Tstart:-1]
equity = equity[Tstart:-1]
dividends = dividends[Tstart:-1]
borr_ret = borr_ret[Tstart:-1]
eq_iss = eq_iss[Tstart:-1]
int_margin = int_margin[Tstart:-1]
def_freq = def_freq[Tstart:-1]
ROE = ROE[Tstart:-1]
leverage = leverage[Tstart:-1]

rD_bar = np.zeros(Nz)
for count, z in enumerate(z_grid):
    rD_series = int_rates_dep[shock == z]
    rD_bar[count] = rD_series.mean()
np.save(output_path+'rDbar.npy', rD_bar)


# ??Distiributions
# Save all the parameters values for this calibration
mytab = pd.DataFrame(columns=["Model", "Data"])
mytab.loc["Nz"] = [Nz, '']
mytab.loc["rho"] = [rho, '']
mytab.loc["sigmau"] = [sigmau, '']
mytab.loc["NrL"] = [NrL, '']
mytab.loc["rL_min"] = [rL_min, '']
mytab.loc["rL_max"] = [rL_max, '']
mytab.loc["NrD"] = [NrD, '']
mytab.loc["rD_min"] = [rD_min, '']
mytab.loc["rD_max"] = [rD_max, '']
mytab.loc["beta"] = [beta, '']
mytab.loc["delta"] = [delta, '']
mytab.loc["cap_req"] = [cap_req, '']
mytab.loc["lam"] = [lam, '']
mytab.loc["rM"] = [rM, '']

mytab.loc[""] = ['', '']

mytab.loc["omega_min"] = [omega_min, '']
mytab.loc["omega_max"] = [omega_max, '']
mytab.loc["a"] = [a, '']
mytab.loc["b"] = [b, '']
mytab.loc["sigmaeps"] = [sigmaeps, '']
mytab.loc["theta_min"] = [theta_min, '']
mytab.loc["theta_max"] = [theta_max, '']

mytab.loc[""] = ['', '']
mytab.loc["--------------------"] = ['', '']
mytab.loc["Moments (%)"] = ['', '']

mytab.loc["R_mean"] = [risk.mean() * 100, '']
mytab.loc["probR_mean"] = [prob_risk.mean() * 100, '']
mytab.loc["Loans_mean"] = [loans.mean() * 100, '']
mytab.loc["Loans_std"] = [loans.std() * 100, '']
mytab.loc["Deposits_mean"] = [deposits.mean() * 100, '']
mytab.loc["Deposits_std"] = [deposits.std() * 100, '']
mytab.loc["Reserves_mean"] = [reserves.mean() * 100, '']
mytab.loc["Profits_mean"] = [profits.mean() * 100, '']
mytab.loc["Profits_vol"] = [profits.std() * 100, '']
mytab.loc["Equity_mean"] = [equity.mean() * 100, '']
mytab.loc["Dividens_mean"] = [dividends.mean() * 100, '']
mytab.loc["IntRatesLoans_mean"] = [int_rates_loan.mean() * 100, '']
mytab.loc["IntRatesLoans_std"] = [int_rates_loan.std() * 100, '']
mytab.loc["IntRatesDep_mean"] = [int_rates_dep.mean() * 100, '']
mytab.loc["IntRatesDep_std"] = [int_rates_dep.std() * 100, '']
mytab.loc["NegDividends_perc"] = [np.sum(dividends < 0) * 100 / len(dividends), '']
mytab.loc["EquityIssuance"] = [eq_iss.mean() * 100, '']
mytab.loc["Default_perc"] = [np.sum(profits < 0) * 100 / len(profits), '']

mytab.loc[""] = ['', '']

mytab.loc["ROE"] = [ROE.mean() * 100, mom_roe]
mytab.loc["DefFreq"] = [def_freq.mean() * 100, mom_deffreq]
mytab.loc["BorrRet"] = [borr_ret.mean() * 100, mom_borrret]
mytab.loc["IntMargin"] = [int_margin.mean() * 100, mom_intmarg]
mytab.loc["Leverage"] = [leverage.mean() * 100, mom_lev]

mytab.to_csv(output_path+'summary.csv')

print("\n")
print(mytab)
print("\n")



