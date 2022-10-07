"""
Model calibration with QE dynamics.

Value function iteration and simulation.

We consider
2 choice variables: rL(t+1), rD(t+1)
5 state variables: rL(t), rD(t), z(t-1), z(t), Q(t).

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

# Start time
time_start = datetime.now()

# ######################################################################################################################
# ####################                           VALUE FUNCTION ITERATION                           ####################
# ######################################################################################################################

# Initialize value function
V_init = np.zeros((NrL, NrD, Nz, Nz, NQ))               # 5 state variables: rL(t), rD(t), z(t-1), z(t), Q(t)

# Useful variables
L_mat = np.zeros((NrL, Nz))                             # depends on rL(t), z(t-1)
R_mat = np.zeros((NrL, Nz))                             # depends on rL(t), z(t-1)
dep_mat = np.zeros(NrD)                                 # depends on rD(t)
res_mat = np.zeros((NrD, NQ))                           # depends on rD(t), Q(t)
dQ_mat = np.zeros((NrL, NrD, Nz, Nz, NQ))               # depends on rL(t), rD(t), z(t-1), z(t), Q(t)
profit_mat = np.zeros((NrL, NrD, Nz, Nz, NQ))           # depends on rL(t), rD(t), z(t-1), z(t), Q(t)
eq_mat = np.zeros((NrL, NrL, NrD, NrD, Nz, Nz, NQ))     # depends on rL(t), rL(t+1), rD(t), rD(t+1), z(t-1), z(t), Q(t)

for rL_ind in range(NrL):
    for zm_ind in range(Nz):
        # choice of risky tech and loan demand
        Lopt, Ropt = LoanDemand(rL_grid[rL_ind], zm_ind, z_grid, TP, Nz, omega_min, omega_max, a, b, sigmaeps)
        L_mat[rL_ind, zm_ind] = Lopt
        R_mat[rL_ind, zm_ind] = Ropt
for rD_ind in range(NrD):
    # deposit supply
    dep_mat[rD_ind] = DepositSupply(rD_grid[rD_ind], theta_min, theta_max)
    for Q_ind in range(NQ):
        # reserves
        res_mat[rD_ind, Q_ind] = Q_grid[Q_ind] + delta * dep_mat[rD_ind]

for rL_ind in range(NrL):
    # loan interest rate
    rL = rL_grid[rL_ind]
    for zm_ind in range(Nz):
        for rD_ind in range(NrD):
            # deposit interest rate
            rD = rD_grid[rD_ind]
            for Q_ind in range(NQ):
                # reserves interest rate
                rM = rM_grid[Q_ind]
                for z_ind in range(Nz):
                    # profits
                    prof, dQ = Profit(R_mat[rL_ind, zm_ind], z_grid[z_ind], Q_grid[Q_ind], rL, L_mat[rL_ind, zm_ind],
                                      dep_mat[rD_ind], res_mat[rD_ind, Q_ind], rD, rM, lam, a, b, sigmaeps)
                    profit_mat[rL_ind, rD_ind, zm_ind, z_ind, Q_ind] = prof

                    dQ_mat[rL_ind, rD_ind, zm_ind, z_ind, Q_ind] = dQ
                    Qp = Q_grid[Q_ind] + dQ
                    for rLp_ind in range(NrL):
                        loanp = L_mat[rLp_ind, z_ind]
                        for rDp_ind in range(NrD):
                            depp = dep_mat[rDp_ind]
                            resp = Qp + delta * depp
                            # equity
                            eq_mat[rL_ind, rLp_ind, rD_ind, rDp_ind, zm_ind, z_ind, Q_ind] = loanp - Qp + resp - depp


# Define the value function for each iteration
def vf_update(V_old, beta, profit_mat, eq_mat, L_mat, dQ_mat, TP, cap_req):
    V_new = np.zeros_like(V_old)
    rD_policy = np.zeros_like(V_old)
    rL_policy = np.zeros_like(V_old)

    for rL_i in range(NrL):                         # rL(t)
        for rD_i in range(NrD):                     # rD(t)
            for zm_i in range(Nz):                  # z(t-1)
                for z_i in range(Nz):               # z(t)
                    for Q_i in range(NQ):           # Q(t)
                        # among each value of rL and rD at t+1, I choose the one that maximizes the VF
                        # V_action = np.zeros((NrL, NrD))

                        # Profit
                        prof_val = profit_mat[rL_i, rD_i, zm_i, z_i, Q_i]
                        prof = prof_val * np.ones((NrL, NrD))
                        # Equity
                        eq = eq_mat[rL_i, :, rD_i, :, zm_i, z_i, Q_i]
                        # Dividends
                        prof[prof < 0] = 0
                        div = prof - eq

                        # Central bank assets law of motion
                        Qp_val = Q_grid[Q_i] + dQ_mat[rL_i, rD_i, zm_i, z_i, Q_i]
                        Qp_i = find_nearest(Q_grid, Qp_val)
                        # Assets
                        asset = L_mat[:, z_i] - Qp_val
                        # Capital requirement
                        asset[asset == 0] = -1
                        cap = eq / asset[:, None]

                        # Expectation
                        expect = np.zeros((NrL, NrD))
                        for zp_i in range(Nz):
                            expect += TP[z_i, zp_i] * V_old[:, :, z_i, zp_i, Qp_i]

                        # Value function for all possible values of rLp
                        V_action = div + beta * expect

                        # Punishment for negative equity
                        V_action[eq <= 0] = -999

                        # Punishment for no loans
                        V_action[asset <= 0] = -999

                        # Punishment for capital requirement not met
                        V_action[cap <= cap_req] = -999

                        # Find optimum
                        opt_ind = np.unravel_index(np.argmax(V_action, axis=None), V_action.shape)
                        V_new[rL_i, rD_i, zm_i, z_i, Q_i] = V_action[opt_ind]
                        rL_policy[rL_i, rD_i, zm_i, z_i, Q_i] = rL_grid[opt_ind[0]]
                        rD_policy[rL_i, rD_i, zm_i, z_i, Q_i] = rD_grid[opt_ind[1]]

    return V_new, rL_policy, rD_policy


# Iterate until convergence
V_old = V_init.copy()

error = np.zeros(max_iterations)
for iteration in range(max_iterations):
    V_new, policy_rL, policy_rD = vf_update(V_old, beta, profit_mat, eq_mat, L_mat, dQ_mat, TP, cap_req)
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
# ###################                                   SIMULATION                                   ###################
# ######################################################################################################################

# Simulation shock
rand_num = np.random.rand(Tsim, 1)

# Initialization of the time series
shock_ind = np.zeros(Tsim, dtype=int)
shock = np.zeros(Tsim)
int_rates_loan_ind = np.zeros(Tsim, dtype=int)
int_rates_loan = np.zeros(Tsim)
int_rates_dep_ind = np.zeros(Tsim, dtype=int)
int_rates_dep = np.zeros(Tsim)
CB_assets_ind = np.zeros(Tsim, dtype=int)
CB_assets = np.zeros(Tsim)
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

#  Initial values
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
    profits[t] = profit_mat[int_rates_loan_ind[t-1], int_rates_dep_ind[t-1], shock_ind[t-1], shock_ind[t], CB_assets_ind[t-1]]
    CB_assets[t] = CB_assets[t-1] + dQ_mat[int_rates_loan_ind[t-1], int_rates_dep_ind[t-1], shock_ind[t-1], shock_ind[t], CB_assets_ind[t-1]]
    CB_assets_ind[t] = find_nearest(Q_grid, CB_assets[t])
    # it might be necessary to update the CB assets amount using points on the Q_grid to avoid overcharge (Q > 1)

    int_rates_loan[t] = policy_rL[int_rates_loan_ind[t-1], int_rates_dep_ind[t-1], shock_ind[t-1], shock_ind[t], CB_assets_ind[t-1]]
    int_rates_loan_ind[t] = np.where(rL_grid == int_rates_loan[t])[0][0]
    int_rates_dep[t] = policy_rD[int_rates_loan_ind[t-1], int_rates_dep_ind[t-1], shock_ind[t-1], shock_ind[t], CB_assets_ind[t-1]]
    int_rates_dep_ind[t] = np.where(rD_grid == int_rates_dep[t])[0][0]

    loans[t] = L_mat[int_rates_loan_ind[t], shock_ind[t]]
    deposits[t] = dep_mat[int_rates_dep_ind[t]]
    reserves[t] = CB_assets[t] + delta * deposits[t]
    equity[t] = loans[t] - CB_assets[t] + reserves[t] - deposits[t]
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
    leverage[t] = equity[t] / (loans[t-1] - CB_assets[t-1] + reserves[t-1])
    # leverage[t] = equity[t] / (loans[t-1] - CB_assets[t-1])           # check which formula is best for leverage
    # leverage[t] = equity[t] / reserves[t-1]


# Time series selection (I remove the first Tstart values from every series)
shock_ind = shock_ind[Tstart:-1]
shock = shock[Tstart:-1]
int_rates_loan_ind = int_rates_loan_ind[Tstart:-1]
int_rates_loan = int_rates_loan[Tstart:-1]
int_rates_dep_ind = int_rates_dep_ind[Tstart:-1]
int_rates_dep = int_rates_dep[Tstart:-1]
CB_assets_ind = CB_assets_ind[Tstart:-1]
CB_assets = CB_assets[Tstart:-1]
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


#  Distiributions
# Save all the parameters values for this calibration
print("\n")
print("Nz: {}".format(Nz))
print("rho: {}".format(rho))
print("sigmau: {}".format(sigmau))
print("NrL: {}".format(NrL))
print("rL_min: {}".format(rL_min))
print("rL_max: {}".format(rL_max))
print("NrD: {}".format(NrD))
print("rD_min: {}".format(rD_min))
print("rD_max: {}".format(rD_max))
print("beta: {}".format(beta))
print("delta: {}".format(delta))
print("cap_req: {}".format(cap_req))
print("lam: {}".format(lam))
print("rM_max: {}".format(rM_max))
print("rM_max: {}".format(rM_min))

print("\n")

print("omega_min: {}".format(omega_min))
print("omega_max: {}".format(omega_max))
print("a: {}".format(a))
print("b: {}".format(b))
print("sigmaeps: {}".format(sigmaeps))
print("theta_min: {}".format(theta_min))
print("theta_max: {}".format(theta_max))

print("\n")
print("--------------------")
print("Moments (%)")

print("R_mean: {}".format(risk.mean() * 100))
print("probR_mean: {}".format(prob_risk.mean() * 100))
print("Loans_mean: {}".format(loans.mean() * 100))
print("Loans_std: {}".format(loans.std() * 100))
print("Deposits_mean: {}".format(deposits.mean() * 100))
print("Deposits_std: {}".format(deposits.std() * 100))
print("Reserves_mean: {}".format(reserves.mean() * 100))
print("CBassets_mean: {}".format(CB_assets.mean() * 100))
print("Profits_mean: {}".format(profits.mean() * 100))
print("Profits_std: {}".format(profits.std() * 100))
print("Equity_mean: {}".format(equity.mean() * 100))
print("Dividens_mean: {}".format(dividends.mean() * 100))
print("IntRatesLoans_mean: {}".format(int_rates_loan.mean() * 100))
print("IntRatesLoans_std: {}".format(int_rates_loan.std() * 100))
print("IntRatesDep_mean: {}".format(int_rates_dep.mean() * 100))
print("IntRatesDep_std: {}".format(int_rates_dep.std() * 100))
print("NegDividends_perc: {}".format(np.sum(dividends < 0) * 100 / len(dividends)))
print("EquityIssuance: {}".format(eq_iss.mean() * 100))

print("\n")

print("ROE: {}, {}".format(ROE.mean() * 100, mom_roe))
print("DefFreq: {}, {}".format(def_freq.mean() * 100, mom_deffreq))
print("BorrRet: {}, {}".format(borr_ret.mean() * 100, mom_borrret))
print("IntMargin: {}, {}".format(int_margin.mean() * 100, mom_intmarg))
print("Leverage: {}, {}".format(leverage.mean() * 100, mom_lev))

print("\n")

# Maybe plot the series of the simulation??? Or at least the CB assets to check what happens

