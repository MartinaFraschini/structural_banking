"""Description...
"""

# Import packages
import os
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

plt.rcParams.update({'font.size': 14})
plt.rc('axes', labelsize=18)

# Grids for rCdelta
rF_type = 'rC'
rCdelta = np.arange(-30, 40)


# Load data for each gamma
path = './res/gamma-'

gamma0_path = '../qe_cbdc_fund/res/rF-rM_gamma-0/'
gamma1_path = './res/gamma-0/'

plot_path = './plot/comp/'

col = ["rD", "rL", "deposits", "cbdc", "leverage", "lending", "roe", "equity", "dividends", "profits", "reserves", 
       "transfer", "CB_funding", "CB_assets", "borr_ret", "eq_iss", "int_margin", "def_freq"]
gamma0 = pd.DataFrame(columns=col)
gamma1 = pd.DataFrame(columns=col)
gamma2 = pd.DataFrame(columns=col)

for ind in rCdelta:
    row0_npy = np.load(gamma0_path+str(ind+1000)+'.npy')
    row0 = pd.DataFrame(row0_npy.reshape((1, len(row0_npy))), columns=col)
    row0.index = [ind/1000]
    gamma0 = pd.concat([gamma0, row0])

    row1_npy = np.load(gamma1_path+str(ind+1000)+'.npy')
    row1 = pd.DataFrame(row1_npy.reshape((1, len(row1_npy))), columns=col)
    row1.index = [ind/1000]
    gamma1 = pd.concat([gamma1, row1])

gamma0["tot_dep"] = gamma0["deposits"] + gamma0["cbdc"]
gamma1["tot_dep"] = gamma1["deposits"] + gamma1["cbdc"]

# Check gamma0
gamma1.iloc[31] = (gamma1.iloc[30] + gamma1.iloc[32]) / 2

# Set up for plots
rCdelta_name = r'$r^C$'

legend_list = ['CB funding', 'No CB funding']

# Create moving average series
roll_gamma0 = 6
roll_gamma1 = 6

gamma0_MA = gamma0.rolling(roll_gamma0).mean()
gamma1_MA = gamma1.rolling(roll_gamma1).mean()


# Plot moments
fig, axs = plt.subplots(4, 3)

axs[0, 0].plot(gamma0_MA["deposits"], color='tab:blue', linestyle='-')
axs[0, 0].plot(gamma1_MA["deposits"], color='tab:green', linestyle='-.')
axs[0, 0].set_xlabel(rCdelta_name)
axs[0, 0].set_ylabel('Bank deposits')
axs[0, 0].legend(legend_list)

axs[0, 1].plot(gamma0_MA["cbdc"], color='tab:blue', linestyle='-')
axs[0, 1].plot(gamma1_MA["cbdc"], color='tab:green', linestyle='-.')
axs[0, 1].set_xlabel(rCdelta_name)
axs[0, 1].set_ylabel('CBDC deposits')

axs[0, 2].plot(gamma0_MA["tot_dep"], color='tab:blue', linestyle='-')
axs[0, 2].plot(gamma1_MA["tot_dep"], color='tab:green', linestyle='-.')
axs[0, 2].set_xlabel(rCdelta_name)
axs[0, 2].set_ylabel('Total deposits')

axs[1, 0].plot(gamma0_MA["rD"], color='tab:blue', linestyle='-')
axs[1, 0].plot(gamma1_MA["rD"], color='tab:green', linestyle='-.')
axs[1, 0].set_xlabel(rCdelta_name)
axs[1, 0].set_ylabel(r'$r^D$')

axs[1, 1].plot(gamma0_MA["rL"], color='tab:blue', linestyle='-')
axs[1, 1].plot(gamma1_MA["rL"], color='tab:green', linestyle='-.')
axs[1, 1].set_xlabel(rCdelta_name)
axs[1, 1].set_ylabel(r'$r^L$')

axs[1, 2].plot(gamma0_MA["int_margin"], color='tab:blue', linestyle='-')
axs[1, 2].plot(gamma1_MA["int_margin"], color='tab:green', linestyle='-.')
axs[1, 2].set_xlabel(rCdelta_name)
axs[1, 2].set_ylabel('Interest rate margin')

axs[2, 0].plot(gamma0_MA["roe"], color='tab:blue', linestyle='-')
axs[2, 0].plot(gamma1_MA["roe"], color='tab:green', linestyle='-.')
axs[2, 0].set_xlabel(rCdelta_name)
axs[2, 0].set_ylabel('ROE')

axs[2, 1].plot(gamma0_MA["lending"], color='tab:blue', linestyle='-')
axs[2, 1].plot(gamma1_MA["lending"], color='tab:green', linestyle='-.')
axs[2, 1].set_xlabel(rCdelta_name)
axs[2, 1].set_ylabel('Lending')

axs[2, 2].plot(gamma0_MA["leverage"], color='tab:blue', linestyle='-')
axs[2, 2].plot(gamma1_MA["leverage"], color='tab:green', linestyle='-.')
axs[2, 2].set_xlabel(rCdelta_name)
axs[2, 2].set_ylabel('Leverage')

axs[3, 0].plot(gamma0_MA["reserves"], color='tab:blue', linestyle='-')
axs[3, 0].plot(gamma1_MA["reserves"], color='tab:green', linestyle='-.')
axs[3, 0].set_xlabel(rCdelta_name)
axs[3, 0].set_ylabel('Reserves')

axs[3, 1].plot(gamma0_MA["CB_funding"], color='tab:blue', linestyle='-')
axs[3, 1].plot(gamma1_MA["CB_funding"], color='tab:green', linestyle='-.')
axs[3, 1].set_xlabel(rCdelta_name)
axs[3, 1].set_ylabel('Central bank funding')

axs[3, 2].plot(gamma0_MA["CB_assets"], color='tab:blue', linestyle='-')
axs[3, 2].plot(gamma1_MA["CB_assets"], color='tab:green', linestyle='-.')
axs[3, 2].set_xlabel(rCdelta_name)
axs[3, 2].set_ylabel('Central bank assets')

fig.set_size_inches(19, 15)
fig.tight_layout()
fig.savefig(plot_path+'ALL.pdf', dpi=100)


fig, axs = plt.subplots()
axs.plot(gamma0_MA["deposits"], color='tab:blue', linestyle='-')
axs.plot(gamma1_MA["deposits"], color='tab:green', linestyle='-.')
axs.set_xlabel(rCdelta_name)
axs.set_ylabel('Bank deposits')
fig.set_size_inches(10, 6)
fig.tight_layout()
fig.savefig(plot_path+'deposits.pdf', dpi=100)

fig, axs = plt.subplots()
axs.plot(gamma0_MA["cbdc"], color='tab:blue', linestyle='-')
axs.plot(gamma1_MA["cbdc"], color='tab:green', linestyle='-.')
axs.set_xlabel(rCdelta_name)
axs.set_ylabel('CBDC deposits')
fig.set_size_inches(10, 6)
fig.tight_layout()
fig.savefig(plot_path+'cbdc.pdf', dpi=100)

fig, axs = plt.subplots()
axs.plot(gamma0_MA["tot_dep"], color='tab:blue', linestyle='-')
axs.plot(gamma1_MA["tot_dep"], color='tab:green', linestyle='-.')
axs.set_xlabel(rCdelta_name)
axs.set_ylabel('Total deposits')
fig.set_size_inches(10, 6)
fig.tight_layout()
fig.savefig(plot_path+'totdep.pdf', dpi=100)

fig, axs = plt.subplots()
axs.plot(gamma0_MA["rD"], color='tab:blue', linestyle='-')
axs.plot(gamma1_MA["rD"], color='tab:green', linestyle='-.')
axs.set_xlabel(rCdelta_name)
axs.set_ylabel(r'$r^D$')
fig.set_size_inches(10, 6)
fig.tight_layout()
fig.savefig(plot_path+'rD.pdf', dpi=100)

fig, axs = plt.subplots()
axs.plot(gamma0_MA["rL"], color='tab:blue', linestyle='-')
axs.plot(gamma1_MA["rL"], color='tab:green', linestyle='-.')
axs.set_xlabel(rCdelta_name)
axs.set_ylabel(r'$r^L$')
fig.set_size_inches(10, 6)
fig.tight_layout()
fig.savefig(plot_path+'rL.pdf', dpi=100)

fig, axs = plt.subplots()
axs.plot(gamma0_MA["int_margin"], color='tab:blue', linestyle='-')
axs.plot(gamma1_MA["int_margin"], color='tab:green', linestyle='-.')
axs.set_xlabel(rCdelta_name)
axs.set_ylabel('Interest rate margin')
fig.set_size_inches(10, 6)
fig.tight_layout()
fig.savefig(plot_path+'int_margin.pdf', dpi=100)

fig, axs = plt.subplots()
axs.plot(gamma0_MA["roe"], color='tab:blue', linestyle='-')
axs.plot(gamma1_MA["roe"], color='tab:green', linestyle='-.')
axs.set_xlabel(rCdelta_name)
axs.set_ylabel('ROE')
fig.set_size_inches(10, 6)
fig.tight_layout()
fig.savefig(plot_path+'roe.pdf', dpi=100)

fig, axs = plt.subplots()
axs.plot(gamma0_MA["lending"], color='tab:blue', linestyle='-')
axs.plot(gamma1_MA["lending"], color='tab:green', linestyle='-.')
axs.set_xlabel(rCdelta_name)
axs.set_ylabel('Lending')
fig.set_size_inches(10, 6)
fig.tight_layout()
fig.savefig(plot_path+'lending.pdf', dpi=100)

fig, axs = plt.subplots()
axs.plot(gamma0_MA["leverage"], color='tab:blue', linestyle='-')
axs.plot(gamma1_MA["leverage"], color='tab:green', linestyle='-.')
axs.set_xlabel(rCdelta_name)
axs.set_ylabel('Leverage')
fig.set_size_inches(10, 6)
fig.tight_layout()
fig.savefig(plot_path+'leverage.pdf', dpi=100)

fig, axs = plt.subplots()
axs.plot(gamma0_MA["reserves"], color='tab:blue', linestyle='-')
axs.plot(gamma1_MA["reserves"], color='tab:green', linestyle='-.')
axs.set_xlabel(rCdelta_name)
axs.set_ylabel('Reserves')
fig.set_size_inches(10, 6)
fig.tight_layout()
fig.savefig(plot_path+'reserves.pdf', dpi=100)

fig, axs = plt.subplots()
axs.plot(gamma0_MA["CB_funding"], color='tab:blue', linestyle='-')
axs.plot(gamma1_MA["CB_funding"], color='tab:green', linestyle='-.')
axs.set_xlabel(rCdelta_name)
axs.set_ylabel('Central bank funding')
fig.set_size_inches(10, 6)
fig.tight_layout()
fig.savefig(plot_path+'CB_funding.pdf', dpi=100)

fig, axs = plt.subplots()
axs.plot(gamma0_MA["CB_assets"], color='tab:blue', linestyle='-')
axs.plot(gamma1_MA["CB_assets"], color='tab:green', linestyle='-.')
axs.set_xlabel(rCdelta_name)
axs.set_ylabel('Central bank assets')
fig.set_size_inches(10, 6)
fig.tight_layout()
fig.savefig(plot_path+'CB_assets.pdf', dpi=100)

