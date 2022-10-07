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
gamma = 100
rCdelta = np.arange(-30, 40)

# Define function to check when model collapses
def check(row):
    row["equity"][0] = row["lending"][0] + row["reserves"][0] - row["deposits"][0] - row["CB_fund"][0]
    row["leverage"][0] = row["equity"][0] / (row["lending"][0] + row["reserves"][0])
    row["dividends"][0] = row["profits"][0] - row["equity"][0]
    row["roe"][0] = (row["profits"][0] - row["equity"][0]) / row["equity"][0]
    row["CB_assets"][0] = row["reserves"][0] + row["cbdc"][0] - row["CB_fund"][0]

    if row["dividends"][0] < 0 or row["leverage"][0] > 0.6 or row["deposits"][0] < 5:
        row["rD"][0] = 0
        row["deposits"][0] = 0
        row["leverage"][0] = 0
        row["lending"][0] = 0
        row["roe"][0] = 0
        row["equity"][0] = 0
        row["dividends"][0] = 0
        row["profits"][0] = 0
        row["reserves"][0] = 0
        row["CB_assets"][0] = row["cbdc"][0]
  
    return row

# Load data for each gamma
rC_path = './res/rF-rC_gamma-' + str(gamma) + '/'
rD_path = './res/rF-rD_gamma-' + str(gamma) + '/'
rM_path = './res/rF-rM_gamma-' + str(gamma) + '/'

plot_path = './plot/rF_gamma-' + str(gamma) + '/'

col = ["rD", "rL", "deposits", "cbdc", "leverage", "lending", "roe", "equity", "dividends", "profits", "reserves", 
       "transfer", "CB_funding", "CB_assets", "borr_ret", "eq_iss", "int_margin", "def_freq"]
rFrC = pd.DataFrame(columns=col)
rFrD = pd.DataFrame(columns=col)
rFrM = pd.DataFrame(columns=col)

for ind in rCdelta:
    rowrC_npy = np.load(rC_path+str(ind+1000)+'.npy')
    rowrC = pd.DataFrame(rowrC_npy.reshape((1, len(rowrC_npy))), columns=col)
    # rowrC = check(rowrC)
    rowrC.index = [ind/1000]
    rFrC = pd.concat([rFrC, rowrC])

    rowrD_npy = np.load(rD_path+str(ind+1000)+'.npy')
    rowrD = pd.DataFrame(rowrD_npy.reshape((1, len(rowrD_npy))), columns=col)
    # rowrD = check(rowrD)
    rowrD.index = [ind/1000]
    rFrD = pd.concat([rFrD, rowrD])

    rowrM_npy = np.load(rM_path+str(ind+1000)+'.npy')
    rowrM = pd.DataFrame(rowrM_npy.reshape((1, len(rowrM_npy))), columns=col)
    # rowrM = check(rowrM)
    rowrM.index = [ind/1000]
    rFrM = pd.concat([rFrM, rowrM])

rFrC["tot_dep"] = rFrC["deposits"] + rFrC["cbdc"]
rFrD["tot_dep"] = rFrD["deposits"] + rFrD["cbdc"]
rFrM["tot_dep"] = rFrM["deposits"] + rFrM["cbdc"]

# Set up for plots
rCdelta_name = r'$r^C$'

legend_list = [r'$r^F = r^C$', \
               r'$r^F = r^D$', \
               r'$r^F = r^M$']

# Create moving average series
roll_rC = 6
roll_rD = 6
roll_rM = 6

rFrC_MA = rFrC.rolling(roll_rC).mean()
rFrD_MA = rFrD.rolling(roll_rD).mean()
rFrM_MA = rFrM.rolling(roll_rM).mean()
#gamma0_MA.iloc[0:roll-1] = gamma0.iloc[0:roll-1]

# Plot moments
fig, axs = plt.subplots(4, 3)

axs[0, 0].plot(rFrC_MA["deposits"], color='tab:blue', linestyle='-')
axs[0, 0].plot(rFrD_MA["deposits"], color='tab:orange', linestyle=':')
axs[0, 0].plot(rFrM_MA["deposits"], color='tab:green', linestyle='-.')
#axs[0, 0].set_ylim(-5, 105)
axs[0, 0].set_xlabel(rCdelta_name)
axs[0, 0].set_ylabel('Bank deposits')
axs[0, 0].legend(legend_list)

axs[0, 1].plot(rFrC_MA["cbdc"], color='tab:blue', linestyle='-')
axs[0, 1].plot(rFrD_MA["cbdc"], color='tab:orange', linestyle=':')
axs[0, 1].plot(rFrM_MA["cbdc"], color='tab:green', linestyle='-.')
#axs[0, 1].set_ylim(-5, 105)
axs[0, 1].set_xlabel(rCdelta_name)
axs[0, 1].set_ylabel('CBDC deposits')

axs[0, 2].plot(rFrC_MA["tot_dep"], color='tab:blue', linestyle='-')
axs[0, 2].plot(rFrD_MA["tot_dep"], color='tab:orange', linestyle=':')
axs[0, 2].plot(rFrM_MA["tot_dep"], color='tab:green', linestyle='-.')
#axs[0, 2].set_ylim(-5, 105)
axs[0, 2].set_xlabel(rCdelta_name)
axs[0, 2].set_ylabel('Total deposits')

axs[1, 0].plot(rFrC_MA["rD"], color='tab:blue', linestyle='-')
axs[1, 0].plot(rFrD_MA["rD"], color='tab:orange', linestyle=':')
axs[1, 0].plot(rFrM_MA["rD"], color='tab:green', linestyle='-.')
#axs[1, 0].set_ylim(1.3, 2.3)
axs[1, 0].set_xlabel(rCdelta_name)
axs[1, 0].set_ylabel(r'$r^D$')

axs[1, 1].plot(rFrC_MA["rL"], color='tab:blue', linestyle='-')
axs[1, 1].plot(rFrD_MA["rL"], color='tab:orange', linestyle=':')
axs[1, 1].plot(rFrM_MA["rL"], color='tab:green', linestyle='-.')
#axs[1, 1].set_ylim(3, 3.6)
axs[1, 1].set_xlabel(rCdelta_name)
axs[1, 1].set_ylabel(r'$r^L$')

axs[1, 2].plot(rFrC_MA["int_margin"], color='tab:blue', linestyle='-')
axs[1, 2].plot(rFrD_MA["int_margin"], color='tab:orange', linestyle=':')
axs[1, 2].plot(rFrM_MA["int_margin"], color='tab:green', linestyle='-.')
#axs[1, 2].set_ylim(1, 2.2)
axs[1, 2].set_xlabel(rCdelta_name)
axs[1, 2].set_ylabel('Interest rate margin')

axs[2, 0].plot(rFrC_MA["roe"], color='tab:blue', linestyle='-')
axs[2, 0].plot(rFrD_MA["roe"], color='tab:orange', linestyle=':')
axs[2, 0].plot(rFrM_MA["roe"], color='tab:green', linestyle='-.')
axs[2, 0].set_xlabel(rCdelta_name)
axs[2, 0].set_ylabel('ROE')

axs[2, 1].plot(rFrC_MA["lending"], color='tab:blue', linestyle='-')
axs[2, 1].plot(rFrD_MA["lending"], color='tab:orange', linestyle=':')
axs[2, 1].plot(rFrM_MA["lending"], color='tab:green', linestyle='-.')
#axs[2, 1].set_ylim(38, 42)
axs[2, 1].set_xlabel(rCdelta_name)
axs[2, 1].set_ylabel('Lending')

axs[2, 2].plot(rFrC_MA["leverage"], color='tab:blue', linestyle='-')
axs[2, 2].plot(rFrD_MA["leverage"], color='tab:orange', linestyle=':')
axs[2, 2].plot(rFrM_MA["leverage"], color='tab:green', linestyle='-.')
axs[2, 2].set_xlabel(rCdelta_name)
axs[2, 2].set_ylabel('Leverage')

axs[3, 0].plot(rFrC_MA["reserves"], color='tab:blue', linestyle='-')
axs[3, 0].plot(rFrD_MA["reserves"], color='tab:orange', linestyle=':')
axs[3, 0].plot(rFrM_MA["reserves"], color='tab:green', linestyle='-.')
axs[3, 0].set_xlabel(rCdelta_name)
axs[3, 0].set_ylabel('Reserves')

axs[3, 1].plot(rFrC_MA["CB_funding"], color='tab:blue', linestyle='-')
axs[3, 1].plot(rFrD_MA["CB_funding"], color='tab:orange', linestyle=':')
axs[3, 1].plot(rFrM_MA["CB_funding"], color='tab:green', linestyle='-.')
axs[3, 1].set_xlabel(rCdelta_name)
axs[3, 1].set_ylabel('Central bank funding')

axs[3, 2].plot(rFrC_MA["CB_assets"], color='tab:blue', linestyle='-')
axs[3, 2].plot(rFrD_MA["CB_assets"], color='tab:orange', linestyle=':')
axs[3, 2].plot(rFrM_MA["CB_assets"], color='tab:green', linestyle='-.')
axs[3, 2].set_xlabel(rCdelta_name)
axs[3, 2].set_ylabel('Central bank assets')

fig.set_size_inches(19, 15)
fig.tight_layout()
fig.savefig(plot_path+'ALL.pdf', dpi=100)


fig, axs = plt.subplots()
axs.plot(rFrC_MA["deposits"], color='tab:blue', linestyle='-')
axs.plot(rFrD_MA["deposits"], color='tab:orange', linestyle=':')
axs.plot(rFrM_MA["deposits"], color='tab:green', linestyle='-.')
#axs.set_ylim(-5, 105)
axs.set_xlabel(rCdelta_name)
axs.set_ylabel('Bank deposits')
fig.set_size_inches(10, 6)
fig.tight_layout()
fig.savefig(plot_path+'deposits.pdf', dpi=100)

fig, axs = plt.subplots()
axs.plot(rFrC_MA["cbdc"], color='tab:blue', linestyle='-')
axs.plot(rFrD_MA["cbdc"], color='tab:orange', linestyle=':')
axs.plot(rFrM_MA["cbdc"], color='tab:green', linestyle='-.')
#axs.set_ylim(-5, 105)
axs.set_xlabel(rCdelta_name)
axs.set_ylabel('CBDC deposits')
fig.set_size_inches(10, 6)
fig.tight_layout()
fig.savefig(plot_path+'cbdc.pdf', dpi=100)

fig, axs = plt.subplots()
axs.plot(rFrC_MA["tot_dep"], color='tab:blue', linestyle='-')
axs.plot(rFrD_MA["tot_dep"], color='tab:orange', linestyle=':')
axs.plot(rFrM_MA["tot_dep"], color='tab:green', linestyle='-.')
#axs.set_ylim(-5, 105)
axs.set_xlabel(rCdelta_name)
axs.set_ylabel('Total deposits')
fig.set_size_inches(10, 6)
fig.tight_layout()
fig.savefig(plot_path+'totdep.pdf', dpi=100)

fig, axs = plt.subplots()
axs.plot(rFrC_MA["rD"], color='tab:blue', linestyle='-')
axs.plot(rFrD_MA["rD"], color='tab:orange', linestyle=':')
axs.plot(rFrM_MA["rD"], color='tab:green', linestyle='-.')
#axs.set_ylim(1.3, 2.3)
axs.set_xlabel(rCdelta_name)
axs.set_ylabel(r'$r^D$')
fig.set_size_inches(10, 6)
fig.tight_layout()
fig.savefig(plot_path+'rD.pdf', dpi=100)

fig, axs = plt.subplots()
axs.plot(rFrC_MA["rL"], color='tab:blue', linestyle='-')
axs.plot(rFrD_MA["rL"], color='tab:orange', linestyle=':')
axs.plot(rFrM_MA["rL"], color='tab:green', linestyle='-.')
#axs.set_ylim(3, 3.6)
axs.set_xlabel(rCdelta_name)
axs.set_ylabel(r'$r^L$')
fig.set_size_inches(10, 6)
fig.tight_layout()
fig.savefig(plot_path+'rL.pdf', dpi=100)

fig, axs = plt.subplots()
axs.plot(rFrC_MA["int_margin"], color='tab:blue', linestyle='-')
axs.plot(rFrD_MA["int_margin"], color='tab:orange', linestyle=':')
axs.plot(rFrM_MA["int_margin"], color='tab:green', linestyle='-.')
#axs.set_ylim(1, 2.2)
axs.set_xlabel(rCdelta_name)
axs.set_ylabel('Interest rate margin')
fig.set_size_inches(10, 6)
fig.tight_layout()
fig.savefig(plot_path+'int_margin.pdf', dpi=100)

fig, axs = plt.subplots()
axs.plot(rFrC_MA["roe"], color='tab:blue', linestyle='-')
axs.plot(rFrD_MA["roe"], color='tab:orange', linestyle=':')
axs.plot(rFrM_MA["roe"], color='tab:green', linestyle='-.')
axs.set_xlabel(rCdelta_name)
axs.set_ylabel('ROE')
fig.set_size_inches(10, 6)
fig.tight_layout()
fig.savefig(plot_path+'roe.pdf', dpi=100)

fig, axs = plt.subplots()
axs.plot(rFrC_MA["lending"], color='tab:blue', linestyle='-')
axs.plot(rFrD_MA["lending"], color='tab:orange', linestyle=':')
axs.plot(rFrM_MA["lending"], color='tab:green', linestyle='-.')
#axs.set_ylim(38, 42)
axs.set_xlabel(rCdelta_name)
axs.set_ylabel('Lending')
fig.set_size_inches(10, 6)
fig.tight_layout()
fig.savefig(plot_path+'lending.pdf', dpi=100)

fig, axs = plt.subplots()
axs.plot(rFrC_MA["leverage"], color='tab:blue', linestyle='-')
axs.plot(rFrD_MA["leverage"], color='tab:orange', linestyle=':')
axs.plot(rFrM_MA["leverage"], color='tab:green', linestyle='-.')
axs.set_xlabel(rCdelta_name)
axs.set_ylabel('Leverage')
fig.set_size_inches(10, 6)
fig.tight_layout()
fig.savefig(plot_path+'leverage.pdf', dpi=100)

fig, axs = plt.subplots()
axs.plot(rFrC_MA["reserves"], color='tab:blue', linestyle='-')
axs.plot(rFrD_MA["reserves"], color='tab:orange', linestyle=':')
axs.plot(rFrM_MA["reserves"], color='tab:green', linestyle='-.')
axs.set_xlabel(rCdelta_name)
axs.set_ylabel('Reserves')
fig.set_size_inches(10, 6)
fig.tight_layout()
fig.savefig(plot_path+'reserves.pdf', dpi=100)

fig, axs = plt.subplots()
axs.plot(rFrC_MA["CB_funding"], color='tab:blue', linestyle='-')
axs.plot(rFrD_MA["CB_funding"], color='tab:orange', linestyle=':')
axs.plot(rFrM_MA["CB_funding"], color='tab:green', linestyle='-.')
axs.set_xlabel(rCdelta_name)
axs.set_ylabel('Central bank funding')
fig.set_size_inches(10, 6)
fig.tight_layout()
fig.savefig(plot_path+'CB_funding.pdf', dpi=100)

fig, axs = plt.subplots()
axs.plot(rFrC_MA["CB_assets"], color='tab:blue', linestyle='-')
axs.plot(rFrD_MA["CB_assets"], color='tab:orange', linestyle=':')
axs.plot(rFrM_MA["CB_assets"], color='tab:green', linestyle='-.')
axs.set_xlabel(rCdelta_name)
axs.set_ylabel('Central bank assets')
fig.set_size_inches(10, 6)
fig.tight_layout()
fig.savefig(plot_path+'CB_assets.pdf', dpi=100)

