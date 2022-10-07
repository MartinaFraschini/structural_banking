#!/bin/bash -l

#SBATCH --account mrocking_cbdc
#SBATCH --mail-type ALL
#SBATCH --mail-user martina.fraschini@unil.ch

#SBATCH --chdir /scratch/mfraschi/structural_banking/parallel/04_qe_cbdc_fund/
#SBATCH --job-name qe_cbdc
#SBATCH --output=/scratch/mfraschi/structural_banking/parallel/04_qe_cbdc_fund/out/cal_%A_%a.out
#SBATCH --error=/scratch/mfraschi/structural_banking/parallel/04_qe_cbdc_fund/out/cal_%A_%a.err

#SBATCH --partition cpu


#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 200MB
#SBATCH --time 00:30:00
#SBATCH --array=0-9

module load gcc/10.4.0 python/3.9.13

ARGS=(-0.030 -0.029 -0.028 -0.027 -0.026 -0.025 -0.024 -0.023 -0.022 -0.021)
### ARGS=(-0.020 -0.019 -0.018 -0.017 -0.016 -0.015 -0.014 -0.013 -0.012 -0.011)
### ARGS=(-0.010 -0.009 -0.008 -0.007 -0.006 -0.005 -0.004 -0.003 -0.002 -0.001)
### ARGS=(0.000 0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009)
### ARGS=(0.010 0.011 0.012 0.013 0.014 0.015 0.016 0.017 0.018 0.019)
### ARGS=(0.020 0.021 0.022 0.023 0.024 0.025 0.026 0.027 0.028 0.029)
### ARGS=(0.030 0.031 0.032 0.033 0.034 0.035 0.036 0.037 0.038 0.039)

python3 /scratch/mfraschi/structural_banking/parallel/04_qe_cbdc_fund/main.py ${ARGS[SLURM_ARRAY_TASK_ID]}
