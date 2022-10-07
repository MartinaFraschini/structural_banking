#!/bin/bash -l

#SBATCH --account mrocking_cbdc
#SBATCH --mail-type ALL
#SBATCH --mail-user martina.fraschini@unil.ch

#SBATCH --chdir /scratch/mfraschi/structural_banking/parallel/02_qe_calib/
#SBATCH --job-name qe_cal
#SBATCH --output=/scratch/mfraschi/structural_banking/parallel/02_qe_calib/out/cal_%A_%a.out
#SBATCH --error=/scratch/mfraschi/structural_banking/parallel/02_qe_calib/out/cal_%A_%a.err

#SBATCH --partition cpu


#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 200MB
#SBATCH --time 00:30:00
#SBATCH --array=0-9

module load gcc/10.4.0 python/3.9.13

ARGS=(0.044 0.045 0.046 0.047 0.048 0.049 0.050 0.051 0.052 0.053)

python3 /scratch/mfraschi/structural_banking/parallel/02_qe_calib/main.py ${ARGS[SLURM_ARRAY_TASK_ID]}
