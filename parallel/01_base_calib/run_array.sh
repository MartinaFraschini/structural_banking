#!/bin/bash -l

#SBATCH --account mrocking_cbdc
#SBATCH --mail-type ALL
#SBATCH --mail-user martina.fraschini@unil.ch

#SBATCH --chdir /scratch/mfraschi/structural_banking/parallel/01_base_calib/
#SBATCH --job-name base_cal
#SBATCH --output=/scratch/mfraschi/structural_banking/parallel/01_base_calib/out/cal_%A_%a.out
#SBATCH --error=/scratch/mfraschi/structural_banking/parallel/01_base_calib/out/cal_%A_%a.err

#SBATCH --partition cpu


#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 200MB
#SBATCH --time 00:30:00
#SBATCH --array=0-9

module load gcc/10.4.0 python/3.9.13

ARGS=(2.7 2.8 2.9 3.0 3.1 3.2 3.3 3.4 3.5 3.6)

python3 /scratch/mfraschi/structural_banking/parallel/01_base_calib/main.py ${ARGS[SLURM_ARRAY_TASK_ID]}
