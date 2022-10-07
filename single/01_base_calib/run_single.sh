#!/bin/bash -l

#SBATCH --account mrocking_cbdc
#SBATCH --mail-type ALL
#SBATCH --mail-user martina.fraschini@unil.ch

#SBATCH --chdir /scratch/mfraschi/structural_banking/single/01_base_calib/
#SBATCH --job-name base_cal
#SBATCH --output=/scratch/mfraschi/structural_banking/single/01_base_calib/out/cal_%A.out
#SBATCH --error=/scratch/mfraschi/structural_banking/single/01_base_calib/out/cal_%A.err

#SBATCH --partition cpu


#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 200MB
#SBATCH --time 00:30:00

module load gcc/9.3.0 python/3.8.8

python3 /scratch/mfraschi/structural_banking/single/01_base_calib/main_single.py
