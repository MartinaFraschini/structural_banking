#!/bin/bash -l

#SBATCH --account mrocking_cbdc
#SBATCH --mail-type ALL
#SBATCH --mail-user martina.fraschini@unil.ch

#SBATCH --chdir /scratch/mfraschi/structural_banking/single/03_qe_cbdc_nofund/
#SBATCH --job-name qe_cbdc
#SBATCH --output=/scratch/mfraschi/structural_banking/single/03_qe_cbdc_nofund/out/cal_%A.out
#SBATCH --error=/scratch/mfraschi/structural_banking/single/03_qe_cbdc_nofund/out/cal_%A.err

#SBATCH --partition cpu


#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 200MB
#SBATCH --time 00:30:00
#SBATCH --array=0-9

module load gcc/10.4.0 python/3.9.13

python3 /scratch/mfraschi/structural_banking/single/03_qe_cbdc_nofund/main.py
