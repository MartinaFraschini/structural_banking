#!/bin/bash -l

#SBATCH --account mrocking_cbdc
#SBATCH --mail-type ALL
#SBATCH --mail-user martina.fraschini@unil.ch

#SBATCH --chdir /scratch/mfraschi/structural_banking/parallel/qe_dynamics/
#SBATCH --job-name qe_dyn
#SBATCH --output=/scratch/mfraschi/structural_banking/parallel/qe_dynamics/out/cal_%A_%a.out
#SBATCH --error=/scratch/mfraschi/structural_banking/parallel/qe_dynamics/out/cal_%A_%a.err

#SBATCH --partition cpu


#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem=8GB
#SBATCH --time 01:30:00
#SBATCH --array=0-9

module load gcc/10.4.0 python/3.9.13

ARGS=(8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0 17.0)

python3 /scratch/mfraschi/structural_banking/parallel/qe_dynamics/main.py ${ARGS[SLURM_ARRAY_TASK_ID]}
