#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=20
#SBATCH --mem=128GB
#SBATCH --time=100:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hector_lm2@hotmail.com

module purge
module load julia

export JULIA_NUM_THREADS=40
julia ../validation/crlb/validation_tsl_crlb_mono_NLS.jl $1
