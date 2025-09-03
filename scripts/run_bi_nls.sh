#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=16GB
#SBATCH --time=6:00:00

module purge
module load julia

julia ../validation/crlb/validation_tsl_crlb_biexp_NLS.jl