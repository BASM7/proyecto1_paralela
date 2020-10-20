#!/bin/bash
#SBATCH --job-name=Glowworm_l
#SBATCH --output=job_result.txt
#SBATCH --partition=nu
#SBATCH --ntasks=32
#SBATCH --time=12:00:00
module load intelpython/3.5
module load mpich/3.2.1
mpiexec python main.py -r 0.4 -g 0.6 -s 0.03 -i 4.0 -l 5.0