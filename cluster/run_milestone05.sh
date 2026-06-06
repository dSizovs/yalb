#!/bin/bash
#SBATCH --job-name=yalb_gpu
#SBATCH --partition=gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --output=gpu_job_%j.log

module purge
module load compiler/gnu/14.2
module load mpi/openmpi/5.0
module load devel/cuda/12.8

# milestone05 args: N  Re  u_lid  tol
./executables/05/milestone05 400 100 0.1 1e-6 --kokkos-print-configuration --kokkos-print-configuration
