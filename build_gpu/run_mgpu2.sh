#!/bin/bash
#SBATCH --job-name=yalb_mgpu
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=2              # 4 MPI ranks
#SBATCH --gres=gpu:2            # 4 GPUs (one per rank)
#SBATCH --time=00:05:00
#SBATCH --output=mgpu_job_%j.log

module purge
module load compiler/gnu/14.2
module load mpi/openmpi/5.0
module load devel/cuda/12.8

# Map each MPI rank to its own GPU via Kokkos' device selection.
# Kokkos reads --kokkos-map-device-id-by=mpi_rank to assign GPUs round-robin.
mpirun -np 2 ./executables/06/milestone06 1000 2000 1.0 \
    --kokkos-map-device-id-by=mpi_rank

