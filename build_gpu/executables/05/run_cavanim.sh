#!/bin/bash
#SBATCH --job-name=yalb_cav
#SBATCH --partition=gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --output=cavanim_%j.log

module purge
module load compiler/gnu/14.2
module load mpi/openmpi/5.0
module load devel/cuda/12.8

cd /home/fr/fr_fr/fr_ds722/yalb/build_gpu/executables/05
rm -f cavframe_*.txt
# args: N Re u_lid tol
./milestone05 200 100 0.1 1e-6
echo "frames written:"
ls cavframe_*.txt | wc -l
