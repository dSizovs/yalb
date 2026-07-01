#!/bin/bash
#SBATCH --job-name=yalb_anim
#SBATCH --partition=gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --output=anim_%j.log

module purge
module load compiler/gnu/14.2
module load mpi/openmpi/5.0
module load devel/cuda/12.8

cd /home/fr/fr_fr/fr_ds722/yalb/build_gpu/executables/02
rm -f frame_*.txt
./milestone02
echo "frames written:"
ls frame_*.txt | wc -l
