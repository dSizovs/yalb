#!/bin/bash
#SBATCH --job-name=yalb_scale
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --time=00:30:00
#SBATCH --output=scaling_%j.log

module purge
module load compiler/gnu/14.2
module load mpi/openmpi/5.0

cd /home/fr/fr_fr/fr_ds722/yalb/build_cpu/executables/06
rm -f scaling.txt

GRID=400
STEPS=100
for P in 1 2 4 8 16 32 64; do
  for REP in 1 2 3; do
    echo "=== p=$P rep=$REP ==="
    mpirun -np $P ./milestone06 $GRID $STEPS 1.0
  done
done

echo "=== scaling.txt ==="
cat scaling.txt
