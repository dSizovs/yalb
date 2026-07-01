#!/bin/bash
# Strong-scaling study: same problem, increasing rank count.
# Usage: ./run_scaling.sh [N] [steps]
set -e

N=${1:-400}
STEPS=${2:-1000}

rm -f scaling.txt

for P in 1 2 4 6 8; do
    echo "=== Running on $P ranks (N=$N, steps=$STEPS) ==="
    mpirun -np $P ./milestone06 $N $STEPS 1.0
done

echo
echo "Done. Timing data in scaling.txt:"
cat scaling.txt
