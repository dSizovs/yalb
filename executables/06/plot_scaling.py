#!/usr/bin/env python3
"""Plot strong-scaling speedup from scaling.txt (cols: nprocs N steps time)."""
import sys
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("scaling.txt", ndmin=2)
# Sort by nprocs.
data = data[np.argsort(data[:, 0])]
procs = data[:, 0]
times = data[:, 3]

# Speedup relative to the 1-rank run.
t1 = times[procs == 1]
if t1.size == 0:
    print("No 1-rank baseline in scaling.txt; cannot compute speedup.")
    sys.exit(1)
t1 = t1[0]
speedup = t1 / times
efficiency = speedup / procs

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Speedup vs ideal.
ax1.plot(procs, speedup, "ro-", label="measured", markersize=8)
ax1.plot(procs, procs, "k--", label="ideal (linear)")
ax1.set_xlabel("number of MPI ranks")
ax1.set_ylabel("speedup  $t_1 / t_p$")
ax1.set_title("Strong scaling: speedup")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Parallel efficiency.
ax2.plot(procs, efficiency * 100, "bo-", markersize=8)
ax2.axhline(100, color="k", linestyle="--", label="ideal (100%)")
ax2.set_xlabel("number of MPI ranks")
ax2.set_ylabel("parallel efficiency  (%)")
ax2.set_title("Strong scaling: efficiency")
ax2.set_ylim(0, 110)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("m6_scaling.png", dpi=120)
print("Saved m6_scaling.png")

# Also print a small table.
print(f"{'ranks':>6} {'time(s)':>10} {'speedup':>9} {'eff(%)':>8}")
for p, t, s, e in zip(procs, times, speedup, efficiency):
    print(f"{int(p):>6} {t:>10.4f} {s:>9.3f} {e*100:>8.1f}")
