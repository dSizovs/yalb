#!/usr/bin/env python3
"""Strong-scaling plots from scaling.txt (cols: nprocs N steps time mlups).
Uses median over repeats. Produces MLUPS-vs-p and speedup-vs-p (log-log)."""
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("scaling.txt", ndmin=2)
procs_all = data[:, 0].astype(int)
mlups_all = data[:, 4]

procs = sorted(set(procs_all))
mlups_med = np.array([np.median(mlups_all[procs_all == p]) for p in procs])
procs = np.array(procs)

# speedup relative to 1 rank
m1 = mlups_med[procs == 1][0]
speedup = mlups_med / m1
eff = speedup / procs

# print a table
print(f"{'p':>4} {'MLUPS':>8} {'speedup':>8} {'eff%':>6}")
for p, m, s, e in zip(procs, mlups_med, speedup, eff):
    print(f"{p:>4} {m:>8.2f} {s:>8.2f} {e*100:>6.1f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

# MLUPS vs p
ax1.plot(procs, mlups_med, "o-", color="C0", markersize=7)
ax1.set_xscale("log", base=2); ax1.set_yscale("log", base=2)
ax1.set_xlabel("MPI processes"); ax1.set_ylabel("MLUPS")
ax1.set_title("Throughput (MLUPS) vs processes")
ax1.grid(True, which="both", alpha=0.3)

# speedup vs p, log-log, with ideal
ax2.plot(procs, speedup, "o-", color="C3", markersize=7, label="measured")
ax2.plot(procs, procs, "k--", label="ideal (linear)")
ax2.set_xscale("log", base=2); ax2.set_yscale("log", base=2)
ax2.set_xlabel("MPI processes"); ax2.set_ylabel(r"speedup $T_1/T_p$")
ax2.set_title("Strong scaling")
ax2.legend(); ax2.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig("m6_scaling_mlups.png", dpi=120)
print("Saved m6_scaling_mlups.png")
