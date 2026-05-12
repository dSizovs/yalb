#!/usr/bin/env python3
"""Plot density and ux profiles for milestone 4 frames (omega=1)."""
import glob
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

frames = sorted(glob.glob("frame_*.txt"))
if not frames:
    print("No frame_*.txt files. Run ./milestone04 frames first.")
    sys.exit(1)

# Load all frames and pull out ux(y) profile (averaged over x, but since
# the shear wave is x-independent, any x-slice works).
profiles = []
for fname in frames:
    with open(fname) as fh:
        Nx, Ny = map(int, fh.readline().split())
        d = np.loadtxt(fh)
    ux = d[:, 1].reshape(Ny, Nx).mean(axis=1)  # avg over x; shape (Ny,)
    rho = d[:, 0].reshape(Ny, Nx).mean(axis=1)
    profiles.append((rho, ux))

ys = np.arange(profiles[0][0].size)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
cmap = plt.cm.viridis(np.linspace(0, 1, len(profiles)))
for (rho, ux), c, fname in zip(profiles, cmap, frames):
    label = os.path.basename(fname).replace("frame_", "t=").replace(".txt", "")
    ax1.plot(ys, ux, color=c, label=label, linewidth=1)
    ax2.plot(ys, rho, color=c, linewidth=1)

ax1.set_xlabel("y")
ax1.set_ylabel("u_x(y)")
ax1.set_title("Shear-wave velocity decay")
ax1.legend(fontsize=7, ncol=2)
ax1.axhline(0, color="black", linewidth=0.5)

ax2.set_xlabel("y")
ax2.set_ylabel("rho(y)")
ax2.set_title("Density (should stay ≈ 1)")
ax2.axhline(1, color="black", linewidth=0.5, linestyle="--")

plt.tight_layout()
plt.savefig("m4_profiles.png", dpi=100)
print("Saved m4_profiles.png")
