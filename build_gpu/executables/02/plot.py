#!/usr/bin/env python3
"""Plot the density blob moving across the grid for milestone 2."""
import glob
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

frames = sorted(glob.glob("frame_*.txt"))
if not frames:
    print("No frame_*.txt files found. Did you run ./milestone02 first?")
    sys.exit(1)

fig, axes = plt.subplots(4, 5, figsize=(15, 10))
for ax, fname in zip(axes.flat, frames):
    with open(fname) as fh:
        Nx, Ny = map(int, fh.readline().split())
        data = np.loadtxt(fh)
    rho = data[:, 0].reshape(Ny, Nx)
    ax.imshow(rho, origin="lower", cmap="viridis", vmin=0, vmax=1)
    ax.set_title(os.path.basename(fname))
    ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
plt.savefig("streaming.png", dpi=100)
print("Saved streaming.png")
