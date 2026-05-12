#!/usr/bin/env python3
"""Plot milestone 3 frames. Default field: density. Pass 'ux' or 'uy' to plot velocity."""
import glob
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

field = sys.argv[1] if len(sys.argv) > 1 else "rho"
field_idx = {"rho": 0, "ux": 1, "uy": 2}.get(field)
if field_idx is None:
    print(f"Unknown field {field!r}. Use rho, ux, or uy.")
    sys.exit(1)

frames = sorted(glob.glob("frame_*.txt"))
if not frames:
    print("No frame_*.txt files found. Run ./milestone03 first.")
    sys.exit(1)

data_stack = []
for fname in frames:
    with open(fname) as fh:
        Nx, Ny = map(int, fh.readline().split())
        d = np.loadtxt(fh)
    data_stack.append(d[:, field_idx].reshape(Ny, Nx))
stack = np.array(data_stack)

vmin, vmax = stack.min(), stack.max()
if field in ("ux", "uy"):
    vabs = max(abs(vmin), abs(vmax))
    vmin, vmax = -vabs, vabs
else:
    # Don't let tiny noise blow up the color scale.
    if vmax - vmin < 1e-3:
        mid = 0.5 * (vmin + vmax)
        vmin, vmax = mid - 5e-4, mid + 5e-4

n_panels = min(8, len(frames))
sel = np.linspace(0, len(frames) - 1, n_panels, dtype=int)

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
cmap = "viridis" if field == "rho" else "RdBu_r"
for ax, idx in zip(axes.flat, sel):
    im = ax.imshow(stack[idx], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(os.path.basename(frames[idx]))
    ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label=field)

out = f"m3_{field}.png"
plt.savefig(out, dpi=100, bbox_inches="tight")
print(f"Saved {out}")
