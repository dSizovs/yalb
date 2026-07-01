#!/usr/bin/env python3
"""Plot lid-driven cavity results. Usage: python3 plot.py <cavity_file.txt> [Re]"""
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
from ghia_data import GHIA_Y, GHIA_U, GHIA_X, GHIA_V

if len(sys.argv) < 2:
    import glob
    files = sorted(glob.glob("cavity_*.txt"))
    if not files:
        print("Usage: python3 plot.py <cavity_file.txt> [Re]")
        sys.exit(1)
    fname = files[-1]
else:
    fname = sys.argv[1]

# Try to grab Re from filename if not given explicitly
Re = None
if len(sys.argv) >= 3:
    Re = float(sys.argv[2])
else:
    m = re.search(r"Re(\d+)", fname)
    if m: Re = float(m.group(1))

with open(fname) as fh:
    Nx, Ny = map(int, fh.readline().split())
    data = np.loadtxt(fh)
rho = data[:, 0].reshape(Ny, Nx)
ux  = data[:, 1].reshape(Ny, Nx)
uy  = data[:, 2].reshape(Ny, Nx)
u_mag = np.sqrt(ux ** 2 + uy ** 2)

# Lid velocity from the top row.
u_lid = ux[-1, :].mean()
if u_lid == 0:
    u_lid = 0.1  # fallback

# === Plot 1: streamlines over velocity magnitude ===
fig, ax = plt.subplots(figsize=(7, 7))
X, Y = np.meshgrid(np.arange(Nx), np.arange(Ny))
strength = u_mag / u_lid
im = ax.imshow(strength, origin="lower", cmap="viridis",
               extent=[0, 1, 0, 1], aspect="equal")
ax.streamplot(np.linspace(0, 1, Nx), np.linspace(0, 1, Ny),
              ux, uy, color="white", density=1.5, linewidth=0.7)
ax.set_title(f"Lid-driven cavity, Re = {Re:g}" if Re else "Lid-driven cavity")
ax.set_xlabel("x / L")
ax.set_ylabel("y / L")
plt.colorbar(im, ax=ax, label="|u| / u_lid")
plt.tight_layout()
plt.savefig(f"cavity_streamlines_Re{int(Re) if Re else 0}.png", dpi=120)
print(f"Saved cavity_streamlines_Re{int(Re) if Re else 0}.png")

# === Plot 2: centerline profiles vs Ghia ===
if Re is not None and int(Re) in GHIA_U:
    Re_int = int(Re)

    # Centerlines from the simulation: u along x=L/2; v along y=L/2.
    xc = Nx // 2
    yc = Ny // 2
    u_center = ux[:, xc] / u_lid              # u(y) at x=L/2; shape (Ny,)
    v_center = uy[yc, :] / u_lid              # v(x) at y=L/2; shape (Nx,)
    y_norm = np.arange(Ny) / (Ny - 1)
    x_norm = np.arange(Nx) / (Nx - 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(u_center, y_norm, "b-", label="simulation")
    ax1.plot(GHIA_U[Re_int], GHIA_Y, "ko", label="Ghia et al. 1982")
    ax1.set_xlabel("u / u_lid")
    ax1.set_ylabel("y / L")
    ax1.set_title(f"u(y) at x = L/2, Re = {Re_int}")
    ax1.axvline(0, color="gray", lw=0.5)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(x_norm, v_center, "b-", label="simulation")
    ax2.plot(GHIA_X, GHIA_V[Re_int], "ko", label="Ghia et al. 1982")
    ax2.set_xlabel("x / L")
    ax2.set_ylabel("v / u_lid")
    ax2.set_title(f"v(x) at y = L/2, Re = {Re_int}")
    ax2.axhline(0, color="gray", lw=0.5)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"cavity_centerlines_Re{Re_int}.png", dpi=120)
    print(f"Saved cavity_centerlines_Re{Re_int}.png")
