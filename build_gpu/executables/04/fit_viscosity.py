#!/usr/bin/env python3
"""Fit kinematic viscosity from shear-wave amplitude decay; compare to theory.

For each amplitude_omega_<val>.txt file (columns: step, max|ux|):
  ln(amplitude) vs step should be linear with slope = -nu * k^2,
  where k = 2*pi / Ny.
Theory: nu_th(omega) = (1/3) * (1/omega - 1/2).
"""
import glob
import re
import sys
import numpy as np
import matplotlib.pyplot as plt

Ny = 50  # must match the executable
k = 2.0 * np.pi / Ny
k2 = k * k

# Find all amplitude files and extract omega from the filename.
files = sorted(glob.glob("amplitude_omega_*.txt"))
if not files:
    print("No amplitude_omega_*.txt files. Run ./milestone04 sweep first.")
    sys.exit(1)

omegas = []
nu_measured = []
for f in files:
    m = re.search(r"omega_([\d.]+)\.txt$", f)
    if not m: continue
    omega = float(m.group(1))
    data = np.loadtxt(f)
    steps = data[:, 0]
    amp   = data[:, 1]
    # Discard the first few steps (transient) and any zeros at the very end.
    mask = (steps > 50) & (amp > 1e-12)
    if mask.sum() < 10:
        print(f"  skipping {f}: too few useful points")
        continue
    slope, _ = np.polyfit(steps[mask], np.log(amp[mask]), 1)
    nu = -slope / k2
    omegas.append(omega)
    nu_measured.append(nu)
    print(f"omega={omega:.3f}: nu_measured={nu:.5f}")

omegas = np.array(omegas)
nu_measured = np.array(nu_measured)

# Theory.
omega_smooth = np.linspace(0.05, 1.95, 200)
nu_theory_smooth = (1.0 / 3.0) * (1.0 / omega_smooth - 0.5)

# Plot.
plt.figure(figsize=(8, 6))
plt.plot(omega_smooth, nu_theory_smooth, "k-",
         label=r"theory: $\nu = (1/3)(1/\omega - 1/2)$")
plt.plot(omegas, nu_measured, "ro", markersize=8, label="measured")
plt.xlabel(r"$\omega$")
plt.ylabel(r"kinematic viscosity $\nu$")
plt.title("Shear-wave decay: measured vs analytical viscosity")
plt.yscale("log")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("m4_viscosity.png", dpi=100)
print("Saved m4_viscosity.png")
