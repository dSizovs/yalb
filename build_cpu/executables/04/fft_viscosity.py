#!/usr/bin/env python3
"""FFT-based viscosity from shear-wave decay.

Each profile_omega_<w>.txt has rows: step  ux(y=0) ux(y=1) ... ux(y=Ny-1).
The initial condition is a pure sin mode k=1. We FFT ux(y) each step and
track the amplitude of mode 1, which decays as exp(-nu * k^2 * t) with
k = 2*pi/Ny. Fit ln(amp) vs step -> nu. Compare to theory (1/3)(1/w-1/2).
"""
import glob, re, sys
import numpy as np
import matplotlib.pyplot as plt

files = sorted(glob.glob("profile_omega_*.txt"))
if not files:
    print("No profile_omega_*.txt; run ./milestone04 sweep first.")
    sys.exit(1)

omegas, nu_fft = [], []
for f in files:
    m = re.search(r"omega_([\d.]+)\.txt$", f)
    if not m: continue
    w = float(m.group(1))
    data = np.loadtxt(f)
    steps = data[:, 0]
    prof  = data[:, 1:]           # shape (nsteps, Ny)
    Ny = prof.shape[1]
    k = 2.0 * np.pi / Ny
    # amplitude of Fourier mode 1 at each step
    fft = np.fft.rfft(prof, axis=1)
    amp1 = np.abs(fft[:, 1]) * (2.0 / Ny)   # physical amplitude of mode 1
    mask = (steps > 50) & (amp1 > 1e-12)
    if mask.sum() < 10:
        continue
    slope, _ = np.polyfit(steps[mask], np.log(amp1[mask]), 1)
    nu = -slope / (k * k)
    omegas.append(w); nu_fft.append(nu)
    print(f"omega={w:.3f}: nu_fft={nu:.6f}")

omegas = np.array(omegas); nu_fft = np.array(nu_fft)
w_s = np.linspace(0.05, 1.95, 300)
nu_th = (1.0/3.0)*(1.0/w_s - 0.5)

plt.figure(figsize=(8,6))
plt.plot(w_s, nu_th, "k-", label=r"theory: $\nu=(1/3)(1/\omega-1/2)$")
plt.plot(omegas, nu_fft, "o", color="C3", markersize=8,
         label="measured (FFT mode 1)")
plt.xlabel(r"$\omega$"); plt.ylabel(r"kinematic viscosity $\nu$")
plt.yscale("log"); plt.grid(True, which="both", alpha=0.3); plt.legend()
plt.title("Viscosity from FFT mode-1 shear-wave decay")
plt.tight_layout()
plt.savefig("m4_viscosity_fft.png", dpi=110)
print("Saved m4_viscosity_fft.png")

# quick accuracy check
nu_th_at = (1.0/3.0)*(1.0/omegas - 0.5)
rel = np.abs(nu_fft - nu_th_at)/nu_th_at
print(f"max relative error vs theory: {rel.max()*100:.2f}%")
