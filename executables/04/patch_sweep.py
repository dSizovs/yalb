src = open('main.cpp').read()

old = '''            for (double omega : omegas) {
                char fname[64];
                std::snprintf(fname, sizeof(fname),
                              "amplitude_omega_%.3f.txt", omega);
                std::ofstream amp(fname);
                run_shear(omega, Nx, Ny, Nsteps, epsilon, &amp,
                          /*frames_prefix=*/"", /*dump_every=*/0);
                std::cout << "omega=" << omega << " -> " << fname << "\\n";
            }'''

new = '''            for (double omega : omegas) {
                char fname[64];
                std::snprintf(fname, sizeof(fname),
                              "amplitude_omega_%.3f.txt", omega);
                std::ofstream amp(fname);
                // Also dump the ux(y) profile over time for FFT mode analysis.
                char pfname[64];
                std::snprintf(pfname, sizeof(pfname),
                              "profile_omega_%.3f.txt", omega);
                std::ofstream prof(pfname);
                run_shear(omega, Nx, Ny, Nsteps, epsilon, &amp,
                          /*frames_prefix=*/"", /*dump_every=*/0,
                          /*out_prof=*/&prof);
                std::cout << "omega=" << omega << " -> " << fname
                          << ", " << pfname << "\\n";
            }'''

assert old in src, "sweep loop not found"
src = src.replace(old, new)
open('main.cpp','w').write(src)
print("sweep wired for profiles")
