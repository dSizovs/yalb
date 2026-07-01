src = open('main.cpp').read()

# 1) Add a profile-output parameter to run_shear's signature.
old_sig = '''static void run_shear(double omega, int Nx, int Ny, int Nsteps, double epsilon,
                      std::ofstream* out_amp,
                      const std::string& frames_prefix,
                      int dump_every) {'''
new_sig = '''static void run_shear(double omega, int Nx, int Ny, int Nsteps, double epsilon,
                      std::ofstream* out_amp,
                      const std::string& frames_prefix,
                      int dump_every,
                      std::ofstream* out_prof = nullptr) {'''
assert old_sig in src, "signature not found"
src = src.replace(old_sig, new_sig)

# 2) After writing max|ux|, also dump the ux(y) profile at x=0 as one line per step.
old_amp = '''        if (out_amp) {
            const double amp = max_abs_ux(v);
            (*out_amp) << step << " " << amp << "\\n";
        }'''
new_amp = '''        if (out_amp) {
            const double amp = max_abs_ux(v);
            (*out_amp) << step << " " << amp << "\\n";
        }

        // Optional: dump ux(y) along x=0 for FFT-based mode analysis.
        if (out_prof) {
            Kokkos::deep_copy(v_h, v);
            (*out_prof) << step;
            for (int y = 0; y < Ny; ++y)
                (*out_prof) << " " << v_h(0, y, 0);
            (*out_prof) << "\\n";
        }'''
assert old_amp in src, "amp block not found"
src = src.replace(old_amp, new_amp)

open('main.cpp','w').write(src)
print("profile dump added")
