#include "d2q9.h"
#include "lb.h"

#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// Set up the shear-wave initial condition:
//   rho = 1
//   ux(y) = epsilon * sin(2 * pi * y / Ny)
//   uy = 0
static void init_shear_wave(const RhoView::HostMirror& rho_h,
                            const VView::HostMirror& v_h,
                            double epsilon) {
    const int Nx = rho_h.extent(0);
    const int Ny = rho_h.extent(1);
    for (int x = 0; x < Nx; ++x) {
        for (int y = 0; y < Ny; ++y) {
            rho_h(x, y) = 1.0;
            v_h(x, y, 0) = epsilon * std::sin(2.0 * M_PI * y / Ny);
            v_h(x, y, 1) = 0.0;
        }
    }
}

static void write_frame(const std::string& filename,
                        const RhoView::HostMirror& rho_h,
                        const VView::HostMirror& v_h) {
    const int Nx = rho_h.extent(0);
    const int Ny = rho_h.extent(1);
    std::ofstream out(filename);
    out << Nx << " " << Ny << "\n";
    for (int y = 0; y < Ny; ++y) {
        for (int x = 0; x < Nx; ++x) {
            out << rho_h(x, y) << " "
                << v_h(x, y, 0) << " "
                << v_h(x, y, 1) << "\n";
        }
    }
}

// Run the shear-wave simulation for given omega.
// If `out_amp` is non-null, append "step  max|ux|" lines for amplitude fitting.
// If `frames_prefix` is non-empty, dump full frames at regular intervals.
static void run_shear(double omega, int Nx, int Ny, int Nsteps, double epsilon,
                      std::ofstream* out_amp,
                      const std::string& frames_prefix,
                      int dump_every) {
    FView   f("f", Nx, Ny, Q);
    FView   f_eq("f_eq", Nx, Ny, Q);
    RhoView rho("rho", Nx, Ny);
    VView   v("v", Nx, Ny, D);

    auto rho_h = Kokkos::create_mirror_view(rho);
    auto v_h   = Kokkos::create_mirror_view(v);

    init_shear_wave(rho_h, v_h, epsilon);
    Kokkos::deep_copy(rho, rho_h);
    Kokkos::deep_copy(v,   v_h);

    // Initialize f from equilibrium of (rho, u).
    compute_equilibrium(rho, v, f);

    int frame_idx = 0;
    for (int step = 0; step < Nsteps; ++step) {
        // Refresh macro quantities before recording / dumping.
        compute_density(f, rho);
        compute_velocity(f, rho, v);

        if (out_amp) {
            const double amp = max_abs_ux(v);
            (*out_amp) << step << " " << amp << "\n";
        }

        if (!frames_prefix.empty() && step % dump_every == 0) {
            Kokkos::deep_copy(rho_h, rho);
            Kokkos::deep_copy(v_h,   v);
            char fname[128];
            std::snprintf(fname, sizeof(fname), "%s_%03d.txt",
                          frames_prefix.c_str(), frame_idx++);
            write_frame(fname, rho_h, v_h);
        }

        streaming(f);
        collision(f, rho, v, f_eq, omega);
    }
}

int main(int argc, char* argv[]) {
    const std::string mode = (argc > 1) ? argv[1] : "sweep";
    if (mode != "sweep" && mode != "frames") {
        std::cerr << "Usage: " << argv[0] << " [sweep|frames]\n";
        return 1;
    }

    Kokkos::initialize(argc, argv);
    {
        const int Nx = 50;
        const int Ny = 50;
        const int Nsteps = 2000;
        const double epsilon = 0.01;  // small to stay in Stokes regime

        if (mode == "frames") {
            // Frames for visualization at omega = 1.
            const double omega = 1.0;
            std::ofstream amp("amplitude_omega_1.000.txt");
            run_shear(omega, Nx, Ny, Nsteps, epsilon, &amp, "frame", 100);
            std::cout << "Wrote frames for omega=1.0\n";
        } else {
            // Sweep over omega values.
            std::vector<double> omegas = {
                0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 1.9
            };
            for (double omega : omegas) {
                char fname[64];
                std::snprintf(fname, sizeof(fname),
                              "amplitude_omega_%.3f.txt", omega);
                std::ofstream amp(fname);
                run_shear(omega, Nx, Ny, Nsteps, epsilon, &amp,
                          /*frames_prefix=*/"", /*dump_every=*/0);
                std::cout << "omega=" << omega << " -> " << fname << "\n";
            }
        }
    }
    Kokkos::finalize();
    return 0;
}
