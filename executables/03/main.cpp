#include "d2q9.h"
#include "lb.h"

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

void write_frame(const std::string& filename,
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

int main(int argc, char* argv[]) {
    // Pick test scenario.
    std::string scenario = (argc > 1) ? argv[1] : "bump";
    if (scenario != "bump" && scenario != "shear") {
        std::cerr << "Usage: " << argv[0] << " [bump|shear]\n";
        return 1;
    }

    Kokkos::initialize(argc, argv);
    {
        const int Nx = 50;
        const int Ny = 50;
        const int Nsteps = 200;
        const int dump_every = 5;          // write every Nth frame
        const double omega = 1.0;          // relaxation rate (0 < omega < 2)

        FView   f("f", Nx, Ny, Q);
        FView   f_eq("f_eq", Nx, Ny, Q);
        RhoView rho("rho", Nx, Ny);
        VView   v("v", Nx, Ny, D);

        // Initialize rho and u on the host, then push to device.
        auto rho_h = Kokkos::create_mirror_view(rho);
        auto v_h   = Kokkos::create_mirror_view(v);

        if (scenario == "bump") {
            // Uniform density 0.5, with a +0.2 bump at the center, zero velocity.
            for (int x = 0; x < Nx; ++x) {
                for (int y = 0; y < Ny; ++y) {
                    rho_h(x, y) = 0.5;
                    v_h(x, y, 0) = 0.0;
                    v_h(x, y, 1) = 0.0;
                }
            }
            const int cx0 = Nx / 2, cy0 = Ny / 2;
            for (int dx = -2; dx <= 2; ++dx) {
                for (int dy = -2; dy <= 2; ++dy) {
                    rho_h(cx0 + dx, cy0 + dy) = 0.7;
                }
            }
        } else {
            // shear: uniform density 0.5; horizontal velocity that varies with y,
            // sinusoidally. A classic shear-wave-like initial condition.
            for (int x = 0; x < Nx; ++x) {
                for (int y = 0; y < Ny; ++y) {
                    rho_h(x, y) = 0.5;
                    const double u_amp = 0.05;
                    v_h(x, y, 0) = u_amp * std::sin(2.0 * M_PI * y / Ny);
                    v_h(x, y, 1) = 0.0;
                }
            }
        }

        Kokkos::deep_copy(rho, rho_h);
        Kokkos::deep_copy(v,   v_h);

        // Initialize f from the equilibrium of the initial (rho, u).
        compute_equilibrium(rho, v, f);

        // Main time loop: stream, then collide.
        int frame_idx = 0;
        for (int step = 0; step < Nsteps; ++step) {
            if (step % dump_every == 0) {
                // Up-to-date macro quantities for plotting.
                compute_density(f, rho);
                compute_velocity(f, rho, v);
                Kokkos::deep_copy(rho_h, rho);
                Kokkos::deep_copy(v_h,   v);
                char fname[64];
                std::snprintf(fname, sizeof(fname), "frame_%03d.txt", frame_idx++);
                write_frame(fname, rho_h, v_h);
            }

            streaming(f);
            collision(f, rho, v, f_eq, omega);
        }

        std::cout << "Scenario: " << scenario << "\n"
                  << "Wrote " << frame_idx << " frames\n";
    }
    Kokkos::finalize();
    return 0;
}
