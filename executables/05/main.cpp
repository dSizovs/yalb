#include "d2q9.h"
#include "lb.h"

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

static void write_field(const std::string& filename,
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
    // Defaults
    int    N      = 200;     // grid is N x N
    double Re     = 100.0;
    double u_lid  = 0.1;
    double tol    = 1e-6;
    int    Nmax   = 200000;
    int    report = 500;

    // Override via command line: positional args N Re u_lid tol
    if (argc > 1) N      = std::atoi(argv[1]);
    if (argc > 2) Re     = std::atof(argv[2]);
    if (argc > 3) u_lid  = std::atof(argv[3]);
    if (argc > 4) tol    = std::atof(argv[4]);

    // Derive omega from Reynolds: Re = u_lid * L / nu, nu = (1/3)(1/omega - 1/2)
    const double L  = static_cast<double>(N);
    const double nu = u_lid * L / Re;
    const double omega = 1.0 / (3.0 * nu + 0.5);

    std::cout << "Lid-driven cavity\n"
              << "  N      = " << N      << "\n"
              << "  Re     = " << Re     << "\n"
              << "  u_lid  = " << u_lid  << "\n"
              << "  nu     = " << nu     << "\n"
              << "  omega  = " << omega  << "\n"
              << "  tol    = " << tol    << "\n";

    if (omega <= 0.0 || omega >= 2.0) {
        std::cerr << "ERROR: omega = " << omega
                  << " out of (0, 2). Choose different Re / u_lid.\n";
        return 1;
    }

    Kokkos::initialize(argc, argv);
    {
        const int Nx = N, Ny = N;

        FView   f("f",      Nx, Ny, Q);
        FView   f_pre("f_pre", Nx, Ny, Q);
        FView   f_eq("f_eq",   Nx, Ny, Q);
        RhoView rho("rho",     Nx, Ny);
        VView   v("v",         Nx, Ny, D);
        VView   v_prev("v_prev", Nx, Ny, D);

        // Initial condition: rho = 1, u = 0 everywhere. Initialize f to equilibrium.
        auto rho_h = Kokkos::create_mirror_view(rho);
        auto v_h   = Kokkos::create_mirror_view(v);
        for (int x = 0; x < Nx; ++x)
            for (int y = 0; y < Ny; ++y) {
                rho_h(x, y) = 1.0;
                v_h(x, y, 0) = 0.0;
                v_h(x, y, 1) = 0.0;
            }
        Kokkos::deep_copy(rho, rho_h);
        Kokkos::deep_copy(v,   v_h);
        compute_equilibrium(rho, v, f);

        // Main loop.
        int step = 0;
        double change = 1.0;
        for (; step < Nmax && change > tol; ++step) {
            // Save current velocity for convergence check (every `report` steps).
            if (step % report == 0) Kokkos::deep_copy(v_prev, v);

            // Save f before streaming for bounce-back.
            Kokkos::deep_copy(f_pre, f);
            streaming(f);
            // We need rho at the wall for the moving-wall correction.
            // Use rho from the previous step (already up-to-date) -- but compute
            // it from f_pre to be safe.
            compute_density(f_pre, rho);
            bounce_back_cavity(f, f_pre, rho, u_lid);
            collision(f, rho, v, f_eq, omega);

            if ((step + 1) % report == 0) {
                compute_velocity(f, rho, v);
                change = max_abs_diff(v, v_prev);
                std::cout << "step " << (step + 1)
                          << "  max|du| = " << change << "\n";
            }
        }

        // Final field for output.
        compute_density(f, rho);
        compute_velocity(f, rho, v);
        Kokkos::deep_copy(rho_h, rho);
        Kokkos::deep_copy(v_h,   v);

        char fname[128];
        std::snprintf(fname, sizeof(fname), "cavity_N%d_Re%g.txt", N, Re);
        write_field(fname, rho_h, v_h);

        std::cout << "Converged after " << step << " steps. Wrote " << fname << "\n";
    }
    Kokkos::finalize();
    return 0;
}
