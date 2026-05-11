#include "d2q9.h"
#include "lb.h"

#include <Kokkos_Core.hpp>
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
    Kokkos::initialize(argc, argv);
    {
        const int Nx = 15;
        const int Ny = 10;
        const int Nsteps = 20;

        FView   f("f", Nx, Ny, Q);
        RhoView rho("rho", Nx, Ny);
        VView   v("v", Nx, Ny, D);

        // Initialize: a "blob" of particles all moving in direction 1 (east),
        // located in the middle of the grid.
        auto f_h = Kokkos::create_mirror_view(f);
        for (int x = 0; x < Nx; ++x) {
            for (int y = 0; y < Ny; ++y) {
                for (int i = 0; i < Q; ++i) {
                    f_h(x, y, i) = 0.0;
                }
            }
        }
        // Drop a blob at (5, 5) moving east (direction 1).
        f_h(5, 5, 1) = 1.0;
        Kokkos::deep_copy(f, f_h);

        // Mirrors for output.
        auto rho_h = Kokkos::create_mirror_view(rho);
        auto v_h   = Kokkos::create_mirror_view(v);

        for (int step = 0; step < Nsteps; ++step) {
            compute_density(f, rho);
            compute_velocity(f, rho, v);

            Kokkos::deep_copy(rho_h, rho);
            Kokkos::deep_copy(v_h, v);

            char fname[64];
            std::snprintf(fname, sizeof(fname), "frame_%03d.txt", step);
            write_frame(fname, rho_h, v_h);

            streaming(f);
        }

        std::cout << "Wrote " << Nsteps << " frames (frame_000.txt .. frame_"
                  << (Nsteps - 1) << ".txt)\n";
    }
    Kokkos::finalize();
    return 0;
}
