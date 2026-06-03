// Milestone 06: MPI domain decomposition of the Lattice Boltzmann shear-wave.
//
// 1D decomposition in x: each rank owns a vertical strip of width nx_local,
// full height Ny. The strip is stored with one ghost column on each side
// (padded width = nx_local + 2). Ranks form a periodic ring in x.
// y stays periodic and fully local.
//
// Validation: shear wave u_x(y) = eps * sin(2*pi*y/Ny). Global max|u_x|
// decays exponentially; fitted viscosity must match the serial result
// (nu = (1/3)(1/omega - 1/2)) regardless of the number of ranks.

#include "d2q9.h"

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using View3 = Kokkos::View<double***>;   // (nx_pad, Ny, Q)
using View3H = View3::HostMirror;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Parameters (overridable: N steps omega)
    int    N       = 200;
    int    Nsteps  = 2000;
    double omega   = 1.0;
    double epsilon = 0.01;
    if (argc > 1) N      = std::atoi(argv[1]);
    if (argc > 2) Nsteps = std::atoi(argv[2]);
    if (argc > 3) omega  = std::atof(argv[3]);

    const int Nx = N, Ny = N;

    // --- 1D decomposition in x ---
    // Distribute Nx columns across nprocs ranks as evenly as possible.
    const int base = Nx / nprocs;
    const int rem  = Nx % nprocs;
    const int nx_local = base + (rank < rem ? 1 : 0);
    // Global x index of this rank's first real column.
    int x_start = rank * base + (rank < rem ? rank : rem);

    const int left  = (rank - 1 + nprocs) % nprocs;
    const int right = (rank + 1) % nprocs;

    const int nx_pad = nx_local + 2;  // +2 ghost columns

    Kokkos::initialize(argc, argv);
    {
        // Distribution functions (padded in x).
        View3 f("f", nx_pad, Ny, Q);
        View3 f_new("f_new", nx_pad, Ny, Q);

        auto f_h = Kokkos::create_mirror_view(f);

        // --- Initialize: equilibrium of rho=1, u_x = eps sin(2 pi y / Ny) ---
        // Real columns are indices 1..nx_local; global x = x_start + (i-1).
        for (int i = 1; i <= nx_local; ++i) {
            for (int y = 0; y < Ny; ++y) {
                const double ux = epsilon * std::sin(2.0 * M_PI * y / Ny);
                const double uy = 0.0;
                const double rho = 1.0;
                const double u2 = ux * ux + uy * uy;
                for (int q = 0; q < Q; ++q) {
                    const double cu = cx[q] * ux + cy[q] * uy;
                    f_h(i, y, q) = w[q] * rho *
                        (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
                }
            }
        }
        Kokkos::deep_copy(f, f_h);

        // Host buffers for halo exchange: one column = Ny * Q doubles.
        const int col_size = Ny * Q;
        std::vector<double> send_left(col_size),  send_right(col_size);
        std::vector<double> recv_left(col_size),  recv_right(col_size);

        std::ofstream amp_out;
        if (rank == 0) {
            char fname[64];
            std::snprintf(fname, sizeof(fname),
                          "amplitude_p%d_omega_%.3f.txt", nprocs, omega);
            amp_out.open(fname);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        const double t0 = MPI_Wtime();

        for (int step = 0; step < Nsteps; ++step) {
            // ---- 1. Halo exchange ----
            // Copy my real edge columns to host send buffers.
            // Left real column  = index 1; right real column = index nx_local.
            Kokkos::deep_copy(f_h, f);
            for (int y = 0; y < Ny; ++y)
                for (int q = 0; q < Q; ++q) {
                    send_left[y * Q + q]  = f_h(1, y, q);
                    send_right[y * Q + q] = f_h(nx_local, y, q);
                }

            // Send my left column to `left` neighbor; receive into my right ghost
            // (from `right` neighbor's left column). And vice versa.
            MPI_Sendrecv(send_left.data(),  col_size, MPI_DOUBLE, left,  0,
                         recv_right.data(), col_size, MPI_DOUBLE, right, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Sendrecv(send_right.data(), col_size, MPI_DOUBLE, right, 1,
                         recv_left.data(),  col_size, MPI_DOUBLE, left,  1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Write received data into ghost columns: 0 (left), nx_local+1 (right).
            for (int y = 0; y < Ny; ++y)
                for (int q = 0; q < Q; ++q) {
                    f_h(0, y, q)            = recv_left[y * Q + q];
                    f_h(nx_local + 1, y, q) = recv_right[y * Q + q];
                }
            Kokkos::deep_copy(f, f_h);

            // ---- 2. Streaming (into f_new), real columns only ----
            // x is periodic via ghosts; y is periodic via modulo (local).
            Kokkos::parallel_for(
                "stream",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1, 0, 0},
                                                       {nx_local + 1, Ny, Q}),
                KOKKOS_LAMBDA(int i, int y, int q) {
                    const int isrc = i - cx[q];               // ghost covers +-1
                    const int ysrc = (y - cy[q] + Ny) % Ny;
                    f_new(i, y, q) = f(isrc, ysrc, q);
                });

            // ---- 3. Collision on real columns ----
            Kokkos::parallel_for(
                "collide",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 0},
                                                       {nx_local + 1, Ny}),
                KOKKOS_LAMBDA(int i, int y) {
                    double rho = 0.0, mx = 0.0, my = 0.0;
                    for (int q = 0; q < Q; ++q) {
                        const double fv = f_new(i, y, q);
                        rho += fv;
                        mx  += fv * cx[q];
                        my  += fv * cy[q];
                    }
                    const double ux = (rho > 0.0) ? mx / rho : 0.0;
                    const double uy = (rho > 0.0) ? my / rho : 0.0;
                    const double u2 = ux * ux + uy * uy;
                    for (int q = 0; q < Q; ++q) {
                        const double cu = cx[q] * ux + cy[q] * uy;
                        const double feq = w[q] * rho *
                            (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
                        f_new(i, y, q) += omega * (feq - f_new(i, y, q));
                    }
                });

            // Swap f <- f_new.
            Kokkos::deep_copy(f, f_new);

            // ---- 4. Global diagnostic: max|u_x| over the whole domain ----
            double local_max = 0.0;
            Kokkos::parallel_reduce(
                "max_ux",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 0},
                                                       {nx_local + 1, Ny}),
                KOKKOS_LAMBDA(int i, int y, double& lm) {
                    double rho = 0.0, mx = 0.0;
                    for (int q = 0; q < Q; ++q) {
                        rho += f(i, y, q);
                        mx  += f(i, y, q) * cx[q];
                    }
                    const double ux = (rho > 0.0) ? mx / rho : 0.0;
                    const double a = (ux < 0.0) ? -ux : ux;
                    if (a > lm) lm = a;
                },
                Kokkos::Max<double>(local_max));

            double global_max = 0.0;
            MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX,
                          MPI_COMM_WORLD);

            if (rank == 0) amp_out << step << " " << global_max << "\n";
        }

        MPI_Barrier(MPI_COMM_WORLD);
        const double t1 = MPI_Wtime();

        // ---- Global mass check (collective communication) ----
        double local_mass = 0.0;
        Kokkos::parallel_reduce(
            "mass",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 0}, {nx_local + 1, Ny}),
            KOKKOS_LAMBDA(int i, int y, double& lm) {
                for (int q = 0; q < Q; ++q) lm += f(i, y, q);
            },
            local_mass);
        double global_mass = 0.0;
        MPI_Allreduce(&local_mass, &global_mass, 1, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);

        if (rank == 0) {
            const double elapsed = t1 - t0;
            std::cout << "nprocs=" << nprocs
                      << "  N=" << N
                      << "  steps=" << Nsteps
                      << "  omega=" << omega
                      << "  time=" << elapsed << " s"
                      << "  total_mass=" << global_mass << "\n";
            // Append timing to a scaling log.
            std::ofstream tlog("scaling.txt", std::ios::app);
            tlog << nprocs << " " << N << " " << Nsteps << " "
                 << elapsed << "\n";
        }
    }
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}
