#include "lb.h"
#include "d2q9.h"

void compute_density(const FView& f, const RhoView& rho) {
    const int Nx = f.extent(0);
    const int Ny = f.extent(1);

    Kokkos::parallel_for(
        "compute_density",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx, Ny}),
        KOKKOS_LAMBDA(int x, int y) {
            double s = 0.0;
            for (int i = 0; i < Q; ++i) {
                s += f(x, y, i);
            }
            rho(x, y) = s;
        });
}

void compute_velocity(const FView& f, const RhoView& rho, const VView& v) {
    const int Nx = f.extent(0);
    const int Ny = f.extent(1);

    Kokkos::parallel_for(
        "compute_velocity",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx, Ny}),
        KOKKOS_LAMBDA(int x, int y) {
            double ux = 0.0, uy = 0.0;
            for (int i = 0; i < Q; ++i) {
                ux += f(x, y, i) * cx[i];
                uy += f(x, y, i) * cy[i];
            }
            const double r = rho(x, y);
            if (r > 0.0) {
                v(x, y, 0) = ux / r;
                v(x, y, 1) = uy / r;
            } else {
                v(x, y, 0) = 0.0;
                v(x, y, 1) = 0.0;
            }
        });
}

void streaming(FView& f) {
    const int Nx = f.extent(0);
    const int Ny = f.extent(1);

    FView f_new("f_new", Nx, Ny, Q);

    Kokkos::parallel_for(
        "streaming",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {Nx, Ny, Q}),
        KOKKOS_LAMBDA(int x, int y, int i) {
            const int xs = (x - cx[i] + Nx) % Nx;
            const int ys = (y - cy[i] + Ny) % Ny;
            f_new(x, y, i) = f(xs, ys, i);
        });

    Kokkos::deep_copy(f, f_new);
}

void compute_equilibrium(const RhoView& rho, const VView& v, const FView& f_eq) {
    const int Nx = rho.extent(0);
    const int Ny = rho.extent(1);

    Kokkos::parallel_for(
        "compute_equilibrium",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx, Ny}),
        KOKKOS_LAMBDA(int x, int y) {
            const double r  = rho(x, y);
            const double ux = v(x, y, 0);
            const double uy = v(x, y, 1);
            const double u2 = ux * ux + uy * uy;
            for (int i = 0; i < Q; ++i) {
                const double cu = cx[i] * ux + cy[i] * uy;
                f_eq(x, y, i) = w[i] * r *
                    (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
            }
        });
}

void collision(FView& f, const RhoView& rho, const VView& v,
               const FView& f_eq, double omega) {
    compute_density(f, rho);
    compute_velocity(f, rho, v);
    compute_equilibrium(rho, v, f_eq);

    const int Nx = f.extent(0);
    const int Ny = f.extent(1);

    Kokkos::parallel_for(
        "collision",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {Nx, Ny, Q}),
        KOKKOS_LAMBDA(int x, int y, int i) {
            f(x, y, i) += omega * (f_eq(x, y, i) - f(x, y, i));
        });
}
