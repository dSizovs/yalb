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
            for (int i = 0; i < Q; ++i) s += f(x, y, i);
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
                ux += f(x, y, i) * cx(i);
                uy += f(x, y, i) * cy(i);
            }
            const double r = rho(x, y);
            if (r > 0.0) { v(x, y, 0) = ux / r; v(x, y, 1) = uy / r; }
            else         { v(x, y, 0) = 0.0;    v(x, y, 1) = 0.0;    }
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
            const int xs = (x - cx(i) + Nx) % Nx;
            const int ys = (y - cy(i) + Ny) % Ny;
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
            const double r = rho(x, y);
            const double ux = v(x, y, 0);
            const double uy = v(x, y, 1);
            const double u2 = ux * ux + uy * uy;
            for (int i = 0; i < Q; ++i) {
                const double cu = cx(i) * ux + cy(i) * uy;
                f_eq(x, y, i) = w(i) * r *
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

double max_abs_ux(const VView& v) {
    const int Nx = v.extent(0);
    const int Ny = v.extent(1);
    double m = 0.0;
    Kokkos::parallel_reduce(
        "max_abs_ux",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx, Ny}),
        KOKKOS_LAMBDA(int x, int y, double& lm) {
            const double a = v(x, y, 0);
            const double ab = (a < 0.0) ? -a : a;
            if (ab > lm) lm = ab;
        },
        Kokkos::Max<double>(m));
    return m;
}

// Opposite direction in D2Q9 (i_bar):
//   0->0, 1->3, 3->1, 2->4, 4->2, 5->7, 7->5, 6->8, 8->6
KOKKOS_INLINE_FUNCTION int opp(int i) {
    constexpr int table[Q] = {0, 3, 4, 1, 2, 7, 8, 5, 6};
    return table[i];
}

void bounce_back_cavity(const FView& f, const FView& f_pre,
                        const RhoView& rho, double u_lid) {
    const int Nx = f.extent(0);
    const int Ny = f.extent(1);
    constexpr double cs2 = 1.0 / 3.0;  // squared speed of sound (lattice units)

    // BOTTOM wall (y = 0): stationary. The populations that "streamed in"
    // from outside are directions with cy > 0 (pointing up): 2, 5, 6.
    // Bounce-back: f_i = f_pre_{opp(i)}
    Kokkos::parallel_for(
        "bb_bottom",
        Kokkos::RangePolicy<>(0, Nx),
        KOKKOS_LAMBDA(int x) {
            f(x, 0, 2) = f_pre(x, 0, 4);
            f(x, 0, 5) = f_pre(x, 0, 7);
            f(x, 0, 6) = f_pre(x, 0, 8);
        });

    // LEFT wall (x = 0): stationary. Incoming pops have cx > 0: 1, 5, 8.
    Kokkos::parallel_for(
        "bb_left",
        Kokkos::RangePolicy<>(0, Ny),
        KOKKOS_LAMBDA(int y) {
            f(0, y, 1) = f_pre(0, y, 3);
            f(0, y, 5) = f_pre(0, y, 7);
            f(0, y, 8) = f_pre(0, y, 6);
        });

    // RIGHT wall (x = Nx-1): stationary. Incoming pops have cx < 0: 3, 6, 7.
    Kokkos::parallel_for(
        "bb_right",
        Kokkos::RangePolicy<>(0, Ny),
        KOKKOS_LAMBDA(int y) {
            f(Nx - 1, y, 3) = f_pre(Nx - 1, y, 1);
            f(Nx - 1, y, 6) = f_pre(Nx - 1, y, 8);
            f(Nx - 1, y, 7) = f_pre(Nx - 1, y, 5);
        });

    // TOP wall (y = Ny-1): moving at (u_lid, 0).
    // Incoming pops have cy < 0: 4, 7, 8.
    // Moving wall correction: -2 * w_i * rho_w * (c_i . u_w) / c_s^2
    // Note: w_i here is the weight of the *incoming* direction.
    Kokkos::parallel_for(
        "bb_top",
        Kokkos::RangePolicy<>(0, Nx),
        KOKKOS_LAMBDA(int x) {
            const int yy = Ny - 1;
            const double rw = rho(x, yy);
            // i = 4 (going down): opposite is 2 (going up). c_4.u = 0*u_lid = 0
            f(x, yy, 4) = f_pre(x, yy, 2);
            // i = 7 (down-left): opposite is 5 (up-right). c_7.u = -1*u_lid
            f(x, yy, 7) = f_pre(x, yy, 5)
                - 2.0 * w(5) * rw * (cx(5) * u_lid) / cs2;
            // i = 8 (down-right): opposite is 6 (up-left). c_8.u = +1*u_lid
            f(x, yy, 8) = f_pre(x, yy, 6)
                - 2.0 * w(6) * rw * (cx(6) * u_lid) / cs2;
        });
}

double max_abs_diff(const VView& v, const VView& v_prev) {
    const int Nx = v.extent(0);
    const int Ny = v.extent(1);
    double m = 0.0;
    Kokkos::parallel_reduce(
        "max_abs_diff",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {Nx, Ny, D}),
        KOKKOS_LAMBDA(int x, int y, int d, double& lm) {
            const double a = v(x, y, d) - v_prev(x, y, d);
            const double ab = (a < 0.0) ? -a : a;
            if (ab > lm) lm = ab;
        },
        Kokkos::Max<double>(m));
    return m;
}
