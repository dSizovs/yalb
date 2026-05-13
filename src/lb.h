#ifndef LB_H
#define LB_H

#include <Kokkos_Core.hpp>

// f(x, y, i), rho(x, y), v(x, y, d)
using FView   = Kokkos::View<double***>;
using RhoView = Kokkos::View<double**>;
using VView   = Kokkos::View<double***>;

// Compute density: rho(x, y) = sum_i f(x, y, i)
void compute_density(const FView& f, const RhoView& rho);

// Compute macroscopic velocity: v(x, y, d) = (1/rho) * sum_i f(x, y, i) * c_i_d
void compute_velocity(const FView& f, const RhoView& rho, const VView& v);

// Streaming: shift f(x, y, i) -> f(x + cx[i], y + cy[i], i), periodic.
void streaming(FView& f);

// Compute equilibrium distribution from local density and velocity.
void compute_equilibrium(const RhoView& rho, const VView& v, const FView& f_eq);

// BGK collision: f_i <- f_i + omega * (f_eq_i - f_i). Recomputes rho, u, f_eq.
void collision(FView& f, const RhoView& rho, const VView& v,
               const FView& f_eq, double omega);

// Max_{x,y} |v(x, y, 0)|.  Used for shear-wave amplitude.
double max_abs_ux(const VView& v);

// Lid-driven cavity bounce-back boundary conditions.
// f_pre holds f BEFORE streaming (we use those values for bounce-back).
// Walls: bottom (y=0), left (x=0), right (x=Nx-1) are stationary;
//        top (y=Ny-1) moves with velocity (u_lid, 0).
void bounce_back_cavity(const FView& f, const FView& f_pre,
                        const RhoView& rho, double u_lid);

// Max_{x,y,d} |v(x, y, d) - v_prev(x, y, d)|.  Used for convergence check.
double max_abs_diff(const VView& v, const VView& v_prev);

#endif
