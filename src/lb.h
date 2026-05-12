#ifndef LB_H
#define LB_H

#include <Kokkos_Core.hpp>

// f(x, y, i), rho(x, y), v(x, y, d)
using FView   = Kokkos::View<double***>;
using RhoView = Kokkos::View<double**>;
using VView   = Kokkos::View<double***>;

// Compute density: rho(x, y) = sum_i f(x, y, i)
void compute_density(const FView& f, const RhoView& rho);

// Compute macroscopic velocity:
//   v(x, y, d) = (1/rho) * sum_i f(x, y, i) * c_i_d
void compute_velocity(const FView& f, const RhoView& rho, const VView& v);

// Streaming: shift f(x, y, i) -> f(x + cx[i], y + cy[i], i), periodic.
void streaming(FView& f);

// Compute equilibrium distribution from local density and velocity:
//   f_eq_i = w_i * rho * [ 1 + 3(c.u) + 9/2 (c.u)^2 - 3/2 |u|^2 ]
void compute_equilibrium(const RhoView& rho, const VView& v, const FView& f_eq);

// BGK collision step (in place):
//   f_i <- f_i + omega * (f_eq_i - f_i)
// Computes rho, u, and f_eq internally.
void collision(FView& f, const RhoView& rho, const VView& v,
               const FView& f_eq, double omega);

// Return max_{x,y} |v(x, y, 0)|.  Used to track shear-wave amplitude.
double max_abs_ux(const VView& v);

#endif
