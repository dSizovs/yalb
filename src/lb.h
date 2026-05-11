#ifndef LB_H
#define LB_H

#include <Kokkos_Core.hpp>

// Type aliases for the Kokkos Views we use throughout.
// f is the distribution function: f(x, y, i) for i in [0..9).
// rho is the density: rho(x, y).
// v is the macroscopic velocity: v(x, y, d) for d in [0..2).
using FView   = Kokkos::View<double***>;  // (Nx, Ny, 9)
using RhoView = Kokkos::View<double**>;   // (Nx, Ny)
using VView   = Kokkos::View<double***>;  // (Nx, Ny, 2)

// Compute density: rho(x, y) = sum_i f(x, y, i)
void compute_density(const FView& f, const RhoView& rho);

// Compute macroscopic velocity:
//   v(x, y, d) = (1/rho(x,y)) * sum_i f(x, y, i) * c_i_d
void compute_velocity(const FView& f, const RhoView& rho, const VView& v);

// Streaming step: shift f(x, y, i) to f(x + cx[i], y + cy[i], i).
// Uses a temporary view since each cell reads from neighbors.
// Periodic boundaries.
void streaming(FView& f);

#endif
