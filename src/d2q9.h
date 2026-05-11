#ifndef D2Q9_H
#define D2Q9_H

#include <Kokkos_Core.hpp>

// D2Q9 lattice: 9 discrete velocity directions in 2D.
//
// Direction layout (index -> (cx, cy)):
//   0: ( 0,  0)   rest
//   1: ( 1,  0)   east
//   2: ( 0,  1)   north
//   3: (-1,  0)   west
//   4: ( 0, -1)   south
//   5: ( 1,  1)   north-east
//   6: (-1,  1)   north-west
//   7: (-1, -1)   south-west
//   8: ( 1, -1)   south-east

constexpr int Q = 9;  // number of discrete velocities
constexpr int D = 2;  // spatial dimensions

// Velocity components.
constexpr int cx[Q] = {0,  1,  0, -1,  0,  1, -1, -1,  1};
constexpr int cy[Q] = {0,  0,  1,  0, -1,  1,  1, -1, -1};

#endif
