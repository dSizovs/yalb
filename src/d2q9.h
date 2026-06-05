#ifndef D2Q9_H
#define D2Q9_H

#include <Kokkos_Core.hpp>

// D2Q9 lattice: 9 discrete velocity directions in 2D.
//   0: ( 0, 0) rest        1: ( 1, 0) E    2: ( 0, 1) N
//   3: (-1, 0) W           4: ( 0,-1) S    5: ( 1, 1) NE
//   6: (-1, 1) NW          7: (-1,-1) SW   8: ( 1,-1) SE

constexpr int Q = 9;  // number of discrete velocities
constexpr int D = 2;  // spatial dimensions

// Device-safe accessors. A constexpr array LOCAL to a KOKKOS_INLINE_FUNCTION
// is usable from both host and CUDA device code, unlike a file-scope
// constexpr array (which CUDA device kernels cannot see).
KOKKOS_INLINE_FUNCTION int cx(int i) {
    constexpr int v[Q] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
    return v[i];
}
KOKKOS_INLINE_FUNCTION int cy(int i) {
    constexpr int v[Q] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
    return v[i];
}
KOKKOS_INLINE_FUNCTION double w(int i) {
    constexpr double v[Q] = {
        4.0 / 9.0,
        1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,
        1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
    };
    return v[i];
}

#endif
