# Local setup notes

## macOS (Apple Silicon)

Install dependencies via Homebrew:
brew install cmake open-mpi googletest
xcode-select --install

## Known issue: Release-mode test crash

On macOS Apple Silicon with Open MPI 5.x and AppleClang 17, the `tests`
binary crashes with SIGTRAP under Release builds. Debug builds work
correctly. Both Debug and Release executables (`main`) run fine.

This appears to be a toolchain interaction (Open MPI 5 / Kokkos init
under -O3), not a code issue. The Debug build is sufficient for
development work.
