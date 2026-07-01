src = open('main.cpp').read()

old = '''        if (rank == 0) {
            const double elapsed = t1 - t0;
            std::cout << "nprocs=" << nprocs
                      << "  N=" << N
                      << "  steps=" << Nsteps
                      << "  omega=" << omega
                      << "  time=" << elapsed << " s"
                      << "  total_mass=" << global_mass << "\\n";
            // Append timing to a scaling log.
            std::ofstream tlog("scaling.txt", std::ios::app);
            tlog << nprocs << " " << N << " " << Nsteps << " "
                 << elapsed << "\\n";
        }'''

new = '''        if (rank == 0) {
            const double elapsed = t1 - t0;
            // MLUPS: million lattice updates per second (global grid N*N, Nsteps).
            const double mlups = (double)N * (double)N * (double)Nsteps
                                 / (elapsed * 1.0e6);
            std::cout << "nprocs=" << nprocs
                      << "  N=" << N
                      << "  steps=" << Nsteps
                      << "  omega=" << omega
                      << "  time=" << elapsed << " s"
                      << "  MLUPS=" << mlups
                      << "  total_mass=" << global_mass << "\\n";
            // Append timing to a scaling log: nprocs N Nsteps elapsed mlups
            std::ofstream tlog("scaling.txt", std::ios::app);
            tlog << nprocs << " " << N << " " << Nsteps << " "
                 << elapsed << " " << mlups << "\\n";
        }'''

assert old in src, "output block not found -- source differs"
src = src.replace(old, new)
open('main.cpp','w').write(src)
print("MLUPS added")
