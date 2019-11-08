This program is the simplest GPU implementation to exercise the basic pattern of host-to-device copy + GPU-SIMD computing + device-to-host copy.

The program parallelize the one-dimensional quantum dynamics by combining MPI, OpenMP and CUDA.

Self-Centric (SC) Parallelization:

each MPI rank only works on a spatial subsystem;

boundary information obtained from neighbor ranks;

long-range information (if needed) by divide-&-
conquer, like real-space multigrids; 
scalability
behavior similar to short-ranged.

After spatial decomposition, each process p which belongs to [0, P–1] (P is the number of processes or MPI ranks) is
assigned a subsystem in the range [pLx, (p+1)Lx] of the total system length PLx. Each subsystem
of length Lx is discretized into Nx mesh points of interval Dx = Lx/Nx.
Each MPI process (with rank myid) spawns two OpenMP threads (each with a distinct
OpenMP thread ID mpid) that handle the lower and higher halves of the mesh points, each using
one GPU device specified by cudaSetDevice(mpid%2). Only the most time-consuming functions for propagating the wave function in time (i.e.,
pot_prop and kin_prop) were converted into GPU kernel functions.
Space Splitting Method (SSM) is utilized to calculate the potential and kientic energy propagator.
