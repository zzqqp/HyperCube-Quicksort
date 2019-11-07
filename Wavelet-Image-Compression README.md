# Wavelet-Image-Compression
This program used hybrid MPI+OpenMP(multithreading) programming to perform wavelet transform using the Daubechies 4 basis. 

Daubechies methods:

smoothed compoents:

I[2*i] = C0 * I[2*i] + C1 * I[2*i+1] + C2 * I[2*i+2] + C3 * I[2*i+3]

detailed components:

I[2*i+1] = C3 * I[2*i] - C2 * I[2*i+1] + C1 * I[2*i+2] - C0 * I[2*i+3]

where the coefficients are defined as

C0 = (1+√3)/4√2 = 0.4829629131445341

C1 = (3+√3)/4√2 = 0.8365163037378079

C2 = (3-√3)/4√2 = 0.2241438680420134

C3 = (1-√3)/4√2 = -0.1294095225512604


A major objective of this program is hands-on experience on the most common
parallelization scheme for continuum simulation. Central to this parallelization is caching boundary data from neighbor processors
(“ghost zone” or “halo”). So, it is required to exchange two rows
and two columns of ghost zones with neighbor processors at every level of wavelet smoothing. In the row
direction, for example, row 0 and 1 were sent to the lower processor in the row direction, while
receiving row nr and nr+1 from the higher neighbor (nr is the number of rows per processor,
which is halved at each level). Cyclic message passing among a group of MPI processes
potentially causes a deadlock, one way to
avoid a deadlock is to use asynchronous message passing. Within
each spatial subdomain, OpenMP threads concurrently execute large for loops.
The program reads a gray-scale image (each pixel value in the range [0,255]) file of size
2Lmax × 2Lmax pixels and outputs a smooth image of size 2Lmin × 2Lmin after Lmax-Lmin levels of
recursive wavelet decompositions in both the row and column directions. Here, we assume that 
the number of MPI processes is P = 4LP (or LP = log4P) and Lmin ≥ LP. Each MPI process
receives a sub-block of size 2Lmax−LP × 2Lmax−LP pixels and returns a sub-block of 2Lmin−LP × 2Lmin−LP
pixels. Each MPI process in turn spawns nthread threads that concurrently execute major for loops
using the #pragma omp parallel for directive. 
