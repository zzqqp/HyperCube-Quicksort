# HyperCube-Quicksort
Implementation in C++ using OpenMPI. Implement a broadcast operation within an L-dimensional subcube. 
An MPI program to perform hypercube quicksort

Initially, each process randomly generates n local elements

Implement a broadcast operation within an L-dimensional subcube. A hierarchy of subcubes can be
implemented by nested calls to MPI_Comm_create().

After N iterations the numbers of process 0 will be < the numbers of process 1 < the numbers of process 2 < ... and so on. Processes can now sort numbers sequentially with normal quicksort.

Results are gathered and written by the master process.
