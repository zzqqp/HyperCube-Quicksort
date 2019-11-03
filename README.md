# HyperCube-Quicksort
Implementation in C using OpenMPI to perform hypercube quicksort.

Pseudocode:

bitvalue := 2dimension-1;
mask := 2dimension - 1;
for L := dimension downto 1
begin
 if myid AND mask = 0 then
 choose a pivot value for the L-dimensional subcube;
 broadcast the pivot from the master to the other members of the subcube;
 partition list[0:nelement-1] into two sublists such that
 list[0:j] ≤ pivot < list[j+1:nelement-1];
 partner := myid XOR bitvalue;
 if myid AND bitvalue = 0 then
 begin
 send the right sublist list[j+1:nelement-1] to partner;
 receive the left sublist of partner;
 append the received list to my left list
 end
 else
 begin
 send the left sublist list[0:j] to partner;
 receive the right sublist of partner;
 append the received list to my right list
 end
 nelement := nelement - nsend + nreceive;
 mask = mask XOR bitvalue;
 bitvalue = bitvalue/2
end
sequential quicksort to list[0:nelement-1]



Initially, each process randomly generates n local elements

Implement a broadcast operation within an L-dimensional subcube. A hierarchy of subcubes can be
implemented by nested calls to MPI_Comm_create().

After N iterations the numbers of process 0 will be < the numbers of process 1 < the numbers of process 2 < ... and so on. Processes can now sort numbers sequentially with normal quicksort.

Results are gathered and written by the master process.
