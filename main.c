
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

#define N 1024 /* Maximum list size */
#define MAX 99 /* Maximum value of a list element */

int nprocs,myid,bitvalue,mask; /* Cube size, dimension, & my rank */


int partition(int list[], int lo, int hi, int pivot) {
	int i = lo - 1, j;
	for (j = lo; j < hi; j++) {
		if (list[j] <= pivot) {
			i++;
			int temp = list[i];
			list[i] = list[j];
			list[j] = temp;
		}
	}
	int temp = list[i + 1];
	list[i + 1] = list[hi];
	list[hi] = temp;
	return (list[i + 1] > pivot) ? i : (i + 1);
}


void quicksort(int list[],int left,int right) {
int pivot,i,j;
int temp;
if (left < right) {
i = left; j = right + 1;
pivot = list[left];
do {
while (list[++i] < pivot && i <= right);
while (list[--j] > pivot);
if (i < j) {
temp = list[i]; list[i] = list[j]; list[j] = temp;
}
} while (i < j);
temp = list[left]; list[left] = list[j]; list[j] = temp;
quicksort(list,left,j-1);
quicksort(list,j+1,right);
}
}







int main(int argc, char *argv[])
{
  int  n=4, i, L, pivot , jj,parnter, nprocs_cube, sum, c, p;

  int send, recv, list[N]={}, tmp[N]={}, others[N]={}, temp[N]={}, procs_cube[N] ={};
 
  MPI_Init(&argc,&argv); /* Initialize the MPI environment */
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);

  bitvalue = nprocs>>1;
  mask =nprocs - 1;
  L = log(nprocs+1e-10)/log(2.0);
  nprocs_cube = nprocs;

  srand((unsigned) myid+1);
  for (i=0; i<n; i++) list[i] = rand()%MAX;



  printf("Before: Rank %2d :",myid);
  for (i=0; i<n; i++) printf("%3d ",list[i]);
  printf("\n");

  MPI_Status status;

  MPI_Comm cube[L][nprocs];
  MPI_Group cube_group[L][nprocs];
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs_cube);
  cube[L][0] = MPI_COMM_WORLD;
  c = myid/nprocs_cube;

  for( i= L;i>0; i--){
	  
  c = myid/nprocs_cube;

	  if ((myid & mask) == 0){
		  sum = 0;
		  for(jj = 0; jj<n ;jj++)
			  sum += list[jj];
		  pivot = sum/n;
	  }

	  MPI_Bcast(&pivot,1,MPI_INT,0,cube[i][c]);
         

	   int pos = partition(list,0, n-1, pivot);
           parnter = myid ^bitvalue;

	   if((myid & bitvalue) ==0){

		   send= 0;
		   recv = 0;

		   for (jj= pos+1; jj<n; jj++) {
                    tmp[send]= list[jj];
                    send++;                 
                }

		    MPI_Send(&send, 1, MPI_INT, parnter, 0, MPI_COMM_WORLD);
    if(send) MPI_Send(&tmp, send, MPI_INT, parnter, 1, MPI_COMM_WORLD);
            MPI_Recv(&recv, 1, MPI_INT, parnter, 0, MPI_COMM_WORLD, &status);

if(recv) MPI_Recv(&others, recv, MPI_INT, parnter, 1, MPI_COMM_WORLD, &status);            


            for (jj= 0 ; jj < recv; jj++) {
                list[ pos+1 + jj ]= others[jj];
            }

			n = n- send + recv; 

	   }
	   else{

		   send= 0;
		   recv = 0;
		   for(jj = 0; jj <= pos ; jj++){
			   tmp[send] = list[jj];
               send++;
		   }


           MPI_Recv(&recv, 1, MPI_INT, parnter, 0, MPI_COMM_WORLD, &status);

        
            if(recv) MPI_Recv(&others, recv, MPI_INT, parnter, 1, MPI_COMM_WORLD, &status);

       
            MPI_Send(&send, 1, MPI_INT, parnter, 0, MPI_COMM_WORLD);
            if(send) MPI_Send(&tmp, send, MPI_INT, parnter, 1, MPI_COMM_WORLD);

			for(jj = 0; jj<recv ;jj++){
				temp[jj] = others[jj];
			}
			for(jj = pos+1; jj<n ;jj++)
				temp[recv + jj-pos -1] = list[jj];

			n = n- send + recv; 

			for(jj = 0; jj<n ;jj++)
				list[jj] = temp[jj];
	   }

	   MPI_Comm_group(cube[i][c],&(cube_group[i][c]));
        nprocs_cube = nprocs_cube/2;
        for(p=0; p<nprocs_cube; p++) procs_cube[p] = p;
        MPI_Group_incl(cube_group[i][c],nprocs_cube,procs_cube,&(cube_group[i-1][2*c ]));
        MPI_Group_excl(cube_group[i][c],nprocs_cube,procs_cube,&(cube_group[i-1][2*c+1]));
       MPI_Comm_create(cube[i][c],cube_group[i-1][2*c ],&(cube[i-1][2*c ]));
        MPI_Comm_create(cube[i][c],cube_group[i-1][2*c+1],&(cube[i-1][2*c+1]));
        MPI_Group_free(&(cube_group[i ][c ]));
       MPI_Group_free(&(cube_group[i-1][2*c ]));
       MPI_Group_free(&(cube_group[i-1][2*c+1]));

	   mask = mask ^ bitvalue;
	   bitvalue = bitvalue >> 1;
	   
  }


 


  quicksort(list,0,n-1);     

  printf("After:  Rank %2d :",myid);
  for (i=0; i<n; i++) printf("%3d ",list[i]);
  printf("\n");



  MPI_Finalize();

  return 0;
}
