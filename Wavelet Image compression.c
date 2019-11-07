#include <stdio.h>
#include <string.h>
#include <omp.h>
#include "mpi.h"

#define NMAX 512
#define MAX 255
#define MAXLINE 1024
#define c0  0.4829629131445341
#define c1  0.8365163037378079
#define c2  0.2241438680420134
#define c3 -0.1294095225512604
#define N 512
#define L 3

int main(int argc, char *argv[]) {
  int nr,nc, s0, s1, sr, sc, n,work, nproc = 4;
  int nthread = 4;            //# of OpenMp threads
  int vproc[2] ={2,2}, vid[2], sid, nbr[2];  // 2*2 spatial decomposition, vector process ID, myID, neighbor ID
  int c,l,i,j,k,r,s,tag = 0;
  FILE *f;
  char line[MAXLINE] = {};
  double image[N+2][N+2] = {};
  double sbnf[N] = {};
  double rbnf[N] = {};

  MPI_Init(&argc,&argv); /* Initialize the MPI environment */
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  MPI_Comm_rank(MPI_COMM_WORLD,&sid);
  MPI_Status status;
  MPI_Request request;

  //Calculate Neighobor ID
  vid[0] = sid/vproc[1];
  vid[1] = sid % vproc[1];
  s0 = (vid[0]-1+vproc[0]) % vproc[0];
  s1 = vid[1];
  nbr[0] = s0*vproc[1]+s1;
  s0 = vid[0];
  s1 = (vid[1]-1+vproc[1] ) % vproc[1];
  nbr[1] = s0*vproc[1] + s1; 

  omp_set_num_threads(nthread); //OpenMp setup

// Read Image
  if(sid==0){

  f=fopen("Lenna512x512.pgm","r");
  fgets(line,MAXLINE,f);
  fgets(line,MAXLINE,f);
  fgets(line,MAXLINE,f);
  sscanf(line,"%d %d",&nc,&nr);
  fgets(line,MAXLINE,f);
  sscanf(line,"%d",&n);

  for (r=0; r<nr; r++) {
    for (c=0; c<nc; c++) {
      image[r][c] = (int)fgetc(f);
    }
  }

  fclose(f);
  }

  nr = N / vproc[0];
  nc = N / vproc[1];
 

//initialization, send sub-image to parallel node
  for(s=1;s<nproc;s++){
    tag = 0;
    sr = s/vproc[1];
    sc = s%vproc[1];
    if(sid ==0){
      for(i=0;i<nr;i++){
        for(j=0;j<nc;j++){
          tag++;
          MPI_Send( &(image[sr*nr+i][sc*nc+j]), 1, MPI_DOUBLE, s, tag, MPI_COMM_WORLD);
        }
      }
    }
    else if(sid==s){
      for(i=0;i<nr;i++){
        for(j=0;j<nc;j++){
          tag++;
          MPI_Recv( &(image[i][j]), 1, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD, &status);
        }
      }

    }
  }

  //Recursive Wavelet Decomposition 
  for(l=0;l<L;l++){
     //Append rows
    memset(sbnf, 0, N*sizeof(double));
    memset(rbnf, 0, N*sizeof(double));
    for(i=0;i<nc;i++){
      sbnf[i] = image[0][i];
      sbnf[i+nc] = image[1][i];
    }
    MPI_Irecv( &rbnf, 2*nc, MPI_DOUBLE, MPI_ANY_SOURCE, 20, MPI_COMM_WORLD, &request);
    MPI_Send( &sbnf, 2*nc, MPI_DOUBLE, nbr[0], 20, MPI_COMM_WORLD);
    MPI_Wait(&request,&status);
    for(i=0;i<nc;i++){
      image[nr][i] = rbnf[i];
      image[nr+1][i] = rbnf[i+nc];
    }
    
    #pragma omp parallel private(r)
    {
    int c;
    int tid = omp_get_thread_num();
    for(c = tid;c<nc;c+=nthread){
      for(r=0;r<nr/2;r++){
        image[r][c] = c0*image[2*r][c]+ c1*image[2*r+1][c] + c2*image[2*r+2][c]+ c3*image[2*r+3][c];
      }
    }
    }
    
    //append columns
    memset(sbnf, 0, N*sizeof(double));
    memset(rbnf, 0, N*sizeof(double));
    for(i=0;i<nr/2;i++){
      sbnf[i] = image[i][0];
      sbnf[i+nr/2] = image[i][1];
    }
    MPI_Irecv( &rbnf, nr, MPI_DOUBLE, MPI_ANY_SOURCE, 20, MPI_COMM_WORLD, &request);
    MPI_Send( &sbnf, nr, MPI_DOUBLE, nbr[1], 20, MPI_COMM_WORLD);
    MPI_Wait(&request,&status);
    for(i=0;i<nr/2;i++){
      image[i][nc] = rbnf[i];
      image[i][nc+1] = rbnf[i+nr/2];
    }
    #pragma omp parallel private(r)
    {
    int c;
    int tid = omp_get_thread_num();
    for(r = tid;r<nr/2;r+=nthread){
      for(c=0;c<nc/2;c++){
        image[r][c] = c0*image[r][2*c]+ c1*image[r][2*c+1] + c2*image[r][2*c+2]+ c3*image[r][2*c+3];
      }
    }
    }
    nr /= 2;
    nc /= 2;
    
  }

    for(s=1; s<nproc; s++){
      tag = 0;
      sr = s/ vproc[1];
      sc = s % vproc[1];
      if(sid ==0){
       for(i=0;i<nr;i++){
        for(j=0;j<nc;j++){
          tag++;
          MPI_Recv( &(image[sr*nr+i][sc*nc+j]), 1, MPI_DOUBLE, s, tag, MPI_COMM_WORLD, &status);
        }
       }
      }
      else if(sid==s){
      for(i=0;i<nr;i++){
        for(j=0;j<nc;j++){
          tag++;
          MPI_Send( &(image[i][j]), 1, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
        }
      }

    }
    }

   //rescale
    if(sid ==0){
      int ma = 0;
      nc *= 2;
      nr *= 2;
    

    for(r=0;r<nr;r++){
      for(c=0;c<nc;c++){
        if(image[r][c]>ma)
        ma = image[r][c];
      }
    }

  f=fopen("Lenna64x64.pgm","w");
  fprintf(f,"P5\n");
  fprintf(f,"# Simple image test\n");
  fprintf(f,"%d %d\n",nc,nr);
  fprintf(f,"%d\n",MAX);

  for (r=0; r<nr; r++) {
    for (c=0; c<nc; c++) {
      work= (int)(image[r][c]*255/ma);
      fputc((char)work,f);
    }
  }

  fclose(f);
  }
  

  MPI_Finalize();

}
