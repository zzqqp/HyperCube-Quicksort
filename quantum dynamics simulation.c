/*******************************************************************************
1D Quantum dynamics (QD) simulation of an electron in one dimension.


*******************************************************************************/
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include <omp.h>
#include <cuda.h>
/******************************************************************************/
#define NX 4992   /* Number of mesh points */
#define NUM_DEVICE 2 
#define NUM_BLOCK 13 
#define NUM_THREAD 192 


int nx = NX/NUM_DEVICE;

/* Function prototypes ********************************************************/
void init_param();
void init_prop();
void init_wavefn();
void single_step();
void pot_prop();
void kin_prop(int);
void periodic_bc();
void calc_energy(int);
void hostdevice(double *d1, double h2[NX+2][2], int offset){
	double *h1 = (double*)malloc(sizeof(double)*2*(nx+2));
	int i=0,j=0;
	for(i=0;i<nx+2;i++){
		for(j=0;j<2;j++){
			h1[2*i+j] = h2[offset+i][j];
		}
	}
	cudaMemcpy(d1,h1,sizeof(double)*2*(nx+2), cudaMemcpyHostToDevice);
}
void devicehost(double h2[NX+2][2], double *d1, int offset){
	double h1[2*(nx+2)] ;
	cudaMemcpy(h1,d1,sizeof(double)*2*(nx+2), cudaMemcpyDeviceToHost);
	int i=0, j = 0;
	for(i=1;i<=nx;i++){
		for(j=0;j<2;j++){
		h2[offset+i][j] = h1[2*i+j];
		}
	}
}

/* Input parameters ***********************************************************/
double LX = 500;       /* Simulation box length */
double DT = 0.004;       /* Time discretization unit */
int NSTEP = 12500;       /* Number of simulation steps */
int NECAL = 100;       /* Interval to calculate energies */
double X0 = 250 ,S0 = 60,E0 =100; /* Center-of-mass, spread & energy of initial wave packet */
double BH = 5,BW = 20;    /* Barrier height & width */
double EH = 50;       /* Edge potential height */

/* Arrays **********************************************************************
psi[NX+2][2]:    psi[i][0|1] is the real|imaginary part of the wave function
                 on mesh point i
wrk[NX+2][2]:    Work array for a wave function
al[2][2]:        al[0|1][0|1] is the half|full-step diagonal kinetic propagator
                 (real|imaginary part)
bux[2][NX+2][2]: bux[0|1][i][] is the half|full-step upper off-diagonal kinetic
                 propagator on mesh i (real|imaginary part)
blx[2][NX+2][2]: blx[0|1][i][] is the half|full-step lower off-diagonal kinetic
                 propagator on mesh i (real|imaginary part)
v[NX+2]:         v[i] is the potential energy at mesh point i
u[NX+2][2]:      u[i][] is the potential propagator on i (real|imaginary part)
*******************************************************************************/
double psi[NX+2][2];
double wrk[NX+2][2];
double al[2][2];
double bux[2][NX+2][2],blx[2][NX+2][2];
double v[NX+2];
double u[NX+2][2];
double *dev_psi, *dev_wrk, *dev_u, *blx0, *blx1, *bux0, *bux1, *al0, *al1;

/* Variables *******************************************************************
dx   = Mesh spacing
ekin = Kinetic energy
epot = Potential energy
etot = Total energy
*******************************************************************************/
double dx;
double ekin,eking,epot,epotg,etot;
int nproc, myid, offset;
MPI_Status status;
MPI_Request request;
dim3 dimGrid(NUM_BLOCK,1,1); // Grid dimensions (only use 1D)
dim3 dimBlock(NUM_THREAD,1,1); // Block dimensions (only use 1D)

int main(int argc, char **argv) {
	
	int step; /* Simulation loop iteration index */

    int myid, nproc;
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&nproc); // # of MPI processes
	MPI_Comm_rank(MPI_COMM_WORLD,&myid); // My MPI rank
    
	

	init_param();  /* Read input parameters */
	init_prop();   /* Initialize the kinetic & potential propagators */
	init_wavefn(); /* Initialize the electron wave function */

	omp_set_num_threads(NUM_DEVICE); //OpenMp setup
	#pragma omp parallel private(step)
    {
		int mpid = omp_get_thread_num();
		offset = nx * mpid;
		cudaSetDevice(mpid % NUM_DEVICE);

		
		size_t size = sizeof(double)*2*(nx+2);
		cudaMalloc((void **) &dev_psi,size); 
		hostdevice(dev_psi, psi, offset);
		cudaMalloc((void **) &dev_wrk,size);
		hostdevice(dev_wrk, wrk, offset);
		cudaMalloc((void **) &dev_u,size);
		hostdevice(dev_u, u, offset);
		cudaMalloc((void **) &blx0,size);
		hostdevice(blx0, blx[0], offset);
		cudaMalloc((void **) &blx1,size);
		hostdevice(blx1, blx[1], offset);
		cudaMalloc((void **) &bux0,size);
		hostdevice(bux0, bux[0], offset);
		cudaMalloc((void **) &bux1,size);
		hostdevice(bux1, bux[1], offset);
		cudaMalloc((void **) &al0,sizeof(double)*2);
		cudaMemcpy(al0,al[0],sizeof(double)*2, cudaMemcpyHostToDevice);
		cudaMalloc((void **) &al1,sizeof(double)*2);
		cudaMemcpy(al1,al[1],sizeof(double)*2, cudaMemcpyHostToDevice);

		


		for (step=1; step<=NSTEP; step++){
			single_step(); /* Time propagation for one step, DT */
			
			
			if (step%NECAL==0) {
			#pragma omp master
			calc_energy(step);
			#pragma omp barrier	
			
		}		

		}



    }

    cudaFree(dev_psi);
	cudaFree(dev_wrk);
	cudaFree(dev_u);
	cudaFree(blx0);
	cudaFree(blx1);
	cudaFree(bux0);
	cudaFree(bux1);
	cudaFree(al0);
	cudaFree(al1);
	MPI_Finalize();
	return 0;
}





/*----------------------------------------------------------------------------*/
void init_param() {
/*------------------------------------------------------------------------------
	Initializes parameters by reading them from input file.
------------------------------------------------------------------------------*/
/*	FILE *fp;

	/* Read control parameters */
/*	fp = fopen("qd1.in","r");
	fscanf(fp,"%le",&LX);
	fscanf(fp,"%le",&DT);
	fscanf(fp,"%d",&NSTEP);
	fscanf(fp,"%d",&NECAL);
	fscanf(fp,"%le%le%le",&X0,&S0,&E0);
	fscanf(fp,"%le%le",&BH,&BW);
	fscanf(fp,"%le",&EH);
	fclose(fp);  

	/* Calculate the mesh size */
	dx = LX/NX;
}

/*----------------------------------------------------------------------------*/
void init_prop() {
/*------------------------------------------------------------------------------
	Initializes the kinetic & potential propagators.
------------------------------------------------------------------------------*/
	int stp,s,i,up,lw;
	double a,exp_p[2],ep[2],em[2];
	double x;

	/* Set up kinetic propagators */
	a = 0.5/(dx*dx);

	for (stp=0; stp<2; stp++) { /* Loop over half & full steps */
		exp_p[0] = cos(-(stp+1)*DT*a);
		exp_p[1] = sin(-(stp+1)*DT*a);
		ep[0] = 0.5*(1.0+exp_p[0]);
		ep[1] = 0.5*exp_p[1];
		em[0] = 0.5*(1.0-exp_p[0]);
		em[1] = -0.5*exp_p[1];

		/* Diagonal propagator */
		for (s=0; s<2; s++) al[stp][s] = ep[s];

		/* Upper & lower subdiagonal propagators */
		for (i=1; i<=NX; i++) { /* Loop over mesh points */
			if (stp==0) { /* Half-step */
				up = i%2;     /* Odd mesh point has upper off-diagonal */
				lw = (i+1)%2; /* Even               lower              */
			}
			else { /* Full step */
				up = (i+1)%2; /* Even mesh point has upper off-diagonal */
				lw = i%2;     /* Odd                 lower              */
			}
			for (s=0; s<2; s++) {
				bux[stp][i][s] = up*em[s];
				blx[stp][i][s] = lw*em[s];
			}
		} /* Endfor mesh points, i */
	} /* Endfor half & full steps, stp */

	/* Set up potential propagator */
	for (i=1; i<=NX; i++) {
		x = myid * LX + dx * i;
		/* Construct the edge potential */
		if ( (myid == 0 && i==1) || (myid == 1 && i==NX) )
			v[i] = EH;
		/* Construct the barrier potential */
		else if ( LX-0.5*BW < x && x < LX+0.5*BW )
			v[i] = BH;
		else
			v[i] = 0.0;
		/* Half-step potential propagator */
		u[i][0] = cos(-0.5*DT*v[i]);
		u[i][1] = sin(-0.5*DT*v[i]);
	}
}

/*----------------------------------------------------------------------------*/
void init_wavefn() {
/*------------------------------------------------------------------------------
	Initializes the wave function as a traveling Gaussian wave packet.
------------------------------------------------------------------------------*/
	int sx,s;
	double x,gauss,psisq,norm_fac, psisqq;

	/* Calculate the the wave function value mesh point-by-point */
	for (sx=1; sx<=NX; sx++) {
		x = myid * LX + dx * sx- X0;
		gauss = exp(-0.25*x*x/(S0*S0));
		psi[sx][0] = gauss*cos(sqrt(2.0*E0)*x);
		psi[sx][1] = gauss*sin(sqrt(2.0*E0)*x);
	}

	/* Normalize the wave function */
	psisq=0.0;
	for (sx=1; sx<=NX; sx++)
		for (s=0; s<2; s++)
			psisq += psi[sx][s]*psi[sx][s];
	psisq *= dx;
	MPI_Allreduce(&psisq, &psisqq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	norm_fac = 1.0/sqrt(psisqq);
	for (sx=1; sx<=NX; sx++)
		for (s=0; s<2; s++)
			psi[sx][s] *= norm_fac;
}

/*----------------------------------------------------------------------------*/
void single_step() {
/*------------------------------------------------------------------------------
	Propagates the electron wave function for a unit time step, DT.
------------------------------------------------------------------------------*/
	pot_prop();  /* half step potential propagation */

	kin_prop(0); /* half step kinetic propagation   */
	kin_prop(1); /* full                            */
	kin_prop(0); /* half                            */

	pot_prop();  /* half step potential propagation */
}

__global__ void gpu_pot_prop(double *dev_psi, double *dev_u) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int sx = tid+1;
	double wr,wi;
	wr = dev_u[2*sx+0] * dev_psi[2*sx+0] - dev_u[2*sx+1] * dev_psi[2*sx+1];
	wi = dev_u[2*sx] * dev_psi[2*sx+1] + dev_u[2*sx+1]*dev_psi[2*sx];
	dev_psi[2*sx] = wr;
	dev_psi[2*sx+1] = wi;
}

/*----------------------------------------------------------------------------*/
void pot_prop() {
/*------------------------------------------------------------------------------
	Potential propagator for a half time step, DT/2.
------------------------------------------------------------------------------*/

		hostdevice(dev_psi,psi,offset);

		gpu_pot_prop <<<dimGrid,dimBlock>>> (dev_psi,dev_u);

		devicehost(psi,dev_psi,offset);

		#pragma omp barrier

}

__global__ void gpu_kin_prop(double *dev_psi, double *dev_wrk, double *dev_al, double *dev_blx, double *dev_bux) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int sx = tid+1;
    double wr=0, wi=0;
	wr = dev_al[0]*dev_psi[2*sx]-dev_al[1]*dev_psi[2*sx+1];
	wi = dev_al[0]*dev_psi[2*sx+1]+dev_al[1]*dev_psi[2*sx];
	wr += ( dev_blx[2*sx]*dev_psi[2*(sx-1)]-dev_blx[2*sx+1]*dev_psi[2*(sx-1)+1] );
	wi += (dev_blx[2*sx]*dev_psi[2*(sx-1)+1] + dev_blx[2*sx+1]*dev_psi[2*(sx-1)] );
	wr += (dev_bux[2*sx]*dev_psi[2*(sx+1)]-dev_bux[2*sx+1]*dev_psi[2*(sx+1)+1] );
	wi += (dev_bux[2*sx] * dev_psi[2*(sx+1)+1] + dev_bux[2*sx+1] * dev_psi[2*(sx+1)] );
	dev_wrk[2*sx] = wr;
	dev_wrk[2*sx+1] = wi;
}

/*----------------------------------------------------------------------------*/
void kin_prop(int t) {
/*------------------------------------------------------------------------------
	Kinetic propagation for t (=0 for DT/2--half; 1 for DT--full) step.
-------------------------------------------------------------------------------*/
	
	

	#pragma omp master
	/* Apply the periodic boundary condition, Only master thread ID 0 executes, while letting the others wait */
	periodic_bc();
	#pragma omp barrier

	/* WRK|PSI holds the new|old wave function */

		hostdevice(dev_psi,psi,offset);
		if(t==0){
			gpu_kin_prop <<<dimGrid,dimBlock>>> (dev_psi,dev_wrk,al0,blx0,bux0);

		}
		else{
			gpu_kin_prop <<<dimGrid,dimBlock>>> (dev_psi,dev_wrk,al1,blx1,bux1);

		}
		devicehost(psi,dev_wrk,offset);
		#pragma omp barrier


}

/*----------------------------------------------------------------------------*/
void periodic_bc() {
/*------------------------------------------------------------------------------
	Applies the periodic boundary condition to wave function PSI, by copying
	the boundary values to the auxiliary array positions at the other ends.
------------------------------------------------------------------------------*/
	int s;
	int pup = (myid+1)%nproc , plw = (myid-1+nproc)%nproc; /* Upper / Lower partner process */
	double dbuf[2] , dbufr[2] ;

/* Cache boundary wave function value at the lower end */
	for (s=0; s<2; s++){
        dbuf[s] = psi[NX][s];
	}
	MPI_Irecv(dbufr, 2, MPI_DOUBLE, plw, 1, MPI_COMM_WORLD, &request);
	MPI_Send(dbuf, 2, MPI_DOUBLE, pup, 1, MPI_COMM_WORLD);
	MPI_Wait(&request, &status);
	for (s=0; s<=1; s++){
		psi[0][s] = dbufr[s];
	}

	/* Cache boundary wave function value at the upper end */
	for (s=0; s<2; s++){
		dbuf[s] = psi[1][s];
	}
	MPI_Irecv(dbufr, 2, MPI_DOUBLE, pup, 2, MPI_COMM_WORLD, &request);
	MPI_Send(dbuf, 2, MPI_DOUBLE, plw, 2, MPI_COMM_WORLD);
	MPI_Wait(&request, &status);

	for (s=0; s<=1; s++){
		psi[NX+1][s] = dbufr[s];
	}
}

/*----------------------------------------------------------------------------*/
void calc_energy(int step) {
/*------------------------------------------------------------------------------
	Calculates the kinetic, potential & total energies, EKIN, EPOT & ETOT.
------------------------------------------------------------------------------*/
	int sx,s;
	double a,bx;


    /* Apply the periodic boundary condition */
	periodic_bc();
	

	/* Tridiagonal kinetic-energy operators */
	a =   1.0/(dx*dx);
	bx = -0.5/(dx*dx);

	/* |WRK> = (-1/2)Laplacian|PSI> */
	for (sx=1; sx<=NX; sx++)
		for (s=0; s<=1; s++)
			wrk[sx][s] = a*psi[sx][s]+bx*(psi[sx-1][s]+psi[sx+1][s]);

	/* Kinetic energy = <PSI|(-1/2)Laplacian|PSI> = <PSI|WRK> */
	ekin = 0.0;
	eking = 0;
	for (sx=1; sx<=NX; sx++)
		ekin += (psi[sx][0]*wrk[sx][0]+psi[sx][1]*wrk[sx][1]);
	ekin *= dx;
	MPI_Allreduce(&ekin, &eking, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	/* Potential energy */
	epot = 0.0;
	epotg = 0;
	for (sx=1; sx<=NX; sx++)
		epot += v[sx]*(psi[sx][0]*psi[sx][0]+psi[sx][1]*psi[sx][1]);
	epot *= dx;
	MPI_Allreduce(&epot, &epotg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	/* Total energy */
	etot = eking+epotg;
	printf("%le %le %le %le\n",DT*step,ekin,epot,etot);
}
