#include <cstdio>
#include <cstdlib>
#include <cassert>
#include "gpu_timer.h"
#include "cpu_timer.h"


/* Size of the input data */
#define N 8388608  
/* Size of the filter */
#define M 64


/* Floating point type */
typedef float FLOAT;

/* Function to setup the filter */
void SetupFilter(FLOAT* filter, int size) {
	for(int i = 0 ; i < size ; i++)
		filter[i] = 1.0/size;
}


/* convolucion en la cpu: requiere dos loops */
void conv_sec(FLOAT* input, FLOAT* output, FLOAT * filter) 
{
	FLOAT temp;
	for(int j=0;j<N;j++){
		temp=0.0;
		for(int i=0;i<M;i++){
	  		temp += filter[i]*input[i+j];
		}
		output[j] = temp;
	}
}

// kernel
__global__ void conv_par(FLOAT* input, FLOAT* output, FLOAT* filter) 
{	  	
	//TODO: completar el kernel de convolucion
	unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;
	FLOAT temp=0;
	if(j<N)
	{
		temp=0.0;
		for (int i = 0; i < M; ++i)
		{
			temp += filter[i]*input[i+j];
		}
		output[j] = temp;
	}
}


////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) 
{
    cudaDeviceProp deviceProp;
    int dev; 
    cudaGetDevice(&dev);
    
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);


	/* Allocate memory on host */
	FLOAT *h_input = (FLOAT *) malloc((N+M) * sizeof(FLOAT));  /* Input data */
	FLOAT *h_output = (FLOAT *) malloc(N * sizeof(FLOAT)); 	   /* Output data */
	FLOAT *check_output = (FLOAT *) malloc(N * sizeof(FLOAT)); /* CPU check */
	/* Allocate memory for filter */
	FLOAT *h_filter = (FLOAT*) malloc(M * sizeof(FLOAT));

	/* Setup the filter */
	SetupFilter(h_filter, M);

	/* Fill (padded periodico) input array with random data */
	for(int i = 0 ; i < N ; i++) 
		h_input[i] = (FLOAT)(rand() % 100); 
	
	for(int i = N ; i < N+M ; i++) 
		h_input[i] = h_input[i-N];


	/* Allocate memory on device */
	FLOAT *d_input, *d_output, *d_filter;
	//TODO: Alocar memoria en device	
	cudaMalloc(&d_input, sizeof(FLOAT)*(N+M));
	cudaMalloc(&d_output, sizeof(FLOAT)*(N));
	cudaMalloc(&d_filter, sizeof(FLOAT)*(M));
	/* Copy input array to device */
	//TODO: copiar d_input <- h_input
	cudaMemcpy(d_input, h_input,(N+M) * sizeof(FLOAT), cudaMemcpyHostToDevice);
	/* Copy the filter to the GPU */
	//TODO: copiar d_filter <- h_filter
	cudaMemcpy(d_filter, h_filter,M * sizeof(FLOAT), cudaMemcpyHostToDevice);
	/* check in the CPU */
	cpu_timer crono_cpu; 
	crono_cpu.tic();
	conv_sec(h_input, check_output, h_filter);
	crono_cpu.tac();

	// TODO: armar una grilla adecuada	
	int threadsPerBlock = 512;
	int totalBlocks = (N+(threadsPerBlock-1))/threadsPerBlock;
	//dim3 block_size(...);
  	//dim3 grid_size(...);

	// TODO: lanzar el kernel 
	/* distintos kernels */
	gpu_timer crono_gpu;
	crono_gpu.tic();
	conv_par<<<totalBlocks,threadsPerBlock>>>(d_input, d_output, d_filter);
	cudaDeviceSynchronize();
	crono_gpu.tac();
	
	printf("[M/N/ms_cpu/ms_gpu]= %d %d %lf  %lf  \n", M, N, crono_cpu.ms_elapsed, crono_gpu.ms_elapsed);

	/* Copy output array to host */
	cudaMemcpy(h_output, d_output, sizeof(FLOAT)*(N), cudaMemcpyDeviceToHost);
	//TODO: completar h_output <- d_output	
	
	/* comparacion */
	//TODO completar
	FLOAT error, maxerror;
	for(int j=0;j<N;j++){
		error = fabs(h_output[j]-check_output[j]); 
		if(maxerror<error) maxerror=error;
	}
	printf("error maximo= %lf (¿es chico?) \n \n", maxerror);

	//TODO: 
	//grafique check_output y h_output para ver graficamente el "filtrado"

	/* Free memory on host */
	free(h_input);
	free(h_output);
	free(h_filter);
	free(check_output);

	/* Free memory on device */
	// TODO: liberar memoria device
	// ...	
}

