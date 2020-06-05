#include <cstdio>
#include <cstdlib>
#include <cassert>
#include "gpu_timer.h"
#include "cpu_timer.h"
#include <cuda_runtime_api.h>
#include <cufft.h>


/* Kernel Execution Parameters Parameters */
const int BLOCK_SIZE = 512;


const int N = 2 << 22;//8388608;  
const int Nh = 64;
const int M = 4096;


/* Floating point type */
typedef float FLOAT;

/* Function to setup the filter */
static void SetupFilter(FLOAT* filter, size_t size, void* user) {
	for(int i = 0 ; i < size ; i++)
		filter[i] = 1.0/size;
}


/* convolucion en la cpu: requiere dos loops */
void conv_cpu(const FLOAT* input, FLOAT* output, const FLOAT* filter) 
{
	FLOAT temp;
	for(int j=0;j<N;j++){
		temp=0.0;
		for(int i=0;i<Nh;i++){
	  		temp += filter[i]*input[i+j];
		}
		output[j] = temp;
	}
}

// convolucion usando indexado unidimensional de threads/blocks
// un thread por cada elemento del output
// todo en memoria global
// lanzamiento: la grilla se puede elegir independiente de N
__global__ void conv_one_thread_per_output_element_all_global
(const FLOAT* input, FLOAT* output, const FLOAT* filter) 
{	  	
	  int j = blockIdx.x * blockDim.x + threadIdx.x;

	  FLOAT temp;
	  while(j<N)
	  {
	  	temp=0.0;
	  	for(int i=0;i<Nh;i++){
	  		temp += filter[i]*input[i+j];
		}	  
	 	output[j]=temp;
		j+=gridDim.x*blockDim.x;
	  }
}

// convolucion usando indexado unidimensional de threads/blocks
// un thread por cada elemento del output
// filtro en memoria constante, el resto en global
// lanzamiento: la grilla se puede elegir independiente de N
__constant__ FLOAT d_filtro_constant[Nh];

__global__ void conv_one_thread_per_output_element_filter_in_constant
(const FLOAT* input, FLOAT* output, const FLOAT* filter) 
{	  	
	  int j = blockIdx.x * blockDim.x + threadIdx.x;

	  FLOAT temp;
	  while(j<N)
	  {
	  	temp=0.0;
	  	for(int i=0;i<Nh;i++){
	  		temp += d_filtro_constant[i]*input[i+j]; // cuidado: solo 64K de constant memory
	  	}	  
	 	output[j]=temp;
		j+=gridDim.x*blockDim.x;
	  }
}




// convolucion usando indexado unidimensional de threads/blocks
// un thread por cada elemento del output
// filtro en memoria shared, datos en global
// lanzamiento: la grilla se puede elegir independiente de N y Nh
__global__ void conv_one_thread_per_output_element_filter_in_shared
(const FLOAT* input, FLOAT* output, const FLOAT* filter) 
{	  	
	  __shared__ FLOAT sh_filter[Nh];
	  int j = blockIdx.x * blockDim.x + threadIdx.x;
	  int tx = threadIdx.x;	

	  // Como blockDim.x puede ser menor a Nh, me aseguro 
	  // que los blockDim.x threads carguen todo el filtro 
	  // dandoles como tarea cargar mas de un elemento, si fuera necesario
	  while(tx<Nh){
		sh_filter[tx]=filter[tx];
		tx+=blockDim.x;
	  }
	  __syncthreads();

	  FLOAT temp; 
	  while(j<N)
	  {
	  	temp=0.0; 
	  	for(int i=0;i<Nh;i++){
	  		temp += sh_filter[i]*input[i+j];
	  	}	  
	  	output[j]=temp;
		j+=gridDim.x*blockDim.x;
	  }
}



// convolucion usando indexado unidimensional de threads/blocks
// un bloque calcula cada elemento del output (que se queda en global)
// filtro en memoria shared, datos segmentados en ventanas cargadas en shared
__global__ void conv_one_thread_per_output_element_all_in_shared
(const FLOAT* input, FLOAT* output, const FLOAT* filter) 
{	  	
	  int k = blockIdx.x;
	  int tx = threadIdx.x;	

	  __shared__ FLOAT sh_filter[Nh];  // sh_filter[i] = filter[i]
	  __shared__ FLOAT sh_input[M+Nh-1]; // sh_input[i]  <= input[k*M+i] 

	  // cargo filtro
	  // caso blockDim.x < Nh contemplado	
	  tx = threadIdx.x;	
	  while(tx<Nh){	
		sh_filter[tx]=filter[tx]; 
		tx+=blockDim.x;
	  }

	  while(k*M<N)// caso gridDim.x*M < N contemplado 
	  { 
	
		  // carga ventana con padding 	
		  // caso blockDim.x < Nh+M contemplado	
		  tx = threadIdx.x;	
		  while(tx<M+Nh){ 
			sh_input[tx]=input[M*k+tx];
			tx+=blockDim.x;		    	
		  }	
		  __syncthreads();	 	

		  // aqui cada thread del bloque "k" calcula uno de estos: output[k*M],...,output[(k+1)*M-1]
		  // caso blockDim.x < M contemplado	
		  tx = threadIdx.x;		  
		  FLOAT temp;	  
		  while(tx<M){
			temp=0.0;
			for(int i=0;i<Nh;i++){
				temp += sh_filter[i]*sh_input[i+tx];
			}	  	  	  		
			output[k*M+tx] = temp; 
			tx+=blockDim.x;		    	
		  }
		  k+=gridDim.x;
		  __syncthreads();// no puedo cargar mas datos si todo el output no esta listo...	 	
	  }
}



// convolucion usando indexado unidimensional de threads/blocks
// un bloque calcula cada elemento del output (que se queda en global)
// filtro en memoria constante, datos segmentados en ventanas cargadas en shared
__global__ void conv_one_thread_per_output_element_input_in_shared_filter_in_constant
(const FLOAT* input, FLOAT* output, const FLOAT* filter) 
{	  	
	  int k = blockIdx.x;
	  int tx = threadIdx.x;	

	  __shared__ FLOAT sh_input[M+Nh-1]; // sh_input[i]  <= input[k*M+i] 

	  while(k*M<N)// caso gridDim.x*M < N contemplado 
	  { 
		  // carga ventana con padding 	
		  // caso blockDim.x < Nh+M contemplado	
		  tx = threadIdx.x;	
		  while(tx<M+Nh){ 
			sh_input[tx]=input[M*k+tx];
			tx+=blockDim.x;		    	
		  }	
		  __syncthreads();	 	

		  // aqui cada thread del bloque "k" calcula uno de estos: output[k*M],...,output[(k+1)*M-1]
		  // caso blockDim.x < M contemplado	
		  tx = threadIdx.x;		  
		  FLOAT temp;
	  
		  while(tx<M){
			temp=0.0;
			for(int i=0;i<Nh;i++){
				temp += d_filtro_constant[i]*sh_input[i+tx];
			}	  	  	  		
			output[k*M+tx] = temp; 
			tx+=blockDim.x;		    	
		  }
		  k+=gridDim.x;
		  __syncthreads();	 	
	  }
}



// convolucion usando indexado unidimensional de threads/blocks
// un thread por cada elemento del output
// todo en memoria global, pero el filtro se carga en registros...
// lanzamiento: la grilla se puede elegir independiente de N
__global__ void conv_one_thread_per_output_element_input_global_filter_in_register
(const FLOAT* input, FLOAT* output, const FLOAT* filter) 
{	  	
	  int j = blockIdx.x * blockDim.x + threadIdx.x;
	 
      FLOAT r_filter[Nh];
	  for(int i=0;i<Nh;i++) r_filter[i]=filter[i];

	  FLOAT temp;
	  while(j<N)
	  {
	  	temp=0.0;
	  	for(int i=0;i<Nh;i++){
	  		temp += r_filter[i]*input[i+j];
	  	}	  
	 	output[j]=temp;
		j+=gridDim.x*blockDim.x;
	  }
}

void correctitud(FLOAT *, FLOAT *, FLOAT *);

int main(int argc, char *argv[]) 
{
	cudaDeviceProp deviceProp;
	int dev; cudaGetDevice(&dev);
	    
	cudaGetDeviceProperties(&deviceProp, dev);
	printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);


	/* Imprime informacion general */
	printf("************************************* \n");
	printf("Tamanio de la senial: %d\n", N);
	printf("Tamanio del filtro: %d\n", Nh);
	printf("Tamanio ventana: %d\n", M);
	printf("************************************* \n");


	/* Aloca memoria en host */
	FLOAT* h_input = (FLOAT *) malloc((N+Nh) * sizeof(FLOAT));  /* Input data */
	FLOAT* h_output = (FLOAT *) malloc(N * sizeof(FLOAT)); /* Output data */
	FLOAT* check_output = (FLOAT *) malloc(N * sizeof(FLOAT)); /* CPU Output data */
	FLOAT* h_filter = (FLOAT*) malloc(Nh * sizeof(FLOAT));


	/* Setup the filter */
	SetupFilter(h_filter, Nh, 0);

	/* Fill (padded periodico) input array with random data */
	for(int i = 0 ; i < N ; i++) 
		h_input[i] = (FLOAT)(rand() % 100); 
	for(int i = N ; i < N+Nh ; i++) 
		h_input[i] = h_input[i-N];


	/* Allocate memory on device */
	FLOAT *d_input, *d_output, *d_filter; //, *d_extended_filter;
	cudaMalloc((void**)&d_input, (N+Nh) * sizeof(FLOAT));
	cudaMalloc((void**)&d_output, N * sizeof(FLOAT));
	cudaMalloc((void**)&d_filter, Nh * sizeof(FLOAT));

	// pongo a cero el device output
	cudaMemset(d_output,0,N * sizeof(FLOAT));

	/* Copy input array to device */
	cudaMemcpy(d_input, h_input, (N+Nh) * sizeof(FLOAT), cudaMemcpyHostToDevice);

	/* Copy the filter to the GPU */
	cudaMemcpy(d_filter, h_filter, Nh * sizeof(FLOAT), cudaMemcpyHostToDevice);

	/* Copy the filter to the GPU in constant memory */
	cudaMemcpyToSymbol(d_filtro_constant,h_filter,sizeof(FLOAT)*Nh);

	/* Sanity check */
	assert(Nh <= M);
	assert(M <= N);
	assert(BLOCK_SIZE < 1025);
	
	/////////////////////////////////////////////////////////////////
	/* check in the CPU */
	cpu_timer cronocpu;
	cronocpu.tic();
	conv_cpu(h_input, check_output, h_filter);
	cronocpu.tac();
	printf("CPU = %lf\n\n", cronocpu.ms_elapsed);
	
	/////////////////////////////////////////////////////////////////
	/* distintos kernels */
	gpu_timer crono;
				
	crono.tic();
	conv_one_thread_per_output_element_all_global<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_input, d_output, d_filter);
	cudaDeviceSynchronize();
	crono.tac();
	printf("filtro e input en memoria global: %lf (x%2.0f) \n", crono.ms_elapsed, cronocpu.ms_elapsed/crono.ms_elapsed);
	correctitud(d_output,h_output,check_output);
			
	/////////////////////////////////////////////////////////////////
	// pongo a cero el device output
	cudaMemset(d_output,0,N * sizeof(FLOAT));
					
	crono.tic();
	conv_one_thread_per_output_element_filter_in_constant <<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_input, d_output, d_filter);
	cudaDeviceSynchronize();
	crono.tac();
	printf("filtro en memoria constante, input en memoria global: %lf (x%2.0f)\n", 
	crono.ms_elapsed, cronocpu.ms_elapsed/crono.ms_elapsed);
	correctitud(d_output,h_output,check_output);


	/////////////////////////////////////////////////////////////////
	// pongo a cero el device output
	cudaMemset(d_output,0,N * sizeof(FLOAT));
		
	crono.tic();
	conv_one_thread_per_output_element_filter_in_shared<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_input, d_output, d_filter);
	cudaDeviceSynchronize();
	crono.tac();
	printf("filtro en memoria shared, input en memoria global: %lf (x%2.0f)\n", 
	crono.ms_elapsed,cronocpu.ms_elapsed/crono.ms_elapsed);
	correctitud(d_output,h_output,check_output);

	/////////////////////////////////////////////////////////////////
	// pongo a cero el device output
	cudaMemset(d_output,0,N * sizeof(FLOAT));		

	crono.tic();
	conv_one_thread_per_output_element_all_in_shared<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_input, d_output, d_filter);
	cudaDeviceSynchronize();
	crono.tac();
	printf("filtro en memoria shared, input en memoria shared: %lf (x%2.0f)\n", 
	crono.ms_elapsed,cronocpu.ms_elapsed/crono.ms_elapsed);
	correctitud(d_output,h_output,check_output);
			
	/////////////////////////////////////////////////////////////////
	// pongo a cero el device output
	cudaMemset(d_output,0,N * sizeof(FLOAT));
					
	crono.tic();
	conv_one_thread_per_output_element_input_in_shared_filter_in_constant<<<N/BLOCK_SIZE,BLOCK_SIZE>>>
	(d_input, d_output,d_filter);
	cudaDeviceSynchronize();
	crono.tac();
	printf("filtro en memoria constante, input en memoria shared: %lf (x%2.0f)\n", 
	crono.ms_elapsed,cronocpu.ms_elapsed/crono.ms_elapsed);
	correctitud(d_output,h_output,check_output);
		
	/////////////////////////////////////////////////////////////////
	// pongo a cero el device output
	cudaMemset(d_output,0,N * sizeof(FLOAT));
					
	crono.tic();
	conv_one_thread_per_output_element_input_global_filter_in_register<<<N/BLOCK_SIZE,BLOCK_SIZE>>>
	(d_input, d_output, d_filter);
	cudaDeviceSynchronize();
	crono.tac();
	printf("filtro en registro, input en memoria global: %lf (x%2.0f)\n", 
	crono.ms_elapsed,cronocpu.ms_elapsed/crono.ms_elapsed);
	correctitud(d_output,h_output,check_output);

	/////////////////////////////////////////////////////////////////
	/* Free memory on host */
	free(h_input);
	free(h_output);
	free(h_filter);

	/* Free memory on device */
	cudaFree(d_input);
	cudaFree(d_output);
	cudaFree(d_filter);

}


void correctitud(FLOAT *d_output, FLOAT *h_output, FLOAT *check_output)
{
	/* Copy output array to host */
	cudaMemcpy(h_output, d_output, N * sizeof(FLOAT), cudaMemcpyDeviceToHost);

	/* comparacion */
	FLOAT error, maxerror;
	for(int j=0;j<N;j++){
		// descomentar para imprimir output
		//printf("%d %f %f %f\n",j, h_input[j], h_output[j], check_output[j]);
		error = fabs(h_output[j]-check_output[j]); //'*100/fabs(check_output[j]);
		if(maxerror<error) maxerror=error;
	}
	printf("error maximo, emax = %lf \n\n", maxerror);
}
