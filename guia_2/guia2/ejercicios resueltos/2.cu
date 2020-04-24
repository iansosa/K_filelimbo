#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include "gpu_timer.h"
#include "cpu_timer.h"


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
/* Floating point type */
typedef float FLOAT;

__host__ __device__ FLOAT function( FLOAT x)
{
    return (x);
}


// kernel
__global__ void integrate_0(FLOAT* output, int N, FLOAT a, FLOAT b) 
{	  	
	//TODO: completar el kernel de convolucion
	unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;
	FLOAT h=(b-a)/N;
	if(j<N)
	{
		output[j]=h*(function(a+j*h)+function(a+(j+1)*h))*0.5;
	}
}



FLOAT reduceCPU(FLOAT *output,int N)
{
	FLOAT aux=0;
	for (int i = 0; i < N; ++i)
	{
		aux=aux+output[i];
	}
	return(aux);
}

FLOAT allCPU(FLOAT a, FLOAT b, int N)
{
	FLOAT output[N];
	FLOAT h=(b-a)/N;
	for (int j = 0; j < N; ++j)
	{
		output[j]=h*(function(a+j*h)+function(a+(j+1)*h))*0.5;
	}
	return(reduceCPU(&output[0],N));
}


int main(int argc, char *argv[]) 
{
	/* Allocate memory on host */
	int N;
    printf("N: ");
    std::cin >>N;

    FLOAT a;
    printf("a: ");
    std::cin >>a;

    FLOAT b;
    printf("b: ");
    std::cin >>b;

    int gpu_caso;
   	printf("GPU CPU or GPU GPU (0 or 1): ");
   	std::cin >>gpu_caso;


	cpu_timer crono_cpu; 
	crono_cpu.tic();
	FLOAT allcpucalc=allCPU(a,b,N);
	crono_cpu.tac();
	printf("allCPU: %lf. Time: %lf \n", allcpucalc,crono_cpu.ms_elapsed);



	FLOAT *h_output = (FLOAT *) malloc(N * sizeof(FLOAT));

	/* Allocate memory on device */
	FLOAT  *d_output;
	FLOAT  *d_output2;
	//TODO: Alocar memoria en device	
	cudaMalloc(&d_output, sizeof(FLOAT)*(N));


	int threadsPerBlock = 512;
	int totalBlocks = (N+(threadsPerBlock-1))/threadsPerBlock;
	printf("%d\n", totalBlocks );
	cudaMalloc(&d_output2, sizeof(FLOAT)*(totalBlocks));
	gpu_timer crono_gpu;
	FLOAT somegpucalc;
	if(gpu_caso==0)
	{
		crono_gpu.tic();
		integrate_0<<<totalBlocks,threadsPerBlock>>>(d_output, N, a,b);
		cudaMemcpy(h_output, d_output, sizeof(FLOAT)*(N), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		somegpucalc=reduceCPU(&h_output[0],N);
		crono_gpu.tac();
		printf("someGPU: %lf. Time: %lf \n", somegpucalc,crono_gpu.ms_elapsed);
	}
  	thrust::host_vector<FLOAT> data_h_i(N);
  	thrust::device_vector<FLOAT> data_v_i = data_h_i;
  	FLOAT* input = thrust::raw_pointer_cast(data_v_i.data());
	if(gpu_caso==1)
	{
		crono_gpu.tic();
		integrate_0<<<totalBlocks,threadsPerBlock>>>(input, N, a,b);
		crono_gpu.tac();
		printf("allGPU: %lf. Time: %lf \n",thrust::reduce(thrust::device,input,input+N),crono_gpu.ms_elapsed);
	}


}

