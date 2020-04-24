#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/detail/config.h>
#include <thrust/detail/execution_policy.h>
//#include <thrust/detail/scan.inl>
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

__global__ void reduce1(FLOAT *g_idata, FLOAT *g_odata, int size){

   extern __shared__ FLOAT sdata[];

   unsigned int tid = threadIdx.x;
   unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
   sdata[tid] = 0;
   if(i<size)
     sdata[tid] = g_idata[i];
   __syncthreads();

   for(unsigned int s=blockDim.x/2; s>0; s>>=1) 
   {
	if (tid < s) {
		sdata[tid] += sdata[tid + s];
	}
	__syncthreads();
   }
   if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void sumandos(FLOAT *g_idata, int size){


   unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

   if(i<size)
   {
   		g_idata[i]=6.0/((1+i)*(1+i));
   }
}

struct mi_operacion
{
	  __device__
	  float operator()(float a)
	  {
		  return 6/(a*a);
	  }
};

struct mi_operacion_inclusive
{
	  __device__
	  double operator()(double a,double b)
	  {
		  return b*b;
	  }
};

struct mi_operacion_inclusive_1
{
	  __device__
	  double operator()(double a,double b)
	  {
	  	//printf("%lf+%lf=%lf\n",a,b, a+1.0/b);
		  return a+1.0/b;
	  }
};


int main(int argc, char *argv[]) 
{
	/* Allocate memory on host */
	int N;
    printf("N: ");
    std::cin >>N;


  	thrust::host_vector<FLOAT> h_x(N, 1);
 	thrust::device_vector<FLOAT> d_x_in = h_x;

  	int threadsPerBlock = 1024;
	int totalBlocks = (N+(threadsPerBlock-1))/threadsPerBlock;

  	thrust::device_vector<FLOAT> d_x_out(totalBlocks);


  	FLOAT* output = thrust::raw_pointer_cast(d_x_out.data());
  	FLOAT* input = thrust::raw_pointer_cast(d_x_in.data());
  	gpu_timer reloj;

  	reloj.tic();
  	sumandos<<<totalBlocks, threadsPerBlock>>>(input, N);
  	reduce1<<<totalBlocks, threadsPerBlock, threadsPerBlock*sizeof(FLOAT)>>>(input, output, N);
  	reduce1<<<1, threadsPerBlock, threadsPerBlock*sizeof(FLOAT)>>>(output, input, totalBlocks);
  	thrust::copy(d_x_in.begin(), d_x_in.end(), h_x.begin());
  	reloj.tac();
  	printf("version 1 %lf    %lfms\n",sqrt(h_x[0]) ,reloj.ms_elapsed);

  	reloj.tic();
  	FLOAT v2=thrust::transform_reduce(thrust::make_counting_iterator(1),thrust::make_counting_iterator(N),mi_operacion(),0.0,thrust::plus<float>());
  	reloj.tac();
  	printf("version 2 %lf    %lfms\n",sqrt(v2) ,reloj.ms_elapsed);

  	thrust::host_vector<double> h_x_serie1(N, 1);
 	thrust::device_vector<double> d_x_in_serie1 = h_x_serie1;
  	reloj.tic();
  	thrust::inclusive_scan(thrust::make_counting_iterator(1),thrust::make_counting_iterator(N),d_x_in_serie1.begin(),mi_operacion_inclusive());
  	thrust::inclusive_scan(d_x_in_serie1.begin(),d_x_in_serie1.end(),d_x_in_serie1.begin(),mi_operacion_inclusive_1());
  	thrust::copy(d_x_in_serie1.begin(), d_x_in_serie1.end(), h_x_serie1.begin());
  	reloj.tac();
  	FILE *f;
  	f=fopen("serie1.txt", "w");
  	for (int i = 0; i < N-1; ++i)
  	{
  		fprintf(f,"%d    %lf\n",i+1,sqrt(6*h_x_serie1[i]));
  	}
  	fclose(f);
  	
}

