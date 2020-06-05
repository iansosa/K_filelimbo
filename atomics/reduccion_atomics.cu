#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <ctime>
#include <sys/time.h>
#include <sstream>
#include <string>
#include <fstream>
#include "gpu_timer.h"
#include "cpu_timer.h"

using namespace std;

// block reduction
__global__ void reduce1(int *g_idata, int *g_odata, int size){

   extern __shared__ int sdata[];

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


// block reduction
__global__ void reduce1_atomic(int *g_idata, int *g_odata, int size){

   extern __shared__ int sdata[];

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
   if (tid == 0) atomicAdd(g_odata,sdata[0]);
}




int main(int argc, char **argv){

  assert(argc==3);
  int size = atoi(argv[1]);
  // crea un vector de host de size "ints" y lo inicializa a 1
  thrust::host_vector<int> data_h_i(size, 1);

  //initialize the data, all values will be 1
  //so the final sum will be equal to size
  int threadsPerBlock = 256;
  int totalBlocks = (size+(threadsPerBlock-1))/threadsPerBlock;

  // una forma de "empaquetar" vectores de device usando thrust
  // crea y aloca un vector de device de "int" y copia el contenido de un vector de host 
  thrust::device_vector<int> data_v_i = data_h_i;

  // crea y aloca un vector de device de totalBlocks elementos "int" 
  thrust::device_vector<int> data_v_o(totalBlocks);

  // los vectors de device son algo mas que un punter a memoria de device
  int* output = thrust::raw_pointer_cast(data_v_o.data());
  int* input = thrust::raw_pointer_cast(data_v_i.data());

  gpu_timer reloj;
  cpu_timer relojcpu;
  int sum=0;

  switch(atoi(argv[2])){

	case 0:
	relojcpu.tic();
	sum = thrust::reduce(data_h_i.begin(),data_h_i.end());
  	cout << "reduccion con thrust en host: " << sum << " (dio bien?) en " << relojcpu.tac() << " ms" << std::endl;
	break;

	case 1:
	reloj.tic();
  	// todo lo anterior se puede hacer en una línea usando la biblioteca thrust 
	sum = thrust::reduce(thrust::device,input,input+size);
  	cout << "reduccion con thrust en device: " << sum 
	     << " (dio bien?) en " << reloj.tac() << " ms" << std::endl;
	break;

	// se puede hacer con un solo kernel? 
	case 2:
	reloj.tic();
	// reduccion en arbol en la memoria shared de cada bloque
  	reduce1_atomic<<<totalBlocks, threadsPerBlock, threadsPerBlock*sizeof(int)>>>(input, output, size);
	sum=data_v_o[0];
	std::cout << "reduccion en bloques+atomic: " << sum << " (dio bien?) en " << reloj.tac() << " ms" << std::endl;
	std::cout << "totalBlocks " << totalBlocks << std::endl;
	break;
  }


  return 0;

}
