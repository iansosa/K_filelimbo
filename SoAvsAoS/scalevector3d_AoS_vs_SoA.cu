#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/random.h>
#include <assert.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/for_each.h>

#include "timer.h"

// Este ejemplo escalea las componentes de un vector de puntos 3d.
// usando AoS y SoA. Usa kernels.

struct MyStruct
{
  int x;
  int y;
  int z;
};

struct MyStruct2
{
  int *x;
  int *y;
  int *z;
};


void initialize_keys(thrust::device_vector<int>& x,thrust::device_vector<int>& y,thrust::device_vector<int>& z)
{
  thrust::default_random_engine rng;
  thrust::uniform_int_distribution<int> dist(0, 2147483647);

  thrust::host_vector<int> h_x(x.size());
  thrust::host_vector<int> h_y(y.size());
  thrust::host_vector<int> h_z(z.size());

  for(size_t i = 0; i < h_x.size(); i++){
    	h_x[i] = dist(rng);
    	h_y[i] = dist(rng);
    	h_z[i] = dist(rng);
  }

  x = h_x;
  y = h_y;
  z = h_z;
}


void initialize_keys(thrust::device_vector<MyStruct>& structures)
{
  thrust::default_random_engine rng;
  thrust::uniform_int_distribution<int> dist(0, 2147483647);

  thrust::host_vector<MyStruct> h_structures(structures.size());

  for(size_t i = 0; i < h_structures.size(); i++){
    h_structures[i].x = dist(rng);
    h_structures[i].y = dist(rng);
    h_structures[i].z = dist(rng);
  }

  structures = h_structures;
}


__global__ 
void miKernel1(MyStruct *S,int N){
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i<N){
		S[i].x*=1;
		S[i].y*=2;
		S[i].z*=3;
	}
}

__global__ 
void miKernel2(MyStruct2 S, int N){
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i<N){
		S.x[i]*=1;
		S.y[i]*=2;
		S.z[i]*=3;
	}
}



int main(void)
{
  size_t N = 2 * 1024 * 1024;

  // Sort Key-Value pairs using Array of Structures (AoS) storage 
  {
    thrust::device_vector<MyStruct> structures(N);

    initialize_keys(structures);

    timer t;

    MyStruct * d_str=thrust::raw_pointer_cast(&structures[0]);	

    miKernel1<<<(N+256-1)/256,256>>>(d_str,N);	

    std::cout << "Escaleo AoS " << 1e3 * t.elapsed() << " milliseconds" << std::endl;
  }

  // Sort Key-Value pairs using Structure of Arrays (SoA) storage 
  {
    thrust::device_vector<int> x(N);
    thrust::device_vector<int> y(N);
    thrust::device_vector<int> z(N);

    initialize_keys(x,y,z);

    timer t;

    int * d_x=thrust::raw_pointer_cast(&x[0]);	
    int * d_y=thrust::raw_pointer_cast(&y[0]);	
    int * d_z=thrust::raw_pointer_cast(&z[0]);	
    
    MyStruct2 SoA={d_x,d_y,d_z};

    miKernel2<<<(N+256-1)/256,256>>>(SoA,N);		
    //miKernel2<<<(N+256-1)/256,256>>>(d_x,d_y,d_z,N);		
    	
    std::cout << "Escaleo SoA " << 1e3 * t.elapsed() << " milliseconds" << std::endl;
  }

  return 0;
}

