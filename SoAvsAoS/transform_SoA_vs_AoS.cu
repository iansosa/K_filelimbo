#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/random.h>
#include <assert.h>

#include "timer.h"

// Este ejemplo compara un transform de un pares key,value
// ordenados en AoS y en SoA

struct MyStruct
{
  int key;
  float value;
};

void initialize_keys(thrust::device_vector<int>& keys)
{
  thrust::default_random_engine rng;
  thrust::uniform_int_distribution<int> dist(0, 2147483647);

  thrust::host_vector<int> h_keys(keys.size());

  for(size_t i = 0; i < h_keys.size(); i++)
    h_keys[i] = dist(rng);

  keys = h_keys;
}


void initialize_keys(thrust::device_vector<MyStruct>& structures)
{
  thrust::default_random_engine rng;
  thrust::uniform_int_distribution<int> dist(0, 2147483647);

  thrust::host_vector<MyStruct> h_structures(structures.size());

  for(size_t i = 0; i < h_structures.size(); i++)
    h_structures[i].key = dist(rng);

  structures = h_structures;
}

struct mifunctor{
	__device__ float operator()(int key, float value){
		int k=key;
		float v=value;
		return sin(k*v)*sin(k*v)+cos(k*v)*cos(k*v);
	}
	__device__ MyStruct operator()(MyStruct s){
		int k=s.key;
		float v=s.value;
		MyStruct sout;
		sout.value=sin(k*v)*sin(k*v)+cos(k*v)*cos(k*v);
		sout.key=k;
		return sout;
	}
};


int main(void)
{
  size_t N = 2 * 1024 * 1024;

  // Sort Key-Value pairs using Array of Structures (AoS) storage 
  {
    thrust::device_vector<MyStruct> structures(N);

    initialize_keys(structures);

    timer t;

    thrust::transform(structures.begin(), structures.end(), structures.begin(), mifunctor());

    std::cout << "AoS sort took " << 1e3 * t.elapsed() << " milliseconds" << std::endl;
  }

  // Sort Key-Value pairs using Structure of Arrays (SoA) storage 
  {
    thrust::device_vector<int>   keys(N);
    thrust::device_vector<float> values(N);

    initialize_keys(keys);

    timer t;
	
    thrust::transform(keys.begin(), keys.end(), values.begin(),values.begin(),mifunctor());

    std::cout << "SoA sort took " << 1e3 * t.elapsed() << " milliseconds" << std::endl;
  }

  return 0;
}

