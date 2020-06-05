#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/random.h>
#include <assert.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/for_each.h>

#include "timer.h"

// Este ejemplo escalea las componentes de un vector de puntos 3d.
// usando AoS y SoA. Usa thrust::for_each 

struct MyStruct
{
  int x;
  int y;
  int z;
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

struct mifunctor{
	__device__ void operator()(thrust::tuple<int &,int &,int &> R){
		thrust::get<0>(R)*=1;
		thrust::get<1>(R)*=2;
		thrust::get<2>(R)*=3;
	}
	__device__ void operator()(MyStruct &s){
		s.x*=1;
		s.y*=2;
		s.z*=3;
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

    thrust::for_each(structures.begin(), structures.end(), mifunctor());

    std::cout << "Escaleo AoS " << 1e3 * t.elapsed() << " milliseconds" << std::endl;
  }

  // Sort Key-Value pairs using Structure of Arrays (SoA) storage 
  {
    thrust::device_vector<int> x(N);
    thrust::device_vector<int> y(N);
    thrust::device_vector<int> z(N);

    initialize_keys(x,y,z);

    timer t;
	
    thrust::for_each(
	thrust::make_zip_iterator( thrust::make_tuple(x.begin(),y.begin(),z.begin()) ),
	thrust::make_zip_iterator( thrust::make_tuple(x.end(),y.end(),z.end()) ),
	mifunctor());
    	
    std::cout << "Escaleo SoA " << 1e3 * t.elapsed() << " milliseconds" << std::endl;
  }


  return 0;
}

