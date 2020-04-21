#include<thrust/device_vector.h>
#include<thrust/host_vector.h>
#include<thrust/transform_reduce.h>
#include<thrust/sequence.h>
#include <thrust/functional.h>
#include<cstdlib>
#include<cstdio>

#include "simple_timer.h"

using namespace thrust::placeholders;

int main(int argc, char **argv)
{

	// segun el device backend elegido, imprimiremos distintas cosas
	#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
	int dev; cudaGetDevice(&dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
	gpu_timer crono;

	#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_OMP
	printf("OMP version, with %d threads\n", omp_get_max_threads());
	omp_timer crono;

	#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CPP
	printf("CPP version\n");
	cpu_timer crono;
	#endif

	unsigned long long N;
	if(argc==2) N=atoi(argv[1]); else N=10000000;
 
	// generamos un vector de numeros aleatorios secuencialmente en el HOST
	thrust::host_vector<float> hx(N);
	thrust::generate(hx.begin(),hx.end(),rand);
	thrust::transform(hx.begin(),hx.end(),hx.begin(),_1/RAND_MAX-0.5);
	thrust::device_vector<float> dx=hx;

	// calculamos el maximo de los elementos del array al cuadrado...
	crono.tic();
	float suma = thrust::transform_reduce(dx.begin(),dx.end(),_1*_1,-1.0,thrust::maximum<float>());

	double ms = crono.tac();

	printf("maximo de los %u numeros =%f, ms=%f\n",int(N), suma, ms);	

	return 0;
}
