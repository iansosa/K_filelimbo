#include<thrust/reduce.h>
#include<thrust/device_vector.h>
#include<thrust/host_vector.h>
#include <thrust/system/omp/execution_policy.h>
#include<thrust/functional.h>

#include<cstdio>
#include "simple_timer.h"

#define BIGNUMBER 1000000000

int main()
{

	thrust::minimum<float> mn;
	thrust::host_vector<float> hx( 50000000 );
	thrust::generate(hx.begin(),hx.end(),rand);
	{
		gpu_timer crono, cronosincopia;
		crono.tic();	
	
		thrust::device_vector<float> dx=hx;	
	
		cronosincopia.tic();
		float suma= thrust::reduce(dx.begin(),dx.end(), BIGNUMBER,mn);

		float ms=crono.tac();
		float mssincopia=cronosincopia.tac();

		printf("suma(gpu)=%f, en %f ms (%f ms sin copia) \n", suma, ms, mssincopia);
	}
	{
		omp_set_dynamic(0);     // Explicitly disable dynamic teams
		for(int i=1;i<=4;i++){
			omp_set_num_threads(i); // Use 4 threads for all consecutive parallel regions
			omp_timer crono;
			crono.tic();	
			float suma = thrust::reduce(thrust::omp::par, hx.begin(), hx.end(),BIGNUMBER,mn);
			float ms=crono.tac();

			printf("suma(omp %d)=%f, en %f ms \n", i, suma, ms);
		}
	}


	return 0;
}
