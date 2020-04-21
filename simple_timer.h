// varios cronometros para procesos en GPU y en CPU
// USO: cpu_timer T; T.cpu_tic();...calculo...;T.cpu_tac(); cout << T.ms_elapsed << "ms\n";

#pragma once
#ifdef __CUDACC__

///////////////////////////////// GPU TIMER ////////////////////////////////
// use CUDA's high-resolution timers when possible
#include <cuda_runtime_api.h>
//#include <thrust/system/cuda_error.h> //previous thrust releases
#include <thrust/system_error.h>
#include <string>
void cuda_safe_call(cudaError_t error, const std::string& message = "")
{
  if(error)
    throw thrust::system_error(error, thrust::cuda_category(), message);
}

struct gpu_timer
{
  cudaEvent_t start;
  cudaEvent_t end;
  float ms_elapsed;
	
  gpu_timer(void)
  {
    cuda_safe_call(cudaEventCreate(&start));
    cuda_safe_call(cudaEventCreate(&end));
    tic();
  }

  ~gpu_timer(void)
  {
    cuda_safe_call(cudaEventDestroy(start));
    cuda_safe_call(cudaEventDestroy(end));
  }

  void tic(void)
  {
    cuda_safe_call(cudaEventRecord(start, 0));
  }

  double tac(void)
  {
    cuda_safe_call(cudaEventRecord(end, 0));
    cuda_safe_call(cudaEventSynchronize(end));

    cuda_safe_call(cudaEventElapsedTime(&ms_elapsed, start, end));
    return ms_elapsed;
  }

  double epsilon(void)
  {
    return 0.5e-6;
  }
};
#endif

///////////////////////////////// OPENMP TIMER ////////////////////////////////
#include <omp.h>
#include <time.h>
#include <math.h>
#include <stdio.h>
struct omp_timer
{
//  clock_t start;
//  clock_t end;
  double wall_timer_start;
  double wall_timer_end;
  float ms_elapsed;

  omp_timer(void)
  {
    tic();
  }

  ~omp_timer(void)
  {
  }

  void tic(void)
  {    	
//    start = clock();
    wall_timer_start = omp_get_wtime(); 
  }

  double tac(void)
  {

//    end = clock();
    wall_timer_end = omp_get_wtime(); 
//    return static_cast<double>(end - start) / static_cast<double>(CLOCKS_PER_SEC);
    return (ms_elapsed=1e3*static_cast<double>(wall_timer_end - wall_timer_start));
  }

  double epsilon(void)
  {
    return 1.0 / static_cast<double>(CLOCKS_PER_SEC);
  }
};


///////////////////////////////// CPU TIMER ////////////////////////////////
#include <ctime>
struct timespec diff(timespec start, timespec end)
{
        timespec temp;
       	if ((end.tv_nsec-start.tv_nsec)<0) {
                temp.tv_sec = end.tv_sec-start.tv_sec-1;
                temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
        } else {
                temp.tv_sec = end.tv_sec-start.tv_sec;
                temp.tv_nsec = end.tv_nsec-start.tv_nsec;
        }
        return temp;
}

struct cpu_timer{
        struct timespec time1, time2;
	double ms_elapsed;

        cpu_timer(){
        	tic();
        }
       ~cpu_timer(){}

        void tic(){
                clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
        }
        double tac(){
                clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
                return(ms_elapsed=elapsed());
        }
        double elapsed(){
            return (double)diff(time1,time2).tv_sec*1000 + (double)diff(time1,time2).tv_nsec*0.000001;
        }
};

