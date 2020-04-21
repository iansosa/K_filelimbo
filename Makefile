APP?=backends

OMP_FLAGS=-fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -lgomp -I/usr/local/cuda/include #-I/share/apps/icnpg/common/thrust-master/
CPP_FLAGS=-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP -I/usr/local/cuda/include #-I/share/apps/icnpg/common/thrust-master/

all: ${APP}

${APP}: $(APP).cu
	nvcc -O2 -o $@ $< -Wno-deprecated-gpu-targets	

cuda_${APP}: $(APP).cu
	nvcc -O2 -o $@ $< -Wno-deprecated-gpu-targets

omp_${APP}: $(APP).cu
	cp $(APP).cu $(APP).cpp
	g++ -O2 $(APP).cpp $(OMP_FLAGS) -o $@
	rm $(APP).cpp

cpp_${APP}: $(APP).cu
	cp $(APP).cu $(APP).cpp
	g++ -O2 $(APP).cpp $(CPP_FLAGS) -o $@
	rm $(APP).cpp

clean:
	rm -f cuda_* omp_* cpp_* *.dat $(APP)

cuda_run: cuda_$(APP)	
	./cuda_$(APP)

omp_run: omp_$(APP)	
	./omp_$(APP)

cpp_run: cpp_$(APP)	
	./cpp_$(APP)

run: $(APP)	
	./$(APP)

cuda_submit: cuda_$(APP)	
	qsub -N cuda_$(APP) jobGPU ./cuda_$(APP); watch qstat

omp_submit: omp_$(APP)
	qsub -N omp_$(APP) jobGPU ./omp_$(APP); watch qstat

cpp_submit: $cpp_(APP)	
	qsub -N cpp_$(APP) jobGPU ./cpp_$(APP) ; watch qstat

submit:	$(APP)
	qsub -N $(APP) jobGPU ./$(APP) 
	#watch qstat

