APPS=scalevector3d_AoS_vs_SoA scalevector3d_AoS_vs_SoA_thrust sorting_AoS_vs_SoA transform_SoA_vs_AoS

all: ${APPS}

%: %.cu
	nvcc  -o $@ $<
clean:
	rm -f ${APPS}

run:	all
	./a.out

submit:	all
	qsub jobGPU

submitwatch:	all
	qsub jobGPU; watch qstat
