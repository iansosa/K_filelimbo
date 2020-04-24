APPS=convolucion

all: ${APPS}

%: %.cu
	nvcc -O2 -arch=sm_20 -o $@ $<
clean:
	rm -f ${APPS}

run:	all
	./convolucion

submit:	all
	qsub jobGPU; watch qstat
