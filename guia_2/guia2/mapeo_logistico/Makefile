APPS=logistic_template

all: ${APPS}

%: %.cu
	nvcc -O2 -arch=sm_20 -o $@ $<
clean:
	rm -f ${APPS}

run:	all
	./logistic_template

submit:	all
	qsub jobGPU; watch qstat
