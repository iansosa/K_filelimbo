// solucion en la cpu

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "cpu_timer.h"

#define SIZE	1024
#define NVECES	1


void VectorAdd(int *a, int *b, int *c, int n)
{
	for(int i=0;i<n;i++)
	{
		c[i] = a[i] + b[i];
	}
}

int main(int argc, char **argv)
{
	int N;
	
	// argumentos por linea de comandos
	if(argc==2) N=atoi(argv[1]);
	else N=SIZE;

	int *a, *b, *c;

	// alocacion de memoria de host
	a = (int *)malloc(N*sizeof(int));
	b = (int *)malloc(N*sizeof(int));
	c = (int *)malloc(N*sizeof(int));

	// inicializacion
	for( int i = 0; i < N; ++i )
	{
		a[i] = i;
		b[i] = i;
		c[i] = 0;
	}

	// timers
	cpu_timer Reloj;
	Reloj.tic();

	// suma vectores c[i]=a[i]+b[i], i=0,...,N-1
	for(int i=0;i<NVECES;i++)
	VectorAdd(a, b, c, N);

	// imprime milisegundos 
	printf("CPU: N= %d t= %lf ms\n", N, Reloj.tac());	

	// verificacion de resultado
	for( int i = 0; i < 10; ++i){
		//printf("c[%d] = %d\n", i, c[i]);
		assert(c[i]==2*i);
	}

	// liberacion de memoria
	free(a);
	free(b);
	free(c);

	return 0;
}
