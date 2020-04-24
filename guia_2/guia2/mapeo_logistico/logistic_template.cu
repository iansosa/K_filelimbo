/*
Itera el mapeo logistico y calcula los exponentes de liapunov en funcion del parametro r.

Entrada: 
las condiciones iniciales y los valores de r que uno quiere calcular, 
el numero de iteraciones, el numero de particulas.

Salida: 
las posiciones finales, y sus exponentes de liapunov en funcion de r.

TODO:
completar el codigo para que corra en GPU, recompilando con

nvcc -DGPUTEST ..... 

Medir la aceleracion usando GPU en funcion del numero de iteraciones 
y de condiciones iniciales o parametros r.

Explorar y Comparar resultados con:

https://en.wikipedia.org/wiki/Logistic_map

*/

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include "simple_timer.h"

struct logistic_variable{
	float x;
	float lambda;
};


struct logistic_lyapunov_combined
{
  const unsigned int trun;

  logistic_lyapunov_combined(unsigned int _trun) : trun(_trun) {}

  __host__ __device__
  logistic_variable operator()(float r, float _x) const
  { 
  	float x=_x;
  	float acum=0.0;
  	for(int i=0;i<trun;i++){
  		acum+=logf( fabsf(r-2.0*r*x) );
  		x=r*x*(1.0-x);
  	}
  	logistic_variable res;
  	res.x=x; res.lambda=acum/trun;
  	return res;	
  }

};

struct mi_operacion
{
    __device__
    float operator()(float a,float b)
    {
      return logf( fabsf(b-2.0*b*a) );
    }
};
// este test calcula los exponentes de lyapunov del mapeo logistico para un rango de r
void test(int argc, char **argv)
{
	unsigned int N;
	unsigned int trun;
	float rmin;
	float rmax;

	if(argc!=5)
	{	
		std::cout << "uso: " << argv[0] << " N trun rmin rmax" << std::endl;	
		exit(1);
	}	
	else{
		N= atoi(argv[1]);
		trun= atoi(argv[2]);
		rmin= atof(argv[3]);
		rmax= atof(argv[4]);
		std::cout << "#parametros: N=" << N << ", trun=" << trun << ", rmin=" << rmin << ", rmax=" << rmax << std::endl;

		if(rmax>4 || rmin<0){
			std::cout << "ojo: [rmin,rmax] debe estar contenido en [0:4]" << std::endl;
			exit(1);	
		}
	}


    using namespace thrust::placeholders;

    // vector de host r distribuido entre [rmin:rmax]. (Chequear que rmin>0, rmax<4)
    thrust::host_vector<float> h_r(N);
    thrust::sequence(h_r.begin(), h_r.end());
    thrust::transform(h_r.begin(), h_r.end(),h_r.begin(),rmin+_1*(rmax-rmin)/N);

    // vector de variables dinamicas x, aleatorio distribuido uniformen en [0,1]
    thrust::host_vector<float> h_x(N);
    std::generate(h_x.begin(), h_x.end(), rand);
    thrust::transform(h_x.begin(), h_x.end(),h_x.begin(),_1/RAND_MAX);

    // vector h_resultado para guardar las variables dinamicas y los exponentes de liapunov
    logistic_variable zero; zero.x=zero.lambda=0.0;
    thrust::host_vector<logistic_variable> h_resultado(N,zero);

  #ifndef GPUTEST
    cpu_timer crono;
    crono.tic();

    logistic_lyapunov_combined operacion(trun);
    for(int i=0;i<N;i++){
	h_resultado[i]=operacion(h_r[i],h_x[i]);	
    }	

    std::cout << "cpu#ms =" << crono.tac();
    std::cout << "\n\n";
//imprime los resultados
  for(int i=0;i<N;i++){
      std::cout << h_r[i] << " " << h_resultado[i].x << " " << h_resultado[i].lambda << std::endl;  
    }
    std::cout << "\n\n";
  #else
    gpu_timer crono;
    crono.tic();

    // TODO:
    // COMPLETE AQUI CON SU CODIGO PARA GPU
    // INCLUYENDO LAS COPIAS H->D, D->H PERTINENTES.
    // USE EL FUNCTOR logistic_lyapunov_combined PROVISTO	
    
    thrust::device_vector<float> d_x = h_x;
    thrust::device_vector<float> d_r = h_r;
    thrust::host_vector<float> h_l(N,0);
    thrust::device_vector<float> d_l = h_l;
    thrust::device_vector<float> intermediate(N);
    for (int i = 0; i < trun; ++i)
    {
      thrust::transform(d_x.begin(), d_x.end(),d_r.begin(),d_x.begin(),_2*_1*(1.0-_1));
      thrust::transform(d_x.begin(), d_x.end(),d_r.begin(),intermediate.begin(),mi_operacion());
      thrust::transform(d_l.begin(), d_l.end(),intermediate.begin(),d_l.begin(),_1+_2);
    }
    thrust::copy(d_x.begin(), d_x.end(), h_x.begin());
    thrust::copy(d_l.begin(), d_l.end(), h_l.begin());
    std::cout << "gpu#ms =" << crono.tac();
    std::cout << "\n\n";
    
    for(int i=0;i<N;i++){
      std::cout << h_r[i] << " " << h_x[i] << " " << h_l[i] << std::endl;  
    }
    std::cout << "\n\n";
  #endif
    // imprime los resultados
/*   for(int i=0;i<N;i++){
    	std::cout << h_r[i] << " " << h_resultado[i].x << " " << h_resultado[i].lambda << std::endl;	
    }
    std::cout << "\n\n";
    /*
	Si imprime los resultados en un fichero "zzz", por ejemplo para 
	10000 particulas en 10000 pasos de tiempo con r en [3.5,4.0] 

	./a.out 10000 10000 3.5 4.0 > zzz

	puede visualizar los resultados en gnuplot (llevese zzz a su computadora) asi:
	gnuplot> set palette model RGB defined ( 0 'red', 1 'green' ); plot 'zzz' u 1:2:(($3>0)?(0):(1)) palette pt 1

	o bien correr en el server

	gnuplot> set term png; set out "dibujo.png"; set palette model RGB defined ( 0 'red', 1 'green' ); 
	plot 'zzz' u 1:2:(($3>0)?(0):(1)) palette pt 1
	
	y llevarse dibujo.png para visualizarlo localmente en su maquina.

	Vera asi el diagrama de bifurcaciones, y los puntos seran pintados de rojo o verde segun el signo de su exponente de liapunov 

	Sino les gusta gnuplot, pueden hacerlo en python, octave, matlab, mathematica, o lo que prefieran.

	Nuevas, creativas, y bellas formas de visualizacion de la solucion son bienvenidas! Compartirlas por favor.
    */
}


int main(int argc, char **argv){
	test(argc,argv);
}

