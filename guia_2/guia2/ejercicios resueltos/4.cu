#include <iostream>
#include <stdlib.h>
#include <ctime>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/zip_iterator.h>
#include <boost/random.hpp>

struct euler
{
  double dt;
  double r;
  double s;
  double b;
  euler(double _dt,double _r,double _s,double _b) : dt(_dt), r(_r), s(_s), b(_b) {}
  __device__ thrust::tuple<float,float,float> operator()(thrust::tuple<float, float, float> t) //difsq
  {
    float dx=s*(thrust::get<1>(t)-thrust::get<0>(t));
    float dy=r*thrust::get<0>(t)-thrust::get<1>(t)-thrust::get<0>(t)*thrust::get<2>(t);
    float dz=thrust::get<0>(t)*thrust::get<1>(t)-b*thrust::get<2>(t);
    thrust::get<0>(t)=thrust::get<0>(t)+dt*dx;
    thrust::get<1>(t)=thrust::get<1>(t)+dt*dy;
    thrust::get<2>(t)=thrust::get<2>(t)+dt*dz;
    return t;
  }
};

int main()
{
  boost::mt19937 rng(static_cast<unsigned int>(std::time(0)));  /// el engine para generar numeros random
  boost::uniform_real<> unif( -1, 1);//la distribucion de probabilidad uniforme entre cero y 2pi
  boost::variate_generator< boost::mt19937&, boost::uniform_real<> > gen( rng , unif );//gen es una funcion que toma el engine y la distribucion y devuelve el numero random

  using namespace thrust::placeholders;
  int N;
  printf("N: ");
  std::cin >>N;

  int steps;
  printf("steps: ");
  std::cin >>steps;

  double dt=0.01;
  double r=28.0;
  double s=10.0;
  double b=8.0/3.0;

  thrust::host_vector<float> h_x(N);
  thrust::host_vector<float> h_y(N);
  thrust::host_vector<float> h_z(N);
  for (int i = 0; i < N; ++i)
  {
    h_x[i]=gen();
    h_y[i]=gen();
    h_z[i]=gen();
  }

  thrust::device_vector<float> x=h_x;
  thrust::device_vector<float> y=h_y;
  thrust::device_vector<float> z=h_z;

  auto begin = thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin(),z.begin()));
  auto end = thrust::make_zip_iterator(thrust::make_tuple(x.end(), y.end(),z.end()));
  for (int i = 0; i < steps; ++i)
  {
    thrust::transform(begin, end,begin, euler(dt,r,s,b));
  }


  thrust::copy(x.begin(), x.end(), h_x.begin());
  thrust::copy(y.begin(), y.end(), h_y.begin());
  thrust::copy(z.begin(), z.end(), h_z.begin());

  FILE *f=fopen("4.txt", "w");
  for (int i = 0; i < N; ++i)
  {
    fprintf(f,"%lf   %lf   %lf\n",h_x[i],h_y[i],h_z[i]);
  }

  std::cout << "done" << std::endl;
  std::cout << thrust::get<1>(begin[1]) << std::endl;
 // printf("%lf\n",  thrust::get<0>(begin[1]));
  return 0;
}