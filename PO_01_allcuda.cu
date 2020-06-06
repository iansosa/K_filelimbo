#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <string>
#include <cassert>
#include <vector>
#include <iostream>
#include <armadillo>
#include <utility>
#include <omp.h>
#include "gpu_timer.h"
#include <boost/numeric/odeint.hpp>


#include <boost/numeric/odeint/stepper/runge_kutta4.hpp>
#include <boost/numeric/odeint/integrate/integrate_const.hpp>
#include <boost/numeric/odeint/external/thrust/thrust.hpp>
#include <boost/random.hpp>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>

using namespace std;

using namespace boost::numeric::odeint;
using namespace thrust::placeholders;
typedef float value_type;


typedef thrust::device_vector< value_type > state_type;
typedef thrust::device_vector< size_t > index_vector_type;


struct returngamma
{
    const value_type *m_xvec;
    int m_N;
    int m_time;
    returngamma(const value_type *xvec,const int N,const int time)
    : m_xvec(xvec), m_N(N), m_time(time) { }

      __device__
      value_type operator()(int i)
      {
          value_type aux=m_xvec[2*m_N*m_time+2*i+1];
          return aux*aux;
      }
};

struct returnJ0
{
    const value_type *m_xvec;
    const value_type *m_A;
    int m_N;
    int m_time;
    returnJ0(const value_type *A,const value_type *xvec,const int N,const int time)
    : m_A(A), m_xvec(xvec), m_N(N), m_time(time) { }

      __device__
      value_type operator()(int i)
      {
          return m_xvec[2*m_N*m_time+2*i+1]*m_A[i*m_N]*sin(m_xvec[2*m_N*m_time+0]- m_xvec[2*m_N*m_time+2*i])/m_N;
      }
};


struct mean_force_calculator
{
    struct getkey : public thrust::unary_function< int , value_type >
    {
        const value_type *m_A;
        const value_type *m_xvec;
        int m_N;
        getkey(const value_type *A,const value_type *xvec,const int N)
        : m_A(A), m_xvec(xvec), m_N(N) { }

        __host__ __device__
        value_type operator()(int i) const
        {
            int m_first_i=i/m_N;
            int m_second_i=i%m_N;
            return m_A[i]*sin(m_xvec[2*m_second_i]- m_xvec[2*m_first_i])/m_N;
        }
    };

    struct getkey_e : public thrust::unary_function< int , value_type >
    {
        const value_type *m_A;
        const value_type *m_xvec;
        int m_N;
        int m_time;
        getkey_e(const value_type *A,const value_type *xvec,const int N,int time)
        : m_A(A), m_xvec(xvec), m_N(N),m_time(time) { }

        __host__ __device__
        value_type operator()(int i) const
        {
            int m_first_i=i/m_N;
            int m_second_i=i%m_N;
            if(m_second_i==0)
            {
                return 0;
            }
            return m_A[i]*sin(m_xvec[2*m_N*m_time+2*m_second_i]- m_xvec[2*m_N*m_time+2*m_first_i])*m_xvec[2*m_N*m_time+2*m_first_i+1]/m_N;
        }
    };
    static state_type get_mean_force( const state_type &x, const state_type &A,const int N)
    {
    	state_type ret(N);
        thrust::reduce_by_key(thrust::make_transform_iterator(thrust::counting_iterator<int>(0), _1/N), thrust::make_transform_iterator(thrust::counting_iterator<int>(N*N), _1/N), thrust::make_transform_iterator(thrust::counting_iterator<int>(0), getkey(thrust::raw_pointer_cast(A.data()),thrust::raw_pointer_cast(x.data()),N)), thrust::make_discard_iterator(), ret.begin());
        return ret;
    }

    static state_type get_energy_transfer( const state_type &x, const state_type &A,const int N,int time)
    {
        state_type ret(N);
        thrust::reduce_by_key(thrust::make_transform_iterator(thrust::counting_iterator<int>(0), _1/N), thrust::make_transform_iterator(thrust::counting_iterator<int>(N*N), _1/N), thrust::make_transform_iterator(thrust::counting_iterator<int>(0), getkey_e(thrust::raw_pointer_cast(A.data()),thrust::raw_pointer_cast(x.data()),N,time)), thrust::make_discard_iterator(), ret.begin());
        return ret;
    }

};


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class phase_oscillators
{

public:

    struct sys_functor
    {
        const value_type *m_x;
        const value_type m_t;
        const value_type *mm_G;
        const value_type *mm_I;
        const value_type *mm_F;
        const value_type *mm_Fw;
        value_type *m_magic;

        sys_functor(const value_type *x ,const  value_type t,const value_type *m_G,const value_type *m_I,const value_type *m_F,const value_type *m_Fw, value_type *magic)
        : m_x( x ),m_t(t),mm_G(m_G),mm_I(m_I),mm_F(m_F),mm_Fw(m_Fw),m_magic(magic) { }

        template< class Tuple >
        __host__ __device__
        void operator()( Tuple t )
        {
            if(thrust::get<1>(t)%2==0)
            {
                thrust::get<2>(t)= m_x[thrust::get<1>(t)+1];
            }
            else
            {
                int position=(thrust::get<1>(t)-1)/2;
                thrust::get<2>(t) = m_magic[position]/mm_I[position]+mm_F[position]*sin(mm_Fw[position]*m_t-m_x[2*position])/mm_I[position]-(mm_G[position]/mm_I[position])*m_x[thrust::get<1>(t)];
            }
        }
    };

    phase_oscillators( int N,const state_type &A,const value_type *G,const value_type *I,const value_type *Fw,const value_type *F)
        : m_A( A ) ,m_G( G ) ,m_I( I ) ,m_F( F ) ,m_Fw( Fw ) , m_N( N )
        {

        }

    void operator() ( const state_type &x , state_type &dxdt , const value_type t )
    {
        state_type ret=mean_force_calculator::get_mean_force(x,m_A,m_N);
        if(ceilf(t)==t && (int) t%10==0)
        {
            printf("tiempo: %lf\n",t);
        }
        thrust::counting_iterator<int> it1(0);
        thrust::counting_iterator<int> it2 = it1 + 2*m_N;
        thrust::for_each(
                thrust::make_zip_iterator( thrust::make_tuple(x.begin(), it1, dxdt.begin() ) ),
                thrust::make_zip_iterator( thrust::make_tuple(x.end(), it2, dxdt.end()   ) ) ,
                sys_functor(thrust::raw_pointer_cast(x.data()),t,m_G,m_I,m_F,m_Fw,thrust::raw_pointer_cast(ret.data()))
                );
    }

private:

    const state_type &m_A;
    const value_type *m_G;
    const value_type *m_I;
    const value_type *m_F;
    const value_type *m_Fw;
    const size_t m_N;
};

struct push_back_state_and_time
{
    state_type &m_states;
    std::vector< value_type >& m_times;
    int &m_current_it;

    int m_N;

    push_back_state_and_time( state_type &states , std::vector< value_type > &times, int N, int &current_it ) : m_states( states ) , m_times( times ), m_N(N), m_current_it(current_it) { }

    __host__ 
    void operator()( const state_type &x , value_type t )
    {
        const value_type *input = thrust::raw_pointer_cast(x.data());
        thrust::copy(x.data(),x.data()+2*m_N,m_states.data()+m_current_it*2*m_N);
        m_times.push_back( t );
        m_current_it=m_current_it+1;
    }
};



void inicialcond(state_type &d_x,int N,boost::mt19937 &rng,int load)
{
    boost::uniform_real<> unif( 0, 0.01);
    boost::variate_generator< boost::mt19937&, boost::uniform_real<> > gen( rng , unif );
   
    thrust::host_vector<value_type> h_x(2*N);
    if(load==0)
    {
        for (int i = 0; i < N; ++i)
        {
            h_x[2*i]=0;
            h_x[2*i+1]=0;
        }
        FILE *w= fopen("Xi.txt", "w");
        for (int i = 0; i < N; ++i)
        {
            fprintf(w, "%f  ",h_x[2*i]);
            fprintf(w, "%f  ",h_x[2*i+1]);
        }
        fclose(w);
    }
    else
    {
        FILE *r= fopen("Xi.txt", "r");
        for (int i = 0; i < N; ++i)
        {
            fscanf(r, "%f", &h_x[2*i]); 
            fscanf(r, "%f", &h_x[2*i+1]); 
        }
        fclose(r);
    }
    d_x=h_x;
}

void fillA(state_type &d_A,int N,boost::mt19937 &rng,int load)
{

    boost::uniform_real<> unif( 0, 1 );
    boost::variate_generator< boost::mt19937&, boost::uniform_real<> > gen( rng , unif );
    thrust::host_vector<value_type> h_A(N*N);
    if(load==0)
    {
        for (int i = 0; i < N; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                h_A[i+N*j]=1.0;
            }
        }
        FILE *w= fopen("Ai.txt", "w");
        for (int i = 0; i < N; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                fprintf(w, "%f  ",h_A[i+N*j]);
            }
            fprintf(w, "\n");     
        }
        fclose(w);
    }
    else
    {
        FILE *r= fopen("Ai.txt", "r");
        for (int i = 0; i < N; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                fscanf(r, "%f", &h_A[i+N*j]);
            }
        }
        fclose(r);
    }
    d_A=h_A;
}

void fillG(state_type &d_G,int N,boost::mt19937 &rng,int load)
{

    boost::normal_distribution<> unif(2.5, 0.2 );
    boost::variate_generator< boost::mt19937&, boost::normal_distribution<> > gen( rng , unif );
    thrust::host_vector<value_type> h_G(N);
    
    if(load==0)
    {
        for (int i = 0; i < N; ++i)
        {
            //h_G[i]=2.5;
            h_G[i]=gen();
        }
        FILE *w= fopen("Gi.txt", "w");
        for (int i = 0; i < N; ++i)
        {
            fprintf(w, "%f  ",h_G[i]);
        }
        fclose(w);
    }
    else
    {
        FILE *r= fopen("Gi.txt", "r");
        for (int i = 0; i < N; ++i)
        {
            fscanf(r, "%f", &h_G[i]);
        }
        fclose(r);
    }
    d_G=h_G;
}

void fillI(state_type &d_I,int N,boost::mt19937 &rng,int load)
{
    thrust::host_vector<value_type> h_I(N);
    if(load==0)
    {
        for (int i = 0; i < N; ++i)
        {
            h_I[i]=1.0;
        }
        FILE *w= fopen("Ii.txt", "w");
        for (int i = 0; i < N; ++i)
        {
            fprintf(w, "%f  ",h_I[i]);
        }
        fclose(w);
    }
    else
    {
        FILE *r= fopen("Ii.txt", "r");
        for (int i = 0; i < N; ++i)
        {
            fscanf(r, "%f", &h_I[i]);
        }
        fclose(r);
    }
    d_I=h_I;
}

void fillW(state_type &d_w,int N,boost::mt19937 &rng,int load)
{
    boost::uniform_real<> unif( 0, 10 );
    boost::variate_generator< boost::mt19937&, boost::uniform_real<> > gen( rng , unif );
    thrust::host_vector<value_type> h_w(N);
    if(load==0)
    {
        for (int i = 0; i < N; ++i)
        {   
            h_w[i]=1.0;
        }
        FILE *w= fopen("Wi.txt", "w");
        for (int i = 0; i < N; ++i)
        {
            fprintf(w, "%f  ",h_w[i]);
        }
        fclose(w);
    }
    else
    {
        FILE *r= fopen("Wi.txt", "r");
        for (int i = 0; i < N; ++i)
        {
            fscanf(r, "%f", &h_w[i]);
        }
        fclose(r);
    }
    d_w=h_w;
}

void fillFw(state_type &d_Fw,int N,boost::mt19937 &rng,int load)
{
    boost::uniform_real<> unif( 0, 10 );
    boost::variate_generator< boost::mt19937&, boost::uniform_real<> > gen( rng , unif );
    thrust::host_vector<value_type> h_Fw(N);
    if(load==0)
    {
        h_Fw[0]=100;
        for (int i = 1; i < N; ++i)
        {
            h_Fw[i]=0;
        }
        FILE *w= fopen("Fwi.txt", "w");
        for (int i = 0; i < N; ++i)
        {
            fprintf(w, "%f  ",h_Fw[i]);
        }
        fclose(w);
    }
    else
    {
        FILE *r= fopen("Fwi.txt", "r");
        for (int i = 0; i < N; ++i)
        {
            fscanf(r, "%f", &h_Fw[i]);
        }
        fclose(r);
    }
    d_Fw=h_Fw;

}

void printsave(size_t steps, thrust::host_vector<value_type> &x_vec,std::vector<value_type> &times,int N,int i)
{

	FILE *f1;
    if(i==0)
    {
        f1=fopen("save.txt","w");
    }
    else
    {
        f1=fopen("save.txt","a");
    }
    
	for( size_t i=0; i<steps; ++i )
	{
		if(((int)(100.0*i/steps))%10==0 && i < steps)
		{
			printf("printing savestate: %d \n", (int)(100.0*i/steps));
		}
		fprintf(f1,"%f  ",times[i] );
		for (int j = 0; j < N; ++j)
		{
			fprintf(f1,"%f	  %f  ",x_vec[2*N*i+2*j],x_vec[2*N*i+2*j+1]); 
		}
		fprintf(f1,"\n");
	}
	fclose(f1);
}

int number_of_loops(value_type Total_time, int N, value_type dt)
{
    int GB_inMemory=5;

    size_t total_bytes=(size_t)((sizeof(value_type)*Total_time*2*N/dt));
    printf("Total_GB_tiempo: %lf\n",(value_type)total_bytes/(1024*1024*1024));
    printf("Total_GB_head: %lf\n",(value_type)sizeof(value_type)*(12*N+N*N)/(1024*1024*1024));
    if(GB_inMemory<(value_type)sizeof(value_type)*(12*N+N*N)/(1024*1024*1024))
    {
        printf("Error: Not enough VRAM\n");
        return -1;
    }
    return(1+(int)(total_bytes/(GB_inMemory*pow(1024,3)-sizeof(value_type)*(12*N+N*N))));
}

void integrate(int N,state_type &x_vec,value_type dt,state_type &d_x,phase_oscillators &sys,value_type start_time,value_type end_time,thrust::host_vector<value_type> &x_vec_host,int i,int save)
{
    runge_kutta4< state_type , value_type , state_type , value_type > stepper;
    size_t steps;
    if(save==1)
    {
        int current_it=0;
        std::vector<value_type> times;
        steps=integrate_adaptive( stepper , sys , d_x , start_time , end_time , dt,push_back_state_and_time( x_vec , times,N,current_it ) );
        thrust::copy(x_vec.begin(), x_vec.end(), x_vec_host.begin());
        printsave(steps,x_vec_host,times,N,i);
    }
    else
    {
        steps=integrate_adaptive( stepper , sys , d_x , start_time , end_time , dt);
    }
}


void printproperties(int N,thrust::host_vector<value_type> &G,thrust::host_vector<value_type> &h_J_gamma,thrust::host_vector<value_type> &h_J_gomega,thrust::host_vector<value_type> &h_J_0)
{
    FILE *f=fopen("W.txt","w");
    for (int j = 1; j < N; ++j)
    { 
        if(((int)(100.0*j/N))%10==0)
        {
            printf("calculating properties: %d \n", (int)(100.0*j/(N)));
        }
        fprintf(f,"%.10lf   %.10lf   %.10lf   %.10lf \n",G[j], G[j]*h_J_gamma[j],h_J_gomega[j],h_J_0[j]); 
    }

    fclose(f);
}

void calcproperties(int N,int steps_estim,state_type &A,state_type &x,thrust::host_vector<value_type> &G,int T_properties)
{
    state_type d_J_gamma(N);
    thrust::fill(d_J_gamma.begin(), d_J_gamma.end(), 0);
    state_type d_J_omega(N);
    state_type d_J_aux(N);
    thrust::fill(d_J_omega.begin(), d_J_omega.end(), 0);
    state_type d_J_0(N);
    thrust::fill(d_J_0.begin(), d_J_0.end(), 0);


    for (int i = T_properties; i < steps_estim; ++i)
    {
        if(((int)(100.0*(i-T_properties)/(steps_estim-T_properties)))%10==0)
        {
            printf("calculating properties: %d \n", (int)(100.0*(i-T_properties)/(steps_estim-T_properties)));
        }
        d_J_aux=mean_force_calculator::get_energy_transfer(x,A,N,i);
        thrust::transform(d_J_omega.begin(),d_J_omega.end(),d_J_aux.begin(),d_J_omega.begin(),_1+_2);
        thrust::transform(thrust::counting_iterator<int>(0),thrust::counting_iterator<int>(N),d_J_aux.begin(),returngamma(thrust::raw_pointer_cast(x.data()),N,i));
        thrust::transform(d_J_gamma.begin(),d_J_gamma.end(),d_J_aux.begin(),d_J_gamma.begin(),_1+_2);
        thrust::transform(thrust::counting_iterator<int>(0),thrust::counting_iterator<int>(N),d_J_aux.begin(),returnJ0(thrust::raw_pointer_cast(A.data()),thrust::raw_pointer_cast(x.data()),N,i));
        thrust::transform(d_J_0.begin(),d_J_0.end(),d_J_aux.begin(),d_J_0.begin(),_1+_2);
    }
    thrust::transform(d_J_gamma.begin(),d_J_gamma.end(),d_J_gamma.begin(),_1/(steps_estim-T_properties));
    thrust::host_vector<value_type> h_J_gamma=d_J_gamma;
    thrust::transform(d_J_omega.begin(),d_J_omega.end(),d_J_omega.begin(),_1/(steps_estim-T_properties));
    thrust::host_vector<value_type> h_J_omega=d_J_omega;
    thrust::transform(d_J_0.begin(),d_J_0.end(),d_J_0.begin(),_1/(steps_estim-T_properties));
    thrust::host_vector<value_type> h_J_0=d_J_0;
    printproperties(N,G,h_J_gamma,h_J_omega,h_J_0);
}

int main()
{
    boost::mt19937 rng(static_cast<unsigned int>(std::time(0)));  /// el engine para generar numeros random

///////////////////////////////////////////////////////////////////////
    int N;
    printf("N: ");
    std::cin >>N;
      
    int load;
    printf("Load IC (0 NO, 1 YES): ");
    std::cin >>load;
    
    value_type Total_time;
    printf("Total_time : ");
    std::cin >>Total_time;

    value_type Saved_time;
    printf("Saved_time : ");
    std::cin >>Saved_time;

    if(Total_time<Saved_time)
    {
        printf("Error: total_time<Saved_time\n");
        return 0;
    }

    value_type T_properties;
    printf("T_properties : ");
    std::cin >>T_properties;

    value_type dt;
    printf("dt : ");
    std::cin >>dt;

	thrust::host_vector<value_type> x(2*N); //condiciones iniciales

    int loops=number_of_loops(Saved_time,N,dt);
    if(loops==-1)
    {
        return 0;
    }
    printf("Loops: %d\n",loops );



//////////////////////////////////////////////////////////////////////////
    gpu_timer reloj;
    reloj.tic();

	state_type d_A(N*N);
    fillA(d_A,N,rng,load);
	state_type d_G(N);
    fillG(d_G,N,rng,load);
	state_type d_I(N);
    fillI(d_I,N,rng,load);
	state_type d_F(N);
    fillFw(d_F,N,rng,load);
	state_type d_Fw(N);
    fillW(d_Fw,N,rng,load);

	state_type d_x(2*N);
    inicialcond(d_x,N,rng,load); ///
////////////////////////////////////////////////////////////////////////////

    int steps_estim=(int)((Saved_time)/(dt*loops));
    if((int)(T_properties/dt)>steps_estim)
    {
        printf("Error: T_properties/dt>steps_estim\n");
    }
    state_type x_vec(2*N*steps_estim+1);
    thrust::host_vector<value_type> x_vec_host(2*N*steps_estim+1);
    phase_oscillators sys(N,d_A,thrust::raw_pointer_cast(d_G.data()),thrust::raw_pointer_cast(d_I.data()),thrust::raw_pointer_cast(d_Fw.data()),thrust::raw_pointer_cast(d_F.data())/*,d_sum_placeholder*/);
    if(dt<Total_time-Saved_time)
    {
        integrate(N,x_vec,dt,d_x,sys,0,Total_time-Saved_time,x_vec_host,0,0);
    }
    for (int i = 0; i < loops; ++i)
    {
        value_type start_time=Total_time-Saved_time+i*(Saved_time)/loops;
        value_type end_time=Total_time-Saved_time+(i+1)*(Saved_time)/loops;
        integrate(N,x_vec,dt,d_x,sys,start_time,end_time,x_vec_host,i,1);
    }
    thrust::host_vector<value_type> G=d_G;
    calcproperties(N,steps_estim,d_A,x_vec,G,(int)(T_properties/dt));

	printf("N=%d\nTiempo: %lfms\n",N,reloj.tac());

	return 0;
}