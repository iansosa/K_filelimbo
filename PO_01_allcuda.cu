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

#include <boost/numeric/odeint.hpp>


#include <boost/numeric/odeint/stepper/runge_kutta4.hpp>
#include <boost/numeric/odeint/integrate/integrate_const.hpp>
#include <boost/numeric/odeint/external/thrust/thrust.hpp>

#include <boost/random.hpp>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/counting_iterator.h>

using namespace std;

using namespace boost::numeric::odeint;

typedef double value_type;


typedef thrust::device_vector< value_type > state_type;
typedef thrust::device_vector< size_t > index_vector_type;

__global__
void calcproperties(double *ecin_aux_d,double *epot_aux_d, double *flux1_aux_d, double *flux_aux_d, double *elost_aux_d, double *x_vec_lin_d, double *I_lin_d, double *A_lin_d, double *G_lin_d, int N, int steps,double K, double *xmed_aux_d)
{
 	int i = blockIdx.x*blockDim.x + threadIdx.x;
	double epot=0;
	double f=0;
	double flux=0;
    double cos_sum = 0.0 , sin_sum = 0.0;


  if (i < steps) 
  	{
		for (int place = 0; place < N; ++place)
		{
			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			ecin_aux_d[place+i*N]=0.5*x_vec_lin_d[1+i*2+steps*2*place]*x_vec_lin_d[1+i*2+steps*2*place]/I_lin_d[place];
			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			epot=0;
			for (int l = 0; l < N; ++l)
			{
				epot=epot-A_lin_d[l+N*place]*cos(x_vec_lin_d[0+i*2+steps*2*l]-x_vec_lin_d[0+i*2+steps*2*place]);
			}
			epot=epot*K/(2*N);
			epot_aux_d[place+i*N]=epot;
			/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			f=0;
			flux=0;
			for (int l = 1; l < N; ++l)
			{
				flux=A_lin_d[l+N*place]*sin(x_vec_lin_d[0+i*2+steps*2*l]-x_vec_lin_d[0+i*2+steps*2*place])*x_vec_lin_d[1+i*2+steps*2*place];		
				f=f+flux;
			}
			f=f*(K/N);
			flux_aux_d[place+i*N]=f;
			flux1_aux_d[place+i*N]=(K/(N))*A_lin_d[0+N*place]*sin(x_vec_lin_d[0+i*2+steps*2*0]-x_vec_lin_d[0+i*2+steps*2*place])*x_vec_lin_d[1+i*2+steps*2*place];
			/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

			elost_aux_d[place+i*N]=-G_lin_d[place]*(x_vec_lin_d[1+i*2+steps*2*place]*x_vec_lin_d[1+i*2+steps*2*place]); //bien

			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		}
    	cos_sum = 0.0 , sin_sum = 0.0;
    	for( size_t l=1 ; l<N ; ++l )
    	{
        	cos_sum += cos( x_vec_lin_d[0+i*2+steps*2*l] );
       	 	sin_sum += sin( x_vec_lin_d[0+i*2+steps*2*l] );
    	}
    	cos_sum /= double( N-1 );
    	sin_sum /= double( N-1 );
		xmed_aux_d[i]=atan2( sin_sum , cos_sum );		
	}
}

struct mean_field_calculator
{
    struct sin_functor : public thrust::unary_function< value_type , value_type >
    {
        __host__ __device__
        value_type operator()( value_type x) const
        {
            return sin( x );
        }
    };

    struct cos_functor : public thrust::unary_function< value_type , value_type >
    {
        __host__ __device__
        value_type operator()( value_type x) const
        {
            return cos( x );
        }
    };

    static std::pair< value_type , value_type > get_mean( const state_type &x )
    {

        value_type sin_sum = thrust::reduce(
                thrust::make_transform_iterator( x.begin() , sin_functor() ) ,
                thrust::make_transform_iterator( x.end() , sin_functor() ) );

        value_type cos_sum = thrust::reduce(
                thrust::make_transform_iterator( x.begin() , cos_functor() ) ,
                thrust::make_transform_iterator( x.end() , cos_functor() ) );

        cos_sum /= value_type( x.size() );
        sin_sum /= value_type( x.size() );

        value_type K = sqrt( cos_sum * cos_sum + sin_sum * sin_sum );
        value_type Theta = atan2( sin_sum , cos_sum );

        return std::make_pair( K , Theta );
    }
};

struct mean_force_calculator
{
    struct mean_force_functor : public thrust::unary_function< value_type , value_type >
    {
    	const value_type *m_A;
    	const value_type *m_xvec;
    	int m_N;
    	mean_force_functor(const value_type *A,const value_type *xvec,const int N) : m_A(A), m_xvec(xvec), m_N(N)
        { }

        __host__ __device__
        value_type operator()(int j) const
        {
        	value_type sum=0;
        	if(j>=m_N)
        	{
        		return sum;
        	}
        	for (int i = 0; i < m_N; ++i)
        	{
        		sum=sum+m_A[j*m_N+i]*sin(m_xvec[2*i]- m_xvec[2*j]);
        	}
            return sum/m_N;
        }
    };


    static state_type get_mean_force( const state_type &x, const state_type &A,const int N)
    {
    	state_type ret(N);

        thrust::transform(thrust::make_counting_iterator(0),thrust::make_counting_iterator(2*N-1),ret.begin(),mean_force_functor(thrust::raw_pointer_cast(A.data()),thrust::raw_pointer_cast(x.data()),N));

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

        sys_functor(const value_type *x ,const  value_type t,const value_type *m_G,const value_type *m_I,const value_type *m_F,const value_type *m_Fw,value_type *magic)
        : m_x( x ),m_t(t),mm_G(m_G),mm_I(m_I),mm_F(m_F),mm_Fw(m_Fw),m_magic(magic) { }

        template< class Tuple >
        __host__ __device__
        void operator()( Tuple t )
        {
            //int current_oscnum;
            if(thrust::get<1>(t)%2==0)
            {
                thrust::get<2>(t)= m_x[thrust::get<1>(t)+1];
            }
            else
            {
                thrust::get<2>(t) = m_magic[(thrust::get<1>(t)-1)/2]/mm_I[(thrust::get<1>(t)-1)/2]+mm_F[(thrust::get<1>(t)-1)/2]*sin(mm_Fw[(thrust::get<1>(t)-1)/2]*m_t-m_x[2*(thrust::get<1>(t)-1)/2])/mm_I[(thrust::get<1>(t)-1)/2]-(mm_G[(thrust::get<1>(t)-1)/2]/mm_I[(thrust::get<1>(t)-1)/2])*m_x[thrust::get<1>(t)];
            }
        }
    };

    phase_oscillators( int N,const state_type &A,const value_type *G,const value_type *I,const value_type *Fw,const value_type *F )
        : m_A( A ) ,m_G( G ) ,m_I( I ) ,m_F( F ) ,m_Fw( Fw ) , m_N( N ) 
        {

        }

    void operator() ( const state_type &x , state_type &dxdt , const value_type t )
    {
        state_type megicsum=mean_force_calculator::get_mean_force(x,m_A,m_N);
        if(ceilf(t)==t && (int) t%10==0)
        {
            printf("tiempo: %lf\n",t);
        }
        thrust::counting_iterator<int> it1(0);
        thrust::counting_iterator<int> it2 = it1 + 2*m_N-1;
        thrust::for_each(
                thrust::make_zip_iterator( thrust::make_tuple( x.begin() , it1, dxdt.begin() ) ),
                thrust::make_zip_iterator( thrust::make_tuple( x.end()   , it2, dxdt.end()   ) ) ,
                sys_functor(thrust::raw_pointer_cast(x.data()),t,m_G,m_I,m_F,m_Fw,thrust::raw_pointer_cast(megicsum.data()))
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
        for (int i = 0; i < m_N*2; ++i)
        {
            m_states[m_current_it*2*m_N+i]=x[i];
        }
        
        m_times.push_back( t );
        m_current_it=m_current_it+1;
    }
};



void inicialcond(thrust::host_vector<value_type> &x,int N,boost::mt19937 &rng,int caso)
{
    boost::uniform_real<> unif( 0, 2*M_PI );//la distribucion de probabilidad uniforme entre cero y 2pi
    boost::variate_generator< boost::mt19937&, boost::uniform_real<> > gen( rng , unif );//gen es una funcion que toma el engine y la distribucion y devuelve el numero random

    if(caso==0)
    {
    	FILE *w= fopen("Xi.txt", "w");
    	for (int i = 0; i < N; ++i)
		{
			fprintf(w, "%f  ",gen() );
			fprintf(w, "%f\n",0.0 );
		}
		fclose(w);
		FILE *r= fopen("Xi.txt", "r");
		for (int i = 0; i < N; ++i)
		{
			fscanf(r, "%lf", &x[2*i]); // posicion inicial i
			fscanf(r, "%lf", &x[2*i+1]); // momento inicial i
		}
		fclose(r);
    }
    if(caso==1)
    {
    	FILE *r= fopen("Xi.txt", "r");
		for (int i = 0; i < N; ++i)
		{
			fscanf(r, "%lf", &x[2*i]); // posicion inicial i
			fscanf(r, "%lf", &x[2*i+1]); // momento inicial i
		}
		fclose(r);
    }
}

void fillA(arma::Mat<double> &A,int N,boost::mt19937 &rng,int caso,double prob_0)
{

    boost::uniform_real<> unif( 0, 1 );//la distribucion de probabilidad uniforme entre cero y 2pi
    boost::variate_generator< boost::mt19937&, boost::uniform_real<> > gen( rng , unif );//gen es una funcion que toma el engine y la distribucion y devuelve el numero random
    if (caso==0)
    {
    	FILE *w= fopen("Ai.txt", "w");
    	for (int i = 0; i < N; ++i)
		{
			for (int j = 0; j <= i; ++j)
			{
				if(gen()>=prob_0)
				{
					fprintf(w, "%f  ",1.0);
				}
				else
				{
					fprintf(w, "%f  ",0.0);
				}

			}
		}
		fclose(w);
		FILE *r= fopen("Ai.txt", "r");
    	for (int i = 0; i < N; ++i)
		{
			for (int j = 0; j <= i; ++j)
			{
				fscanf(r,"%lf",&A(i,j));
			}
		}
		fclose(r);
    	for (int i = 0; i < N; ++i)
		{
			for (int j = N-1; j > i; --j)
			{
				A(i,j)=A(j,i);
			}
		}
    }
    if(caso==1)
    {
    	FILE *r= fopen("Ai.txt", "r");
    	for (int i = 0; i < N; ++i)
		{
			for (int j = 0; j <= i; ++j)
			{
				fscanf(r,"%lf",&A(i,j));
			}
		}		
		fclose(r);
    	for (int i = 0; i < N; ++i)
		{
			for (int j = N-1; j > i; --j)
			{
				A(i,j)=A(j,i);
			}
		}
    }
}

void fillG(std::vector<double> &G,int N,boost::mt19937 &rng,int caso)
{

    boost::normal_distribution<> unif(2.5, 0.2 );//la distribucion de probabilidad uniforme entre cero y 2pi
    boost::variate_generator< boost::mt19937&, boost::normal_distribution<> > gen( rng , unif );//gen es una funcion que toma el engine y la distribucion y devuelve el numero random

    if(caso==0)
    {
    	FILE *w= fopen("Gi.txt", "w");
		for (int i = 0; i < N; ++i)
		{
			fprintf(w, "%lf  ", gen());
		}
		fclose(w);
		FILE *r= fopen("Gi.txt", "r");
		for (int i = 0; i < N; ++i)
		{
			fscanf(r, "%lf", &G[i]);
		}
		fclose(r);
	}
	if(caso==1)
	{
		FILE *r= fopen("Gi.txt", "r");
		for (int i = 0; i < N; ++i)
		{
			fscanf(r, "%lf", &G[i]);
		}
		fclose(r);
	}
}

void fillI(std::vector<double> &I,int N,boost::mt19937 &rng,int caso)
{
    if (caso==0)
    {
    	FILE *w= fopen("Ii.txt", "w");
		for (int i = 0; i < N; ++i)
		{
			fprintf(w, "%lf  ", 1.0);
		}
		fclose(w);
		FILE *r= fopen("Ii.txt", "r");
		for (int i = 0; i < N; ++i)
		{
			fscanf(r, "%lf", &I[i]);
		}
		fclose(r);
    }
    if(caso==1)
    {
		FILE *r= fopen("Ii.txt", "r");
		for (int i = 0; i < N; ++i)
		{
			fscanf(r, "%lf", &I[i]);
		}
		fclose(r);
    }

}

void fillW(std::vector<double> &Fw,int N,boost::mt19937 &rng,int caso)
{
    boost::uniform_real<> unif( 0, 10 );//la distribucion de probabilidad uniforme entre cero y 2pi
    boost::variate_generator< boost::mt19937&, boost::uniform_real<> > gen( rng , unif );//gen es una funcion que toma el engine y la distribucion y devuelve el numero random

    if(caso==0)
    {
    	FILE *w= fopen("Wi.txt", "w");
    	for (int i = 0; i < N; ++i)
		{
			fprintf(w, "%lf  ", 1.0);
		}
		fclose(w);
		FILE *r= fopen("Wi.txt", "r");
    	for (int i = 0; i < N; ++i)
		{
			fscanf(r, "%lf", &Fw[i]);
		}
		fclose(r);
    }
    if(caso==1)
    {
		FILE *r= fopen("Wi.txt", "r");
    	for (int i = 0; i < N; ++i)
		{
			fscanf(r, "%lf", &Fw[i]);
		}
		fclose(r);
    }
}

void fillFw(std::vector<double> &Fw,int N,boost::mt19937 &rng,int caso)
{
    boost::uniform_real<> unif( 0, 10 );//la distribucion de probabilidad uniforme entre cero y 2pi
    boost::variate_generator< boost::mt19937&, boost::uniform_real<> > gen( rng , unif );//gen es una funcion que toma el engine y la distribucion y devuelve el numero random

    if(caso==0)
    {
    	FILE *w= fopen("Fwi.txt", "w");
    	for (int i = 0; i < N; ++i)
		{
			if(i==0)
			{
				fprintf(w, "%lf  ", 10.0);
			}
			else
			{
				fprintf(w, "%lf  ", 0.0);
			}
			
		}
		fclose(w);
		FILE *r= fopen("Fwi.txt", "r");
    	for (int i = 0; i < N; ++i)
		{
			fscanf(r, "%lf", &Fw[i]);
		}
		fclose(r);
    }
    if(caso==1)
    {
		FILE *r= fopen("Fwi.txt", "r");
    	for (int i = 0; i < N; ++i)
		{
			fscanf(r, "%lf", &Fw[i]);
		}
		fclose(r);
    }
}

void printsave(size_t steps, thrust::host_vector<value_type> &x_vec,std::vector<double> &times,int N,int i, int loops) //1 tiempo. 2 posicion. 3 momento. 4 energia potencial. 5 energia cinetica. 6 energia. 7 energia Total
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
		if(i%(steps/100)==0 && i < steps)
		{
			printf("printing savestate: %d \n", (int)(100.0*i/steps));
		}
		fprintf(f1,"%.15lf  ",times[i] );
		for (int j = 0; j < N; ++j)
		{
			fprintf(f1,"%.15lf	  %.15lf   ",x_vec[2*N*i+2*j],x_vec[2*N*i+2*j+1]); //1 posicion. 2 momento. 3 energia potencial. 4 energia cinetica. 5 energia total
		}
		fprintf(f1,"\n");
	}
	fclose(f1);
}

void pasteA(arma::Mat<double> &A,thrust::host_vector< value_type > &h_A,int N)
{
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			h_A[i+N*j]=A(i,j);
		}
	}
}

void paste(std::vector<value_type> &G,thrust::host_vector< value_type > &h_G,int N)
{
	for (int i = 0; i < N; ++i)
	{
		h_G[i]=G[i];
	}
}

int number_of_loops(value_type Total_time, int N, value_type dt)
{
    int GB_inMemory=3.5;

    size_t total_bytes=(size_t)((sizeof(value_type)*Total_time*2*N/dt));
    return(1+(int)(total_bytes/(GB_inMemory*pow(1024,3)-sizeof(value_type)*(6*N+N*N))));
}

void integrate(int N,state_type &x_vec,value_type dt,state_type &d_x,phase_oscillators &sys,value_type start_time,value_type end_time,thrust::host_vector<value_type> &x_vec_host,int i, int loops)
{
    runge_kutta4< state_type , value_type , state_type , value_type > stepper;
    int current_it=0;
    std::vector<value_type> times;
    size_t steps=integrate_adaptive( stepper , sys , d_x , start_time , end_time , dt,push_back_state_and_time( x_vec , times,N,current_it ) );
    thrust::copy(x_vec.begin(), x_vec.end(), x_vec_host.begin());
    printsave(steps,x_vec_host,times,N,i,loops);
    for (int j = 0; j < N; ++j)
    {
        d_x[2*j]=x_vec[2*N*(steps-1)+2*j];
        d_x[2*j+1]=x_vec[2*N*(steps-1)+2*j+1];
    }
}


int main()
{
    boost::mt19937 rng(static_cast<unsigned int>(std::time(0)));  /// el engine para generar numeros random


///////////////////////////////////////////////////////////////////////
    int N;
    printf("N: ");
    std::cin >>N;
    
    int load;
    printf("Load CI (0 NO, 1 YES): ");
    std::cin >>load;

    value_type Total_time;
    printf("Total_time : ");
    std::cin >>Total_time;

    value_type dt;
    printf("dt : ");
    std::cin >>dt;




    arma::Mat<value_type> A(N,N);
    std::vector<value_type> I(N);
    std::vector<value_type> G(N);
    std::vector<value_type> F(N);
    std::vector<value_type> Fw(N);
	thrust::host_vector<value_type> x(2*N); //condiciones iniciales

    int loops=number_of_loops(Total_time,N,dt);
    loops=2;
    printf("%d\n",loops );



//////////////////////////////////////////////////////////////////////////
	fillA(A,N,rng,load,0);
	fillG(G,N,rng,load);
	fillI(I,N,rng,load);
	fillFw(F,N,rng,load);
	fillW(Fw,N,rng,load);
	inicialcond(x,N,rng,load); ///

	thrust::host_vector< value_type > h_A(N*N);
	pasteA(A,h_A,N);
	thrust::host_vector< value_type > h_G(N);
	paste(G,h_G,N);
	thrust::host_vector< value_type > h_I(N);
	paste(I,h_I,N);
	thrust::host_vector< value_type > h_Fw(N);
	paste(Fw,h_Fw,N);
	thrust::host_vector< value_type > h_F(N);
	paste(F,h_F,N);

	state_type d_A=h_A;
	state_type d_G=h_G;
	state_type d_I=h_I;
	state_type d_Fw=h_Fw;
	state_type d_F=h_F;

	state_type d_x=x;
////////////////////////////////////////////////////////////////////////////

    int steps_estim=(int)(Total_time/(dt*loops));
    state_type x_vec(2*N*steps_estim+1);
    thrust::host_vector<value_type> x_vec_host(2*N*steps_estim+1);
    phase_oscillators sys(N,d_A,thrust::raw_pointer_cast(d_G.data()),thrust::raw_pointer_cast(d_I.data()),thrust::raw_pointer_cast(d_Fw.data()),thrust::raw_pointer_cast(d_F.data()));
    for (int i = 0; i < loops; ++i)
    {
        value_type start_time=i*Total_time/loops;
        value_type end_time=(i+1)*Total_time/loops;
        integrate(N,x_vec,dt,d_x,sys,start_time,end_time,x_vec_host,i,loops);
    }
     




////////////////////////////////////////////////////////////////////////////////

	
	/*system("cat save_1.txt save_2.txt save_3.txt save_4.txt> save.txt");
	//printstuff(A,steps,x_vec,times,I,N,K,G);
	system("cat ac_1.txt ac_2.txt ac_3.txt ac_4.txt> ac.txt");
	system("rm -f {ac_1,ac_2,ac_3,ac_4}.txt");
	system("rm -f {save_1,save_2,save_3,save_4}.txt");*/
	printf("N=%d\n",N);

	return 0;
}