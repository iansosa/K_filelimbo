#include<stdio.h>
#include <cuda_runtime.h>
#include "gpu_timer.h"
#include "cpu_timer.h"
#include<cublas_v2.h>
#include<cublasXt.h>
#include <fstream>

// dimension de la matriz por default
#define DIM	1024

// si prefiere trabajar con indices de fila y columna 
// estos macros son utiles:

// C[IDX2C(i,j,M)] == valor en fila i (=0,...,Width-1) columna (j=0,1,...Height-1), row-major-C
#define  IDX2C(i,j,ld) (((j)*(ld))+( i )) 

// C[IDX2F(i,j,M)] == valor en fila i (=1,...,Width) columna (j=1,...Height), column-major-F
#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1)) 


// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
// Matrices are stored in column-major order:
// M(row, col) = *(M.elements + col * M.width + row)
typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

// Thread block size
#define BLOCK_SIZE 16



////////////////////////////////////////////////////////////////////////
// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
               cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size,
               cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}


// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // En "C-row-major" order seria 
    /*for (int e = 0; e < A.width; ++e)
        Cvalue += A.elements[row * A.width + e]
                * B.elements[e * B.width + col];
    C.elements[row * C.width + col] = Cvalue;
    */	

    // "column-major" order
    for (int e = 0; e < A.width; ++e)
    	Cvalue += A.elements[row + e*A.width]
        * B.elements[e + col * B.width];

    C.elements[row + col * C.width] = Cvalue;
}


// Matrix multiplication in cpu (column-major)
void MatMulcpu(Matrix A, Matrix B, Matrix C)
{
    for(int row=0; row<A.width;row++){
    	for(int col=0; col<A.height;col++){
    		float Cvalue = 0;
    		for (int e = 0; e < A.width; ++e)
        		Cvalue += A.elements[row + e*A.width]
                	* B.elements[e + col * B.width];

    		C.elements[row + col * C.width] = Cvalue;
    	}
    }
}



////////////////////////////////////////////////////////////////////////
// cublasXt API:
// como cublas, pero recibe matrices input y output del HOST
// se encarga de todas las alocaciones en device copias HD y DH 
void MatMulCublasXt(const Matrix A, const Matrix B, Matrix C)
{
	float  al=1.0f;                 
	float bet =0.0f;
	int m=C.width;

	cublasXtHandle_t manija;
	cublasXtCreate(&manija);

	int devices[1] = { 0 }; 
	cublasXtDeviceSelect(manija, 1, devices);

	cublasXtSgemm(manija,CUBLAS_OP_N,CUBLAS_OP_N,m,m,m,&al,A.elements,m,B.elements,m,&bet,C.elements,m);

	cublasXtDestroy(manija);
}



////////////////////////////////////////////////////////////////////////
// cublas API:
// como cublas, pero recibe matrices input y output del DEVICE
// nos tenemos que encargar de todas las alocaciones en device y copias HD y DH 
void MatMulCublas(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    //cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);

    Matrix d_B;
    d_B.width = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    //cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

	// buena costumbre: hacer algo con los codigos de error
    	cublasStatus_t stat;

	float  al=1.0f;                 
	float bet =0.0f;
	int m=C.width;

	cublasHandle_t manija;
	stat=cublasCreate(&manija);

	// Esto:
    	//cudaMemcpy(A.elements, d_A.elements, size,cudaMemcpyDeviceToHost);
	//cudaMemcpy(B.elements, d_B.elements, size,cudaMemcpyDeviceToHost);
	//cudaMemcpy(C.elements, d_C.elements, size,cudaMemcpyDeviceToHost);
	// Es equivalente a esto:
	stat = cublasSetMatrix(m,m,sizeof(float),(A.elements),m,(d_A.elements) ,m);//a -> d_a
	stat = cublasSetMatrix(m,m,sizeof(float),(B.elements),m,(d_B.elements) ,m);//b -> d_b
	stat = cublasSetMatrix(m,m,sizeof(float),(C.elements),m,(d_C.elements) ,m);//c -> d_c

	// multiplication
	stat=cublasSgemm(manija,CUBLAS_OP_N,CUBLAS_OP_N,m,m,m,&al, d_A.elements,m,d_B.elements,m, &bet, d_C.elements,m);

	// La variable stat se puede usar asi (recomendado para todas las llamadas...)
	if (stat != CUBLAS_STATUS_SUCCESS)
    	{
        	fprintf(stderr, "!!!! CUBLAS Sgemm error\n");
        	exit(1);
    	}

    	// Hacer esto:
    	//cudaMemcpy(C.elements, d_C.elements, size,cudaMemcpyDeviceToHost);
	// es equivalente a esto:
	stat=cublasGetMatrix(m,m,sizeof(float),(d_C.elements) ,m,(C.elements),m);	//d_c->c

    // Free device memory
    cublasDestroy(manija); // liberamos las "variables ocultas" de cublas
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

// activamos todas las APIs

/*#define SIMPLECPU
#define SIMPLECUDA
#define CUBLAS
#define CUBLASXt
*/

////////////////////////////////////////////////////////////////////////
// imprime matrix C ordenada a lo "fortran", column-major 
void print_matrix(Matrix C,std::ofstream &fout){
	// para evitar imprimir demasiado, imprimimos sub-matriz de max x max
	int max=128;
	int maxheight=(C.height>max)?(max):(C.height);
	int maxwidth=(C.width>max)?(max):(C.width);

	int i,j;
	for(i=0;i<maxheight;i++){
		for(j=0;j<maxwidth-1;j++){
			fout << C.elements[j*C.width+i] << " ";
		}
		fout << C.elements[j*C.width+i] << "\n";
	}
}


int main(int argc, char **argv)
{
	// Usaremos, por simplicidad, matrices cuadradas de NxN
	// N viene del primer argumento de la linea de comandos, sino se fija a DIM=1024 
	int N;
	if(argc>1){
		N=atoi(argv[1]);
	}else N=DIM;

	// matrices para hacer C=A*B	
        Matrix A, B, C;

	// inicializacion de las dimensiones de las matrices (cuadradas por simplicidad)
	A.width=B.width=C.width=A.height=B.height=C.height=N; 
	
    	size_t size = A.width * A.height * sizeof(float);

	// alocacion de matrices en el HOST
	A.elements=(float *)malloc(size);
	B.elements=(float *)malloc(size);
	C.elements=(float *)malloc(size);

	// inicializacion de matrices en el HOST
	for(int i=0;i<A.width*A.height;i++)
	{
		// matriz aleatoria
		A.elements[i]=rand()*1.f/RAND_MAX;
		//B.elements[i]=rand()*1.f/RAND_MAX;

		// matriz identidad
		B.elements[i]=(i%(A.width+1)==0)?(1.0f):(0.0f);
		C.elements[i]=0.0f;
	}

	std::ofstream foutA("A.dat");
	std::ofstream foutB("B.dat");
	print_matrix(A,foutA);
	print_matrix(B,foutB);

	// objeto timer de CPU (¿porque de CPU?)
	cpu_timer reloj_cpu;


	#ifdef SIMPLECPU
	reloj_cpu.tic();
	MatMulcpu(A,B,C);
	printf("N= %d simple cpu: %f ms\n", N, reloj_cpu.tac());
	std::ofstream foutCPU("Ccpu.dat");
	print_matrix(C,foutCPU);
	#endif

	#ifdef SIMPLECUDA
	MatMul(A,B,C); //warmup
	reloj_cpu.tic();
	MatMul(A,B,C);
	printf("N= %d simple_CUDA: %f ms\n",N, reloj_cpu.tac());
	std::ofstream foutSC("Ccuda.dat");
	print_matrix(C,foutSC);
	#endif

	#ifdef CUBLAS
	MatMulCublas(A,B,C); //warmup
	reloj_cpu.tic();
	MatMulCublas(A,B,C);
	printf("N= %d cublas: %f ms\n", N, reloj_cpu.tac());
	std::ofstream foutCUBLAS("Ccublas.dat");
	print_matrix(C,foutCUBLAS);
	#endif


	#ifdef CUBLASXt
	MatMulCublasXt(A,B,C); //warmup
	reloj_cpu.tic();
	MatMulCublasXt(A,B,C);
	printf("N= %d cublasXt: %f ms\n", N, reloj_cpu.tac());
	std::ofstream foutCUBLASXt("Ccublasxt.dat");
	print_matrix(C,foutCUBLASXt);
	#endif


        return 0;
}

