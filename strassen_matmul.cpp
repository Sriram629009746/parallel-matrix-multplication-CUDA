#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>


using namespace std;


int N;
int THRESHOLD;
int BLOCK_SIZE;

//fill up the matrix
void fillMatrix(int *M)
{
    for(int i = 0 ; i < N*N ; i++)
	{
        M[i] = rand()%100;
    }
}


//spilt matrix of size n to 4 sub-matrices of size n/2
__global__ void split(int *X11, int *X12, int *X21, int *X22, int *X, int n) 
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if(i < n && j < n) 
	{
		X11[i * n + j] = X[i * 2 * n + j];
		X12[i * n + j] = X[i * 2 * n + j + n];
		X21[i * n + j] = X[(i + n) * 2 * n + j];
		X22[i * n + j] = X[(i + n) * 2 * n + j + n];
	}
}


//kernel function to add two matrices
__global__ void add(int *A, int *B, int *C, int n) 
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if(i < n && j < n) 
	{
		C[i * n + j] = A[i * n + j] + B[i * n + j];
	}
}


//kernel function to subtract two matrices
__global__ void sub(int *A, int *B, int *C, int n) 
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if(i < n && j < n) 
	{
		C[i * n + j] = A[i * n + j] - B[i * n + j];
	}
}


//kernel function to multiply two terminal matrices
__global__ void mul(int *A, int *B, int *C, int n) 
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	if(i < n && j < n) 
	{
		C[i * n + j] = 0;
		for(int k = 0; k < n; k++) 
		{
			C[i * n + j] += A[i * n + k] * B[k * n + j];
		}
	}
}


// kernel function to merge C11 and C12 and C21 and C22
__global__ void merge(int *C11, int *C12, int *C21, int *C22, int *C, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if(i < n && j < n) 
	{
		C[i * 2 * n + j] = C11[i * n + j];
		C[i * 2 * n + j + n] = C12[i * n + j];
		C[(i + n) *2 * n + j] = C21[i * n + j];
		C[(i + n) * 2 * n + j + n] = C22[i * n + j];
	}
}


//strassen algorithm for matrix multiplication
void strassen(int* A , int* B, int* C , int n)
{
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	
	int *A_gpu, *B_gpu, *C_gpu;
	
	cudaMalloc((void **)&A_gpu, sizeof(int) * n * n);
	cudaMalloc((void **)&B_gpu, sizeof(int) * n * n);
	cudaMalloc((void **)&C_gpu, sizeof(int) * n * n);
	cudaMemcpy(A_gpu, A, sizeof(int) * n * n, cudaMemcpyHostToDevice);
	cudaMemcpy(B_gpu, B, sizeof(int) * n * n, cudaMemcpyHostToDevice);

	//in N is less than threshold, compute result using mul
	if (n <= THRESHOLD)
	{	
		//cout << "reached THRESHOLD" << endl;
		dim3 grid((size_t)ceil((float)n / (int)block.x), (size_t)ceil((float)n / (int)block.y));
		
		mul<<<grid, block>>>(A_gpu, B_gpu, C_gpu, n);
		
		cudaDeviceSynchronize();
		
		cudaMemcpy(C, C_gpu, sizeof(int) * n * n, cudaMemcpyDeviceToHost);
		
		cudaFree(A_gpu);
		cudaFree(B_gpu);
		cudaFree(C_gpu);
		
		return;
	}
	
	// allocate cuda memory for submatrices
	int *A11, *A12, *A21, *A22, *B11, *B12, *B21, *B22, *C11, *C12, *C21, *C22, *M1, *M2 ,*M3 , *M4 , *M5 , *M6 , *M7;
	int m = n / 2 ;
	cudaMalloc((void **)&A11, sizeof(int) * m * m);
	cudaMalloc((void **)&A12, sizeof(int) * m * m);
	cudaMalloc((void **)&A21, sizeof(int) * m * m);
	cudaMalloc((void **)&A22, sizeof(int) * m * m);
	cudaMalloc((void **)&B11, sizeof(int) * m * m);
	cudaMalloc((void **)&B12, sizeof(int) * m * m);
	cudaMalloc((void **)&B21, sizeof(int) * m * m);
	cudaMalloc((void **)&B22, sizeof(int) * m * m);
	cudaMalloc((void **)&C11, sizeof(int) * m * m);
	cudaMalloc((void **)&C12, sizeof(int) * m * m);
	cudaMalloc((void **)&C21, sizeof(int) * m * m);
	cudaMalloc((void **)&C22, sizeof(int) * m * m);
	cudaMalloc((void **)&M1, sizeof(int) * m * m);
	cudaMalloc((void **)&M2, sizeof(int) * m * m);
	cudaMalloc((void **)&M3, sizeof(int) * m * m);
	cudaMalloc((void **)&M4, sizeof(int) * m * m);
	cudaMalloc((void **)&M5, sizeof(int) * m * m);
	cudaMalloc((void **)&M6, sizeof(int) * m * m);
	cudaMalloc((void **)&M7, sizeof(int) * m * m);
	
	int *wAM1, *wBM1, *wAM2, *wBM3, *wBM4, *wAM5, *wAM6, *wBM6, *wAM7, *wBM7, *wM1M4, *wM5M7, *wM1M2, *wM3M6;
	
	cudaMalloc((void **)&wAM1, sizeof(int) * m * m);
	cudaMalloc((void **)&wBM1, sizeof(int) * m * m);
	cudaMalloc((void **)&wAM2, sizeof(int) * m * m);
	cudaMalloc((void **)&wBM3, sizeof(int) * m * m);
	cudaMalloc((void **)&wBM4, sizeof(int) * m * m);
	cudaMalloc((void **)&wAM5, sizeof(int) * m * m);
	cudaMalloc((void **)&wAM6, sizeof(int) * m * m);
	cudaMalloc((void **)&wBM6, sizeof(int) * m * m);
	cudaMalloc((void **)&wAM7, sizeof(int) * m * m);
	cudaMalloc((void **)&wBM7, sizeof(int) * m * m);
	
	cudaMalloc((void **)&wM1M4, sizeof(int) * m * m);
	cudaMalloc((void **)&wM5M7, sizeof(int) * m * m);
	
	cudaMalloc((void **)&wM1M2, sizeof(int) * m * m);
	cudaMalloc((void **)&wM3M6, sizeof(int) * m * m);
	
	
	dim3 grid((size_t)ceil((float)m / (int)block.x), (size_t)ceil((float)m / (int)block.y));
	
	//split the matrix A to 4 parts
	split<<<grid, block>>>(A11, A12, A21, A22, A_gpu, m);
	cudaDeviceSynchronize();
	//split the matrix B to 4 parts
	split<<<grid, block>>>(B11, B12, B21, B22, B_gpu, m);
	cudaDeviceSynchronize();
	
	//M1
	add<<<grid, block>>>(A11, A22, wAM1, m);
	cudaDeviceSynchronize();
	
	add<<<grid, block>>>(B11, B22, wBM1, m);
	cudaDeviceSynchronize();
	
	strassen(wAM1 , wBM1 ,M1 , m);
	cudaDeviceSynchronize();
	
	//M2
	add<<<grid, block>>>(A21, A22, wAM2, m);
	cudaDeviceSynchronize();
	
	strassen(wAM2 , B11 , M2 , m);
	cudaDeviceSynchronize();
	
	//M3
	sub<<<grid, block>>>(B12, B22, wBM3, m);
	cudaDeviceSynchronize();
	
	strassen(A11 , wBM3 , M3 , m);
	cudaDeviceSynchronize();
	
	//M4
	sub<<<grid, block>>>(B21, B11, wBM4, m);
	cudaDeviceSynchronize();
	
	strassen(A22 , wBM4 , M4 , m );
	cudaDeviceSynchronize();
	
	
	//M5
	add<<<grid, block>>>(A11, A12, wAM5, m);
	cudaDeviceSynchronize();
	
	strassen(wAM5 , B22 , M5 , m );
	cudaDeviceSynchronize();
	
	
	
	//M6
	sub<<<grid, block>>>(A21, A11, wAM6, m);
	cudaDeviceSynchronize();
	
	add<<<grid, block>>>(B11, B12, wBM6, m); 
	cudaDeviceSynchronize();
	strassen(wAM6 , wBM6 , M6 , m);
	cudaDeviceSynchronize();
	
	//M7
	sub<<<grid, block>>>(A12, A22, wAM7, m);
	cudaDeviceSynchronize();
	
	add<<<grid, block>>>(B21, B22, wBM7, m);
	cudaDeviceSynchronize();
	
	strassen(wAM7 , wBM7 , M7 , m);
	cudaDeviceSynchronize();
	
	
	
	
	//C11
	add<<<grid, block>>>(M1, M4, wM1M4, m);
	cudaDeviceSynchronize();
	
	sub<<<grid, block>>>(M7, M5, wM5M7, m);
	cudaDeviceSynchronize();
	add<<<grid, block>>>(wM1M4, wM5M7, C11, m);
	cudaDeviceSynchronize();


	//C12
	add<<<grid, block>>>(M3, M5, C12, m);
	cudaDeviceSynchronize();

	//C21
	add<<<grid, block>>>(M2, M4, C21, m);
	cudaDeviceSynchronize();

	//C22
	sub<<<grid, block>>>(M1, M2, wM1M2, m);
	cudaDeviceSynchronize();
	add<<<grid, block>>>(M3, M6, wM3M6, m);
	cudaDeviceSynchronize();
	add<<<grid, block>>>(wM1M2, wM3M6, C22, m);
	cudaDeviceSynchronize();

	//merege the C11 , C12 , C21 , C22
	merge<<<grid, block>>>(C11, C12, C21, C22, C_gpu, m);	
	cudaDeviceSynchronize();

	cudaMemcpy(C, C_gpu, sizeof(int) * n * n, cudaMemcpyDeviceToHost);
	
	//free the allocated memory
	cudaFree(A11); 
	cudaFree(A12); 
	cudaFree(A21); 
	cudaFree(A22); 
	cudaFree(B11); 
	cudaFree(B12); 
	cudaFree(B21); 
	cudaFree(B22); 
	
	cudaFree(M1);
	cudaFree(M2);
	cudaFree(M3);
	cudaFree(M4);
	cudaFree(M5);
	cudaFree(M6);
	cudaFree(M7);
	
	cudaFree(wAM1); 
	cudaFree(wBM1);
	cudaFree(wAM2);
	cudaFree(wBM3);
	cudaFree(wBM4);
	cudaFree(wAM5);
	cudaFree(wAM6);
	cudaFree(wBM6);
	cudaFree(wAM7);
	cudaFree(wBM7);
	
	cudaFree(wM1M4);
	cudaFree(wM5M7);
	cudaFree(wM1M2);
	cudaFree(wM3M6);
	
	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(C_gpu);
}


int main(int argc, char* argv[])
{
	int k = atoi(argv[1]);
	N = pow(2,k);

	int kprime = atoi(argv[2]);
	THRESHOLD = N/pow(2,kprime);
	
	BLOCK_SIZE = atoi(argv[3]);
	
	cout << "K = "<< k << " , k'= " << kprime << endl;
	
	cout<<"Input Size: " << N <<" Recursion Threshold: " <<THRESHOLD<<" No of threads: " <<BLOCK_SIZE*BLOCK_SIZE<<endl; 
	
    size_t bytes = N * N * sizeof(int);
	
    int *h_A;
    int *h_B;
    int *h_C;
	
    h_A = (int *)malloc(bytes);
    h_B = (int *)malloc(bytes);
    h_C = (int *)malloc(bytes);
	
	
    fillMatrix(h_A);
    fillMatrix(h_B);
	
    struct timespec start, stop;
	clock_gettime(CLOCK_REALTIME, &start);
	
	strassen(h_A,h_B,h_C,N);
	
    clock_gettime(CLOCK_REALTIME, &stop);
	double total_time = (stop.tv_sec-start.tv_sec)+0.000000001*(stop.tv_nsec-start.tv_nsec);
	
    cout<<"Time taken for Strassen multiplication in seconds "<<total_time<<endl;
	
}