#include <cuda_runtime.h>
#include <cstdio>

#define CHECK(call) {cudaError_t e=call;if(e!=cudaSuccess){printf("CUDA error: %s\n",cudaGetErrorString(e));exit(1);}}

__global__ void matmul_simple(double *A, double *B, double *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        double sum = 0;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    int M=10;
    int K=20;
    int N=20;
    double *h_A=new double[M*K], *h_B=new double[K*N], *h_C=new double[M*N];
    
    // Random init
    srand(1234);
    for(int i=0; i<M*K; i++) h_A[i]=(rand()/(double)RAND_MAX*2-1);
    for(int i=0; i<K*N; i++) h_B[i]=(rand()/(double)RAND_MAX*2-1);
    
    printf("Matrix A =\n");
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < K; j++) 
            printf("%6.6f ", h_A[i*M+j]);
        printf("\n");
    }
    printf("Matrix B =\n");
    for(int i = 0; i < K; i++) {
        for(int j = 0; j < N; j++) 
            printf("%6.6f ", h_B[i*K+j]);
        printf("\n");
    }
    // GPU
    double *d_A, *d_B, *d_C;
    CHECK(cudaMalloc(&d_A,M*K*sizeof(double)));
    CHECK(cudaMalloc(&d_B,K*N*sizeof(double)));
    CHECK(cudaMalloc(&d_C,M*N*sizeof(double)));
    CHECK(cudaMemcpy(d_A,h_A,M*K*sizeof(double),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B,h_B,K*N*sizeof(double),cudaMemcpyHostToDevice));
    
    // SIMPLE 2D LAUNCH
    dim3 block(16, 16);
    dim3 grid((N+15)/16, (M+15)/16);
    matmul_simple<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CHECK(cudaDeviceSynchronize());
    
    CHECK(cudaMemcpy(h_C, d_C, M*N*sizeof(double), cudaMemcpyDeviceToHost));

    printf("Matrix C =\n");
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) 
            printf("%6.6f ", h_C[i*M+j]);
        printf("\n");
    }    
    // Cleanup
    delete[] h_A; delete[] h_B; delete[] h_C;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}
