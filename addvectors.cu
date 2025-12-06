#include <cuda_runtime.h>
#include <cstdio>
#include <random>

__global__ void add(double *a, double *b, double *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    int n = 5;  // 1M elements
    printf("Adding %d random vectors:\n", 2);
    
    // Random init [0.0, 10.0)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0f, 10.0f);
    
    double *h_a = new double[n];
    double *h_b = new double[n];
    double *h_c = new double[n];
    
    for(int i = 0; i < n; i++) {
        h_a[i] = dis(gen);
        h_b[i] = dis(gen);
    }
    for (int i = 0; i < n; i++) {
        if (i==0){
          printf("A:%.2lf ", h_a[i]);   
        }else{
        printf("%.2lf ", h_a[i]);

        }
    }
    printf("\n");
    for (int i = 0; i < n; i++) {
        if (i==0){
          printf("B:%.2lf ", h_b[i]);   
        }else{
        printf("%.2lf ", h_b[i]);

        }
    }
    printf("\n");

    
    double *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n*sizeof(double));
    cudaMalloc(&d_b, n*sizeof(double));
    cudaMalloc(&d_c, n*sizeof(double));
    
    cudaMemcpy(d_a, h_a, n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n*sizeof(double), cudaMemcpyHostToDevice);
    
    int threads = min(256, n); ;
    int blocks = (n + threads - 1) / threads;
    add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_c, d_c, n*sizeof(double), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < n; i++) {
        if (i==0){
          printf("C:%.2lf ", h_c[i]);   
        }else{
        printf("%.2lf ", h_c[i]);

        }
    }
    printf("\n");    
    
    return 0;
}
