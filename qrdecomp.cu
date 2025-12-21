#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
///////////////////////////////////////////////////////////
// Kernels: no atomicAdd, 1 block handles one column
///////////////////////////////////////////////////////////

__global__ void dot_block(const double* x, const double* y, int n, double* out) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int i   = tid;

    double sum = 0.0;
    while (i < n) {
        sum += x[i] * y[i];
        i += blockDim.x;
    }
    sdata[tid] = sum;
    __syncthreads();

    // reduction in one block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0)
        *out = sdata[0];
}

__global__ void saxpy_block(double* y, const double* x, double alpha, int n) {
    int i = threadIdx.x;
    while (i < n) {
        y[i] -= alpha * x[i];      // y = y - alpha * x
        i += blockDim.x;
    }
}

__global__ void norm_and_normalize_block(double* x, int n, double* out_norm) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int i   = tid;

    double sum = 0.0;
    while (i < n) {
        sum += x[i] * x[i];
        i += blockDim.x;
    }
    sdata[tid] = sum;
    __syncthreads();

    // reduction in one block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    double nrm;
    if (tid == 0) {
        nrm = std::sqrt(sdata[0]);
        *out_norm = nrm;
    }
    __syncthreads();

    // broadcast nrm from shared[0]
    nrm = std::sqrt(sdata[0]);  // same value for all threads

    // normalize
    i = tid;
    while (i < n) {
        x[i] /= nrm;
        i += blockDim.x;
    }
}

///////////////////////////////////////////////////////////
// Host function: Gram–Schmidt in terms of kernels above
///////////////////////////////////////////////////////////

void qr_gram_schmidt_cuda(const std::vector<double>& A_col, // column-major A
                          int m, int n,
                          std::vector<double>& Q_col,       // column-major Q
                          std::vector<double>& R_row)       // row-major R
{
    double *d_A, *d_Q, *d_u, *d_tmp;
    cudaMalloc(&d_A, m * n * sizeof(double));
    cudaMalloc(&d_Q, m * n * sizeof(double));
    cudaMalloc(&d_u, m * sizeof(double));
    cudaMalloc(&d_tmp, sizeof(double));

    cudaMemcpy(d_A, A_col.data(), m * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_Q, 0, m * n * sizeof(double));

    Q_col.assign(m * n, 0.0);
    R_row.assign(n * n, 0.0);

    int threads = 256;
    int shmem   = threads * sizeof(double);

    for (int j = 0; j < n; ++j) {
        double* d_aj = d_A + j * m;  // column a_j
        double* d_uj = d_u;          // u_j

        // u_j = a_j
        cudaMemcpy(d_uj, d_aj, m * sizeof(double), cudaMemcpyDeviceToDevice);

        // subtract projections on previous e_k
        for (int k = 0; k < j; ++k) {
            double* d_ek = d_Q + k * m;  // e_k

            double r_kj = 0.0;
            dot_block<<<1, threads, shmem>>>(d_aj, d_ek, m, d_tmp);
            cudaMemcpy(&r_kj, d_tmp, sizeof(double), cudaMemcpyDeviceToHost);

            R_row[k * n + j] = r_kj;

            saxpy_block<<<1, threads>>>(d_uj, d_ek, r_kj, m);
        }

        // r_jj = ||u_j|| and e_j = u_j / ||u_j||
        double r_jj = 0.0;
        norm_and_normalize_block<<<1, threads, shmem>>>(d_uj, m, d_tmp);
        cudaMemcpy(&r_jj, d_tmp, sizeof(double), cudaMemcpyDeviceToHost);
        R_row[j * n + j] = r_jj;

        // store e_j into Q
        cudaMemcpy(d_Q + j * m, d_uj, m * sizeof(double), cudaMemcpyDeviceToDevice);
    }

    cudaMemcpy(Q_col.data(), d_Q, m * n * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_Q); cudaFree(d_u); cudaFree(d_tmp);
}

///////////////////////////////////////////////////////////
// Example main
///////////////////////////////////////////////////////////

int main() {
    int m = 12, n = 14;
    std::vector<double> A_col(m * n);    // column-major: [a1 | a2 | a3]

// RNG setup (once)
    std::mt19937_64 rng(1234);           // seed
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

// Fill columns a1..an
    for (int j = 0; j < n; ++j) {        // column index
        for (int i = 0; i < m; ++i) {    // row index
            A_col[j * m + i] = dist(rng);  // element A(i,j)
    }
    }
    std::vector<double> Q_col, R_row;
    qr_gram_schmidt_cuda(A_col, m, n, Q_col, R_row);

    auto print_col = [&](const char* name,
                         const std::vector<double>& M, int rows, int cols) {
        std::cout << name << ":\n";
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j)
                std::cout << std::setw(10) << std::setprecision(6)
                          << M[j * rows + i] << " ";
            std::cout << "\n";
        }
        std::cout << "\n";
    };
    auto print_row = [&](const char* name,
                         const std::vector<double>& M, int rows, int cols) {
        std::cout << name << ":\n";
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j)
                std::cout << std::setw(10) << std::setprecision(6)
                          << M[i * cols + j] << " ";
            std::cout << "\n";
        }
        std::cout << "\n";
    };

    printf("Matrix A =\n");
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) 
            printf("%6.6f,", A_col[j*m+i]);
        printf("\n");
    }
    print_col("Q", Q_col, m, n);   // columns = e1,e2,e3
    print_row("R", R_row, n, n);   // upper triangular

    return 0;
}
