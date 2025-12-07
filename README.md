CUDA Linear Algebra in C++

A lightweight, high-performance CUDA playground for core linear-algebra operations written in modern C++.
This project is built for learning, experimentation, and extension—whether you're exploring GPU computing for the first time or optimizing math kernels for real scientific workloads.

🚀 Features
1. Vector Addition (GPU-Accelerated)

A minimal yet illustrative example of how to:

Allocate device memory
Launch CUDA kernels
Transfer data between host/device

This serves as the foundation for understanding parallel decomposition and thread/block indexing.

2. Eigenvalue Computation for Real Symmetric Matrices

Implements eigenvalue calculation for real symmetric matrices using CUDA-enabled linear algebra routines.
Leverages well-structured GPU algorithms for symmetric matrices
Demonstrates device-level performance for fundamental numerical linear algebra tasks


NVIDIA GPU with CUDA support

CUDA Toolkit (12.x recommended) (In this examples we will use 12.1)

CMake 3.10+


GPU acceleration shines when working with:

Large vector sizes (10⁶+)

Matrices above a few hundred rows/cols

Repeated batch operations

Smaller problems still run correctly but may not outperform CPU versions—ideal for educational purposes.

🧱 Next Steps (Planned / Easy Extensions)

Matrix–matrix multiplication (GEMM)

QR decomposition on GPU

Iterative solvers (power method, Lanczos)

Visualization notebooks (Python ↔ CUDA bindings)

