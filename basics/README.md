CUDA Linear Algebra in C++

A lightweight, high-performance CUDA playground for core linear-algebra operations written in modern C++.
Overview

This repository focuses on small, readable CUDA programs that make GPU programming concepts concrete through linear algebra examples. It targets both learning and benchmarking, using CUDA Toolkit 12.1 in the current examples and relying on a CUDA-capable NVIDIA GPU plus CMake 3.10 or newer.
What is benchmarked

The current benchmark set covers three GPU-oriented workloads:

    Vector addition: a minimal CUDA kernel used to study device memory allocation, host-device transfers, kernel launch configuration, and thread/block indexing. This is the simplest entry point for understanding how block size changes runtime and hardware utilization on the GPU.

    Eigenvalue computation for real symmetric matrices: a numerical linear algebra workload built around CUDA-enabled routines for symmetric problems, useful for exploring GPU behavior on dense matrix operations where arithmetic intensity is higher than in vector-add style kernels.

    QR decomposition on GPU: a GPU implementation of QR decomposition with comparison against NumPy, intended to show where CUDA acceleration helps and where CPU-side libraries may still be competitive depending on matrix size and launch overhead.

Why these benchmarks matter

GPU acceleration is most informative and usually most beneficial when the workload is large enough to keep many threads active and to amortize launch and transfer overhead. NVIDIA’s profiling guidance emphasizes that GPUs need substantial parallel work and enough concurrent blocks to keep the device busy, which is why large vectors, medium-to-large matrices, and repeated batch operations are the most meaningful cases to benchmark.

In practice, the most useful benchmark ranges in this repository are:

    Large vector sizes, especially around 10^6 elements and above.

    Matrices with at least a few hundred rows and columns.

    Repeated runs or batches, where kernel launch overhead is less dominant.

Smaller inputs still run correctly and remain useful for teaching, but they may not outperform CPU implementations because overheads can dominate the actual arithmetic work.
Environment

The current setup assumes:

    NVIDIA GPU with CUDA support.

    CUDA Toolkit 12.x recommended; the current examples use 12.1.

    CMake 3.10 or later.

Planned extensions

Likely next additions include iterative solvers such as the power method or Lanczos method, along with visualization notebooks that connect Python to CUDA experiments for easier result inspection and comparison.
Next steps

A practical roadmap for extending this repository is:

    Add systematic input-size sweeps for vectors and matrices so each algorithm can be benchmarked across small, medium, and large problems instead of a single fixed case. This makes it easier to see when GPU acceleration starts to outweigh launch and transfer overhead.

    Add CPU baselines for every operation, including NumPy or Eigen-based references where appropriate, so the repository measures not only correctness but also crossover points between CPU and GPU execution.

    Add Nsight Compute profiling scripts for each kernel to track runtime, occupancy, memory throughput, cache behavior, and launch configuration in a repeatable way. NVIDIA’s profiling workflow is built around identifying whether kernels are memory-bound, compute-bound, or limited by latency and launch shape.

    Extend the linear algebra set with iterative solvers such as the power method and Lanczos method, which naturally connect to the existing eigenvalue and decomposition examples and provide more realistic numerical workloads.

    Add batched experiments for repeated QR, eigenvalue, and matrix workloads, because GPUs are most revealing when the problem exposes enough parallel work and amortizes launch overhead over many operations.

    Add visualization notebooks in Python for plotting timing curves, throughput trends, and accuracy comparisons, making it easier to compare kernels and parameter choices side by side.

    Add automated correctness tests against trusted CPU libraries so performance experiments do not drift away from numerical correctness as kernels become more optimized.
