# Heterogeneous Compute Playground

A lightweight playground for exploring performance and energy behavior across two worlds:
CUDA linear algebra kernels in modern C++, and Verilog waveform analysis through Python.

## Overview

This repository combines:
- **CUDA** experiments for core linear-algebra operations.
- **Verilog** simulation workflows using `wave.vcd`.
- **Python visualization** to analyze switching activity, state breakdowns, and energy-like proxies.

The goal is to build intuition for high-performance compute from both the GPU and hardware-design sides.

## Features

### CUDA / C++
- Modern C++ CUDA implementation.
- Vector addition and other linear-algebra building blocks.
- Designed to be lightweight, fast, and easy to extend.

### Verilog / VCD analysis
- Reads simulation traces from `wave.vcd`.
- Extracts signal activity over time.
- Computes switching activity, utilization, and state fractions.
- Generates execution plots, breakdown plots, and heatmaps.

### Python visualization
- Signal timeline plots.
- State breakdown analysis.
- Memory-vs-compute heatmaps.
- Simple energy proxy metrics based on switching activity and signal states.

## Why this repo?

This project is not just about raw speed.
It is about understanding how compute behaves:
- on a GPU,
- inside a digital design,
- and across time in simulation traces.

That makes it useful for performance engineering, hardware-aware ML, and energy-efficient compute exploration.

## Project structure

```text
.
├── basic/                # CUDA / C++ source code
├── verilogFPGA/            # Verilog modules
├── wave.vcd            # Simulation waveform
├── cudaenergy.py       # VCD parsing and plotting
└── output/             # Generated plots and results (Ongoing)
```

## Python analysis

The `cudaenergy.py` script:
- loads `wave.vcd`,
- parses signals with `vcdvcd`,
- computes switching activity,
- estimates an energy proxy,
- and saves plots for execution and signal breakdowns.

Example usage:

```bash
/bin/python3 cudaenegy.py wave.vcd --op "TOP.op[1:0]" --y "TOP.y[7:0]"
```

## CUDA build

```bash
mkdir build
cd build
cmake ..
make
./your_cuda_binary
```

## Outputs

The repository can generate:
- waveform execution plots,
- state breakdown figures,
- memory/compute heatmaps,
- CUDA benchmark results.

## Roadmap

- Add more CUDA linear algebra kernels.
- Extend Verilog modules and waveform cases.
- Compare signal activity across different designs.
- Add performance and energy benchmarking tables.
