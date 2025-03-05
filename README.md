# GEMM Optimization Project

This repository contains my optimized **General Matrix Multiplication (GEMM)** library, developed in **C++** with **SIMD (AVX/FMA)** for high-performance computing. Built as part of UC Berkeley’s EECS HPC coursework on the DOE-NERSC Perlmutter supercomputer, it achieves **66.58% CPU peak performance** (vs. 87.08% BLAS) and a **15× speedup** over naive implementations.

## Project Structure
-MY SOLUTION CAN BE FOUND IN -> dgemm-blocked.c
-Other files are dependencies or benchmarks

## Overview
- **Duration**: January 2025 – March 2025  
- **Goal**: Optimize matrix multiplication for speed and efficiency using parallel computing techniques.  
- **Key Results**:  
  - Performance: **66.58% CPU peak** (up from 4.47% naive)  
  - Speedup: **15×** over baseline  
  - Benchmark: 66.58% vs. BLAS’s 87.08%  

## Features
- **SIMD Optimization**: Leverages AVX/FMA instructions with an **8×6 micro-kernel** for maximum register use and spatial locality.  
- **Cache-Aware Blocking**: Repacks matrix A into a block-contiguous format, doubling temporal locality and slashing cache misses.  
- **Loop Unrolling**: Enhances throughput by minimizing overhead in inner loops.  

## Tech Stack
- **Languages**: C++, Assembly (for SIMD intrinsics)  
- **Tools**: GCC, Git, Perlmutter Supercomputer (NERSC), Command Line  
- **Libraries**: Standard C++ (no external dependencies beyond SIMD intrinsics)  

