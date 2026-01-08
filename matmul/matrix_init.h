#ifndef MATRIX_INIT_H
#define MATRIX_INIT_H

// Matrix initialization and CPU reference utilities for matmul benchmarking

// Allocate and initialize two square matrices with random values in [0, 1]
// Used for testing matrix multiplication kernels
void allocateAndInitMatrices(float **h_A, float **h_B, int N);

// CPU reference implementation of matrix multiplication using double precision
// Computes C = A × B where all matrices are N×N in row-major layout
// Uses double precision internally for better accuracy, then casts to float
void matmul_cpu_reference(const float *A, const float *B, float *C, int N);

#endif
