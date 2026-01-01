#ifndef VECTOR_INIT_H
#define VECTOR_INIT_H

#include <cuda_runtime.h>

// Allocate and initialize one host vector with random values
void allocateAndInitVector(float **h_vec, int n);

// Allocate one device vector
void allocateDeviceVector(float **d_vec, int n);

// Transfer one vector from host to device
void transferToDevice(float *d_vec, const float *h_vec, int n);

// Transfer one vector from device to host
void transferFromDevice(float *h_vec, const float *d_vec, int n);

// Free one host vector
void freeHostVector(float *h_vec);

// Free one device vector
void freeDeviceVector(float *d_vec);

#endif
