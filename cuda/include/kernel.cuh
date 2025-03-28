#ifndef KERNEL_CUH
#define KERNEL_CUH

// CUDA kernel function
__global__ void addVectors(float* a, float* b, float* c, int size);

// Host wrapper function
void launchAddVectors(float* a, float* b, float* c, int size);

#endif