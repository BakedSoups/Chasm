#include "kernel.cuh"
#include <stdio.h>

__global__ void addVectors(float* a, float* b, float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

void launchAddVectors(float* a, float* b, float* c, int size) {
    // Calculate grid and block sizes
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch kernel
    addVectors<<<blocksPerGrid, threadsPerBlock>>>(a, b, c, size);
    
    // Wait for GPU to finish
    cudaDeviceSynchronize();
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}