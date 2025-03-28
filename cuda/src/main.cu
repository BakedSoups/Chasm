#include <stdio.h>
#include <stdlib.h>
#include "kernel.cuh"

int main() {
    // Problem size
    int size = 1000000;
    size_t bytes = size * sizeof(float);
    
    // Host arrays
    float *h_a, *h_b, *h_c;
    
    // Allocate host memory
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);
    
    // Initialize host arrays
    for (int i = 0; i < size; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }
    
    // Device arrays
    float *d_a, *d_b, *d_c;
    
    // Allocate device memory
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    
    // Copy data to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    
    // Launch kernel
    launchAddVectors(d_a, d_b, d_c, size);
    
    // Copy result back to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    
    // Verify results
    bool success = true;
    for (int i = 0; i < size; i++) {
        if (h_c[i] != 3.0f) {{
            "version": "2.0.0",
            "tasks": [
                {
                    "label": "CUDA: Build",
                    "type": "shell",
                    "command": "\"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin/nvcc.exe\"",
                    "args": [
                        "-g",
                        "${file}",
                        "-o",
                        "${fileDirname}/${fileBasenameNoExtension}.exe"
                    ],
                    "group": {
                        "kind": "build",
                        "isDefault": true
                    },
                    "problemMatcher": []
                }
            ]
        }
            printf("Error: h_c[%d] = %f\n", i, h_c[i]);
            success = false;
            break;
        }
    }
    
    if (success) {
        printf("Vector addition successful!\n");
    }
    
    // Free memory
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return 0;
}