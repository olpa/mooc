#include <cuda_device_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void vector_add(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void vector_add_stride(float* a, float* b, float* c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = idx; i < n; i += stride) {
    c[i] = a[i] + b[i];
  }
}


#define VECTOR_SIZE 1000000
#define VECTOR_BYTES (VECTOR_SIZE * sizeof(float))
#define BLOCK_SIZE 256

void run_vector_add(float* a, float* b, float* c) {
    int num_blocks = (VECTOR_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vector_add<<<num_blocks, BLOCK_SIZE>>>(a, b, c, VECTOR_SIZE);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: Kernel launch failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    // No need for cudaDeviceSynchronize: will sync in cudaMemcpy.
}

void run_vector_add_stride(float* a, float* b, float* c) {
    int num_blocks = (VECTOR_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vector_add_stride<<<num_blocks, BLOCK_SIZE>>>(a, b, c, VECTOR_SIZE);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: Kernel launch failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    // No need for cudaDeviceSynchronize: will sync in cudaMemcpy.
}

int main(int argc, char* argv[]) {
    int use_stride = 0;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--stride") == 0) {
            use_stride = 1;
        } else if (strcmp(argv[i], "--no-stride") == 0) {
            use_stride = 0;
        } else {
            printf("Usage: %s [--stride|--no-stride]\n", argv[0]);
            printf("  --stride: Use strided kernel execution\n");
            printf("  --no-stride: Use standard kernel execution (default)\n");
            return 1;
        }
    }

    // Allocate memory for vectors on host
    float *h_vec_a = (float*)malloc(VECTOR_BYTES);
    float *h_vec_b = (float*)malloc(VECTOR_BYTES);
    float *h_vec_sum = (float*)malloc(VECTOR_BYTES);
    float *h_vec_sum_fixture = (float*)malloc(VECTOR_BYTES);
    float *d_a, *d_b, *d_c;
    
    // Check host memory allocation
    if (!h_vec_a || !h_vec_b || !h_vec_sum || !h_vec_sum_fixture) {
        printf("Error: Failed to allocate host memory\n");
        return 1;
    }
    
    // Allocate memory on device
    cudaError_t err = cudaMalloc(&d_a, VECTOR_BYTES);
    if (err != cudaSuccess) {
        printf("Error: Failed to allocate device memory for d_a: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    err = cudaMalloc(&d_b, VECTOR_BYTES);
    if (err != cudaSuccess) {
        printf("Error: Failed to allocate device memory for d_b: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    err = cudaMalloc(&d_c, VECTOR_BYTES);
    if (err != cudaSuccess) {
        printf("Error: Failed to allocate device memory for d_c: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Read data from fixture.bin
    FILE *file = fopen("fixture.bin", "rb");
    if (!file) {
        printf("Error: Could not open fixture.bin\n");
        return 1;
    }

    fread(h_vec_a, VECTOR_BYTES, 1, file);
    fread(h_vec_b, VECTOR_BYTES, 1, file);
    fread(h_vec_sum_fixture, VECTOR_BYTES, 1, file);
    fclose(file);

    err = cudaMemcpy(d_a, h_vec_a, VECTOR_BYTES, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Error: Failed to copy data from host to device for d_a: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    err = cudaMemcpy(d_b, h_vec_b, VECTOR_BYTES, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Error: Failed to copy data from host to device for d_b: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Call appropriate vector addition function based on stride flag
    if (use_stride) {
        run_vector_add_stride(d_a, d_b, d_c);
    } else {
        run_vector_add(d_a, d_b, d_c);
    }

    err = cudaMemcpy(h_vec_sum, d_c, VECTOR_BYTES, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Error: Failed to copy data from device to host for result: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Verify that vec_a + vec_b = vec_sum
    for (int i = 0; i < VECTOR_SIZE; i++) {
        if (h_vec_sum_fixture[i] != h_vec_sum[i]) {
            printf("Verification failed at index %d: %f != %f\n",
                   i, h_vec_sum_fixture[i], h_vec_sum[i]);
            return 1;
        }
    }

    printf("Verification passed: vec_a + vec_b = vec_sum\n");

    // Free memory
    cudaFree(d_c);
    cudaFree(d_b);
    cudaFree(d_a);
    free(h_vec_sum_fixture);
    free(h_vec_sum);
    free(h_vec_b);
    free(h_vec_a);

    return 0;
}
