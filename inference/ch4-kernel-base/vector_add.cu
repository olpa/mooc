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
}

int main() {
    // Allocate memory for vectors on host
    float *h_vec_a = (float*)malloc(VECTOR_BYTES);
    float *h_vec_b = (float*)malloc(VECTOR_BYTES);
    float *h_vec_sum = (float*)malloc(VECTOR_BYTES);
    float *h_vec_sum_fixture = (float*)malloc(VECTOR_BYTES);
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, VECTOR_BYTES);
    cudaMalloc(&d_b, VECTOR_BYTES);
    cudaMalloc(&d_c, VECTOR_BYTES);

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

    cudaMemcpy(d_a, h_vec_a, VECTOR_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_vec_b, VECTOR_BYTES, cudaMemcpyHostToDevice);

    run_vector_add(d_a, d_b, d_c);

    cudaMemcpy(h_vec_sum, d_c, VECTOR_BYTES, cudaMemcpyDeviceToHost);

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
    free(h_vec_a);
    free(h_vec_b);
    free(h_vec_sum);

    return 0;
}
