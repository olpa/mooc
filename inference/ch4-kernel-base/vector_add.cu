#include <stdio.h>
#include <stdlib.h>

__global__ void vector_add(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

#define VECTOR_SIZE 1000000
#define VECTOR_BYTES (VECTOR_SIZE * sizeof(float))

int main() {
    // Allocate memory for vectors on host
    float *h_vec_a = (float*)malloc(VECTOR_BYTES);
    float *h_vec_b = (float*)malloc(VECTOR_BYTES);
    float *h_vec_sum = (float*)malloc(VECTOR_BYTES);

    // Read data from fixture.bin
    FILE *file = fopen("fixture.bin", "rb");
    if (!file) {
        printf("Error: Could not open fixture.bin\n");
        return 1;
    }

    fread(h_vec_a, VECTOR_BYTES, 1, file);
    fread(h_vec_b, VECTOR_BYTES, 1, file);
    fread(h_vec_sum, VECTOR_BYTES, 1, file);
    fclose(file);

    // Verify that vec_a + vec_b = vec_sum
    for (int i = 0; i < VECTOR_SIZE; i++) {
        if (h_vec_a[i] + h_vec_b[i] != h_vec_sum[i]) {
            printf("Verification failed at index %d: %f + %f != %f\n",
                   i, h_vec_a[i], h_vec_b[i], h_vec_sum[i]);
            return 1;
        }
    }

    printf("Verification passed: vec_a + vec_b = vec_sum\n");

    // Free memory
    free(h_vec_a);
    free(h_vec_b);
    free(h_vec_sum);

    return 0;
}
