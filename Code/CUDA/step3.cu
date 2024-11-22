#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cstdio>

#define EPSILON 1e-7

bool checkAnswer(float* answer, float* calculated, int length) {
    for (int i = 0; i < length; i++) {
        if (fabs(answer[i] - calculated[i]) > EPSILON) {
            return false;
        }
    }
    return true;
}

__global__ void update_z(float theta, float* z_vm1, float* zhat_v, float* z_v, int length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) {
        z_v[i] = (1 - theta) * z_vm1[i] + theta * zhat_v[i];
    }
}

void update_z_sequential(float theta, int length, float* z_vm1, float* zhat_v, float* z_v) {
    for (int i = 0; i < length; i++) {
        z_v[i] = (1 - theta) * z_vm1[i] + theta * zhat_v[i];
    }
}

int main() {
    FILE* input;
    FILE* output;

    // Fixed subfolder to work with snprintf
    char subfolder = '1';
    char inputPath[256];
    char outputPath[256];
    snprintf(inputPath, sizeof(inputPath), "FinalProject/build/step3/%c/input.txt", subfolder);
    snprintf(outputPath, sizeof(outputPath), "FinalProject/build/step3/%c/output.txt", subfolder);

    int m;
    int N;
    int n_u;
    float theta;

    input = fopen(inputPath, "r");
    if (input == NULL) {
        perror("Error opening input file");
        return 1;
    }

    output = fopen(outputPath, "r");
    if (output == NULL) {
        perror("Error opening output file");
        fclose(input);
        return 1;
    }

    if (fscanf(input, "%d %d %d %f", &n_u, &N, &m, &theta) != 4) {
        perror("Error reading dimensions and theta");
        fclose(input);
        fclose(output);
        return 1;
    }

    int length = n_u * N;
    float *z_v, *zhat_v, *z_vm1, *actual_z_v;

    if (cudaMallocManaged(&z_v, length * sizeof(float)) != cudaSuccess ||
        cudaMallocManaged(&zhat_v, length * sizeof(float)) != cudaSuccess ||
        cudaMallocManaged(&z_vm1, length * sizeof(float)) != cudaSuccess ||
        cudaMallocManaged(&actual_z_v, length * sizeof(float)) != cudaSuccess) {
        perror("Error allocating CUDA memory");
        fclose(input);
        fclose(output);
        return 1;
    }

    for (int i = 0; i < length; i++) fscanf(input, "%f", &z_vm1[i]);
    for (int i = 0; i < length; i++) fscanf(input, "%f", &zhat_v[i]);
    for (int i = 0; i < length; i++) fscanf(output, "%f", &actual_z_v[i]);

    fclose(input);
    fclose(output);

    // Run sequential update
    update_z_sequential(theta, length, z_vm1, zhat_v, z_v);

    // Check result of sequential update
    if (checkAnswer(actual_z_v, z_v, length)) {
        printf("Sequential result is correct\n");
    } else {
        printf("Sequential result is incorrect\n");
    }

    // Launch CUDA kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (length + threadsPerBlock - 1) / threadsPerBlock;
    update_z<<<blocksPerGrid, threadsPerBlock>>>(theta, z_vm1, zhat_v, z_v, length);

    // Synchronize
    cudaDeviceSynchronize();

    // Check CUDA result
    if (checkAnswer(actual_z_v, z_v, length)) {
        printf("CUDA result is correct\n");
    } else {
        printf("CUDA result is incorrect\n");
        for (int i = 0; i < length; i++) {
            if (fabs(actual_z_v[i] - z_v[i]) > EPSILON) {
                printf("Mismatch at index %d: expected %f, got %f\n", i, actual_z_v[i], z_v[i]);
            }
        }
    }

    // Free CUDA memory
    cudaFree(z_v);
    cudaFree(zhat_v);
    cudaFree(z_vm1);
    cudaFree(actual_z_v);

    return 0;
}
