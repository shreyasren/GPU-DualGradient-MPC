#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define EPSILON 1e-7

bool checkAnswer(float* answer, float* calculated, int length) {
    bool correct = true;
    for (int i = 0; i < length; i++) {
        if (fabs(answer[i] - calculated[i]) > EPSILON) {
            std::cout << "Mismatch at index " << i << ": expected " << answer[i] << ", but got " << calculated[i] << std::endl;
            correct = false;
        }
    }
    return correct;
}


// CUDA kernel for computing zhat
__global__ void computeZhat(float* M_G,float* w_v, float* g_P, float* zhat_v, int N, int n_u, int m) {
    /*
    In this kernel there will be multiple block launches. The number of block launches will be a multiple of 5.
    The size of the matrix M_G will determine the number of blocks launched. if m * N / 5 > 256, then we will need to increase
    the total number of blocks launched. The number of threads per block will be 256. The we can seperate out what compututation needs
    to happen based on which block we are in. This should minimize control divergence while allowing for individual blocks to calulate 
    different cases.
    */

    //MG is a matrix of size N x m flatted from N * N_u x m

    //Y is going to repersent N
    //X is going to repersent m

    extern __shared__ float partialWv[];
    

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y;

    int grid = gridDim.x;
    int block = blockIdx.x;

    // the first 4/5 blocks should all use the same calculation for the flattened version of M_G
    if (block <= (4 * grid / 5)) {
        partialWv[(y * N) + x] += M_G[x + y * m] * w_v[x];
    }else{
        for (int i = 0; i < n_u; i++) {
            partialWv[y * n_u + i] += M_G[x + y * m] * w_v[x];
        }
    }

   
}


int main() {
    FILE* input;
    FILE* output;

    // File paths
    char subfolder = '1';
    char inputPath[256];
    char outputPath[256];
    snprintf(inputPath, sizeof(inputPath), "FinalProject/build/step2/%c/input.txt", subfolder);
    snprintf(outputPath, sizeof(outputPath), "FinalProject/build/step2/%c/output.txt", subfolder);

    int m, N, n_u;

    // Open input and output files
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

    // Read dimensions
    if (fscanf(input, "%d %d %d", &n_u, &N, &m) != 3) {
        perror("Error reading dimensions");
        fclose(input);
        fclose(output);
        return 1;
    }

    int length = N * n_u;

    // Allocate host memory
    float* M_G = (float*)malloc(N * m * sizeof(float));
    float* w_v = (float*)malloc(m * sizeof(float));
    float* g_P = (float*)malloc(length * sizeof(float));
    float* exp_zhat = (float*)malloc(length * sizeof(float));
    float* zhat_v_cpu = (float*)malloc(length * sizeof(float));

    // Read input data
    for (int i = 0; i < N * m; i++) fscanf(input, "%f", &M_G[i]);
    for (int i = 0; i < m; i++) fscanf(input, "%f", &w_v[i]);
    for (int i = 0; i < length; i++) fscanf(input, "%f", &g_P[i]);

    // Read expected output
    for (int i = 0; i < length; i++) fscanf(output, "%f", &exp_zhat[i]);

    fclose(input);
    fclose(output);

    //print M_G
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < m; j++) {
    //         std::cout << M_G[i * m + j] << " ";
    //     }
    // }

    // Allocate device memory
    float *d_M_G, *d_w_v, *d_g_P, *d_zhat_v;
    cudaMalloc(&d_M_G, N * m * sizeof(float));
    cudaMalloc(&d_w_v, m * sizeof(float));
    cudaMalloc(&d_g_P, length * sizeof(float));
    cudaMalloc(&d_zhat_v, length * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_M_G, M_G, N * m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w_v, w_v, m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_g_P, g_P, length * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    // Define block dimensions
    dim3 blockDim(n_u, min(1024 / n_u, N)); // BlockDim.x = n_u, BlockDim.y adjusted to fit within 1024 threads/block

    // Calculate the number of blocks in the grid
    dim3 gridDim((N + blockDim.y - 1) / blockDim.y, 1); // GridDim.x determines how many blocks are needed to cover all rows

    // Shared memory size: n_u * N elements (since each block will need this much space)
    int sharedMemSize = n_u * N * sizeof(float);

    // Launch the kernel
    computeZhat<<<gridDim, blockDim, sharedMemSize>>>(M_G, w_v, g_P, zhat_v_cpu, N, n_u, m);


    // Synchronize
    cudaDeviceSynchronize();

    // Copy results back to host
    float* zhat_v_gpu = (float*)malloc(length * sizeof(float));
    cudaMemcpy(zhat_v_gpu, d_zhat_v, length * sizeof(float), cudaMemcpyDeviceToHost);

    // Validate GPU result
    if (checkAnswer(exp_zhat, zhat_v_gpu, length)) {
        std::cout << "CUDA result is correct" << std::endl;
    } else {
        std::cout << "CUDA result is incorrect" << std::endl;
    }

    // Free memory
    free(M_G);
    free(w_v);
    free(g_P);
    free(exp_zhat);
    free(zhat_v_cpu);
    free(zhat_v_gpu);

    cudaFree(d_M_G);
    cudaFree(d_w_v);
    cudaFree(d_g_P);
    cudaFree(d_zhat_v);

    return 0;
}
