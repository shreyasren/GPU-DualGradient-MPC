#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "seq_functions.h"
#include "kernel_functions.h"

#define EPSILON 1e-6
#define COMP_EPSILON 1e-8
//#define ENABLE_FLATTEN_MATRICES
#define ENABLE_FLIPPED_MATRICES

// Function to print a vector
void printVector(const float* vec, int size, const char* name) {
    printf("%s: [", name);
    for (int i = 0; i < size; ++i) {
        printf("%f", vec[i]);
        if (i < size - 1) printf(", ");
    }
    printf("]\n");
}

// Function to print a matrix
void printMatrix(const float* mat, int rows, int cols, const char* name) {
    printf("%s:\n", name);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%f ", mat[i * cols + j]);
        }
        printf("\n");
    }
}

void readData(const char *filename, int *n_u, int *N, int *m, int *num_iterations, float *L, float **M_G, float **g_P, float **G_L, float **p_D, float **theta, float **beta){
	
	// open files and read 
	FILE* file = fopen(filename, "r"); 
	
	if (fscanf(file, "%d %d %d %d %f", n_u, N, m, num_iterations, L) != 5) {
        perror("Error reading data");
        fclose(file);
    }
	
	#ifdef ENABLE_FLATTEN_MATRICES
        *M_G = (float*)calloc((*N) * (*m), sizeof(float));
		for(int cnt = 0; cnt < (*N)*(*m); cnt++) fscanf(file, "%f", &(*M_G)[cnt]);
    #else
        *M_G = (float*)calloc((*N) * (*n_u) * (*m), sizeof(float));
		for(int cnt = 0; cnt < (*N)*(*n_u)*(*m); cnt++) fscanf(file, "%f", &(*M_G)[cnt]);
    #endif
	
	*g_P = (float*)calloc((*N) * (*n_u), sizeof(float)); 
	for(int cnt = 0; cnt < (*N) * (*n_u); cnt++) fscanf(file, "%f", &(*g_P)[cnt]); 
	
	#ifdef ENABLE_FLATTEN_MATRICES
        *G_L = (float*)calloc((*N) * (*m), sizeof(float));
		for(int cnt = 0; cnt < (*N) * (*m) cnt++) fscanf(file, "%f", &(*G_L)[cnt]);
    #else
        *G_L = (float*)calloc((*N) * (*n_u) * (*m), sizeof(float));
		for(int cnt = 0; cnt < (*N) * (*n_u) * (*m); cnt++) fscanf(file, "%f", &(*G_L)[cnt]);
    #endif
	
	*p_D = (float*)calloc(*m, sizeof(float)); 
	for(int cnt = 0; cnt < *m; cnt++) fscanf(file, "%f", &(*p_D)[cnt]); 
	
	*theta = (float*)calloc(*num_iterations, sizeof(float)); 
	for (int cnt = 0; cnt < *num_iterations; cnt++) fscanf(file, "%f", &(*theta)[cnt]); 
	*beta = (float*)calloc(*num_iterations, sizeof(float)); 
	for (int cnt = 0; cnt < *num_iterations; cnt++) fscanf(file, "%f", &(*beta)[cnt]);
	fclose(file);
	
}

void initializeVariables(float **y_v, float **y_vp1, float **w_v, float **zhat_v, float **z_v, int n_u, int N, int m){
	
	*y_v = (float*)calloc(m, sizeof(float)); 
	*y_vp1 = (float*)calloc(m, sizeof(float)); 
	*w_v = (float*)calloc(m, sizeof(float)); 
	*zhat_v = (float*)calloc(n_u*N, sizeof(float)); 
	*z_v = (float*)calloc(n_u*N, sizeof(float)); 
	
}

int main(){
	
	// System variables 
	int n_u;
	int N;
	int m;
	int num_iterations; 
	float L; 
	
	// Constants computed off-line 
	float *M_G; 
	float *g_P; 
	float *G_L;
	float *p_D;
	float *theta; 
	float *beta; 
	
	// Variables computed on-line  
	float *y_vp1;
	float *y_v; 
	float *w_v;
	float *zhat_v;
	float *z_v; 
	
	// Read in data from text file and initialize variables 
	char filename[256];	
	snprintf(filename, sizeof(filename), "inputs_gpad/input_big.txt");
	readData(filename, &n_u, &N, &m, &num_iterations, &L, &M_G, &g_P, &G_L, &p_D, &theta, &beta); 
	initializeVariables(&y_v, &y_vp1, &w_v, &zhat_v, &z_v, n_u, N, m); 
	
	/*
	printMatrix(M_G, n_u*N, m, "M_G"); 
	printMatrix(G_L, m, n_u*N, "G_L"); 
	printMatrix(g_P, n_u*N, 1, "g_P"); 
	printMatrix(p_D, m, 1, "p_D"); 
	printMatrix(theta, num_iterations, 1, "theta"); 
	printMatrix(beta, num_iterations, 1, "beta");
	printMatrix(y_vp1, m, 1, "y_vp1"); 
	*/
	
	
	// Write algorithm here!
	int v = 0; 
	struct timeval gpu_start, gpu_finish; 
	long gpu_exectime = 0;
	
	// STEP 1
	float *dy_vp1; 
	float *dy_v; 
	float *dM_G;
	float *dg_P;
	float *dw_v; 
	float *dz_v; 
	float *dzhat_v;
	float* dp_D;
	float* dG_L;
	cudaMalloc((void **)&dy_vp1, m * sizeof(float)); 
	cudaMalloc((void **)&dy_v, m * sizeof(float)); 
	cudaMalloc((void **)&dM_G, N * n_u * m * sizeof(float));
	cudaMalloc((void **)&dg_P, N * n_u * sizeof(float));
	cudaMalloc((void **)&dw_v, m * sizeof(float)); 
	cudaMalloc((void **)&dz_v, n_u * N * sizeof(float)); 
	cudaMalloc((void **)&dzhat_v, n_u * N * sizeof(float));
	cudaMalloc((void**)&dp_D, m*sizeof(float));
	#ifdef ENABLE_FLATTEN_MATRICES
		cudaMalloc((void**)&dG_L, N*m*sizeof(float)); 
		cudaMemcpy(dG_L, G_L, N*m*sizeof(float), cudaMemcpyHostToDevice);
	#else 
		cudaMalloc((void**)&dG_L, N*n_u*m*sizeof(float)); 
		cudaMemcpy(dG_L, G_L, N*n_u*m*sizeof(float), cudaMemcpyHostToDevice);
	#endif
	cudaMemcpy(dy_vp1, y_vp1, m * sizeof(float), cudaMemcpyHostToDevice); 
	cudaMemcpy(dy_v, y_v, m * sizeof(float), cudaMemcpyHostToDevice); 
	cudaMemcpy(dM_G, M_G, N * n_u * m * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dg_P, g_P, N * n_u * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dz_v, z_v, n_u * N * sizeof(float), cudaMemcpyHostToDevice); 
	cudaMemcpy(dzhat_v, zhat_v, n_u * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dp_D, p_D, m*sizeof(float), cudaMemcpyHostToDevice);
	
	dim3 gridDimStep1(ceil((float)m / (float)256.0f), 1 , 1);
    dim3 blockDimStep1(256, 1, 1);
	dim3 gridDimStep2(ceil((float)(n_u * N * 0.25) / (float)256.0f), 1 , 1);
	dim3 blockDimStep2(256, 1, 1);
	dim3 gridDimStep3(ceil((float)n_u*N / 256.0f), 1, 1); 
	dim3 blockDimStep3(256, 1, 1);  
	dim3 gridDimStep4((int)ceil((float)m/256.0), 1, 1);
	dim3 blockDimStep4(256, 1, 1); 
	dim3 gridDimCopy((int)ceil((float)m/256.0), 1, 1);
	dim3 blockDimCopy(256, 1, 1); 
	
	for (int i = 0; i < 100; i++){
		gettimeofday(&gpu_start, NULL);
		// STEP 1
		StepOneGPADKernel<<<gridDimStep1, blockDimStep1>>>(dy_vp1, dy_v, dw_v, beta[v], m); 
		cudaDeviceSynchronize();
		// STEP 2 & COPY STEP
		StepTwoGPADKernel<<<gridDimStep2, blockDimStep2, m*sizeof(float)>>>(dM_G, dw_v, dg_P, dzhat_v, N, n_u, m); 
		DeviceArrayCopy<<<gridDimCopy, blockDimCopy>>>(dy_v, dy_vp1, m); 
		cudaDeviceSynchronize(); 
		// STEP 3 & STEP 4 
		StepThreeGPADKernel<<<gridDimStep3, blockDimStep3>>>(theta[v], dzhat_v, dz_v, n_u*N); 
		StepFourGPADFlippedParRows<<<gridDimStep4, blockDimStep4, n_u*N*sizeof(float)>>>(dG_L, dy_vp1, dw_v, dp_D, dzhat_v, N, n_u, m, 3660);
		cudaDeviceSynchronize(); 
		gettimeofday(&gpu_finish, NULL);
		gpu_exectime += abs((long)gpu_finish.tv_usec - (long)gpu_start.tv_usec);
	}
	cudaMemcpy(y_vp1, dy_vp1, m * sizeof(float), cudaMemcpyDeviceToHost); 
	cudaMemcpy(z_v, dz_v, n_u * N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(w_v, dw_v, m * sizeof(float), cudaMemcpyDeviceToHost);
	//printMatrix(w_v, m, 1, "w_v"); 
	//printMatrix(z_v, n_u*N, 1, "z_v"); 
	//printMatrix(y_vp1, m, 1, "p_D");
	cudaFree(dy_vp1); 
	cudaFree(dy_v); 
	cudaFree(dw_v); 
	cudaFree(dz_v); 
	cudaFree(dzhat_v);
	cudaFree(dp_D);
	
	printf("Avg. GPU Execution Time over %d trial(s) = %lu usec\n", 100, gpu_exectime/100);
	
	// Free memory from the heap
	free(M_G);  
	free(g_P);
	free(zhat_v); 
	free(w_v); 
	free(p_D); 
	free(G_L); 
	free(y_vp1);
	free(theta);
	free(beta);
	free(z_v); 	
	
	return 0; 
}