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
#define ENABLE_GPU
//#define ENABLE_STEP2
#define ENABLE_STEP4

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

int readData(const char *filename, int *n_u, int *N, int *m, float **M_G, float **w_v, float **g_P){
	
	// open files and read 
	FILE* file = fopen(filename, "r"); 
	
	if (fscanf(file, "%d %d %d", n_u, N, m) != 3) {
        perror("Error reading data");
        fclose(file);
        return 1;
    }
	
	#ifdef ENABLE_FLATTEN_MATRICES
        *M_G = (float*)calloc((*N) * (*m), sizeof(float));
		for(int cnt = 0; cnt < N*m; cnt++) fscanf(file, "%f", &M_G[cnt]);
    #else
        *M_G = (float*)calloc((*N) * (*n_u) * (*m), sizeof(float));
		for(int cnt = 0; cnt < N*n_u*m; cnt++) fscanf(file, "%f", &M_G[cnt]);
    #endif
	
}

int main(int argc, char *argv[]){
	
	// List of pointer declarations
	float *w_v; 
	float *g_P;
	float *zhat_v; 
	float *prod_Mw; 
	float *p_D; 
	float *y_vp1;
	float *prod_Gz;
	float *G_L;
	
	// Pointers for expected results 
	float *exp_prod_Mw; 
	float *exp_zhat;
	float *exp_prod_Gz; 
	float *exp_sum 
	float *exp_y_vp1;
	
	if (argc != 2) {
        fprintf(stderr, "Usage: %s <subfolder number>\n", argv[0]);
        return 1;
    }
	
	int n_u, N, m; 
	int num_trials = 10; 
	int max_num_threads = 5000; 
	int cnt = 0; 
	FILE *file; 
    char *subfolder_number = argv[1];
	
	char inpfile1[256];
	char inpfile3[256];
	char outpfile1[256]; 
	char outpfile3[256];
    snprintf(inpfile1, sizeof(inpfile1), "step1/%s/input.txt", subfolder_number);
	snprintf(outpfile1, sizeof(outpfile1), "step1/%s/output.txt", subfolder_number);
	snprintf(inpfile3, sizeof(inpfile3), "step3/%s/input.txt", subfolder_number);
	snprintf(outpfile3, sizeof(outpfile3), "step3/%s/output.txt", subfolder_number);
	
	
	// Fetching data for step 2
	#ifdef ENABLE_STEP2	
	char inpfile2[256];
	char outpfile2[256];
	#ifdef ENABLE_FLATTEN_MATRICES
		snprintf(inpfile2, sizeof(inpfile2), "step2/%s/input.txt", subfolder_number);
		snprintf(outpfile2, sizeof(outpfile2), "step2/%s/output.txt", subfolder_number);
	#else
		snprintf(inpfile2, sizeof(inpfile2), "step2/%s_unflat/input.txt", subfolder_number);
		snprintf(outpfile2, sizeof(outpfile2), "step2/%s_unflat/output.txt", subfolder_number);
	#endif 
	
	file = fopen(inpfile2, "r"); 
	if (file == NULL){
		perror("Error opening file!"); 
		return 1; 
	}
	
	if (fscanf(file, "%d %d %d", &n_u, &N, &m) != 3) {
        perror("Error reading data");
        fclose(file);
        return 1;
    }
	
	// Dynamically allocate variables we need 
	#ifdef ENABLE_FLATTEN_MATRICES
		M_G = (float*)calloc(N*m, sizeof(float));
		for(cnt = 0; cnt < N*m; cnt++) fscanf(file, "%f", &M_G[cnt]);
	#else
		M_G = (float*)calloc(N*n_u*m, sizeof(float));
		for(cnt = 0; cnt < N*n_u*m; cnt++) fscanf(file, "%f", &M_G[cnt]);
	#endif  
	w_v = (float*)calloc(m, sizeof(float)); 
	g_P = (float*)calloc(N*n_u, sizeof(float));
	zhat_v = (float*)calloc(N*n_u, sizeof(float)); 
	prod_Mw = (float*)calloc(N*n_u, sizeof(float)); 
	exp_prod_Mw = (float*)calloc(N*n_u, sizeof(float)); 
	exp_zhat = (float*)calloc(N*n_u, sizeof(float));
	
	// Populate input matrices 
	for(cnt = 0; cnt < m; cnt++) fscanf(file, "%f", &w_v[cnt]);
	for(cnt = 0; cnt < N*n_u; cnt++) fscanf(file, "%f", &g_P[cnt]);
	
	printf("\nw_v:\n"); 
	for(int j = 0; j < m; j++){
		printf("%f\n", w_v[j]); 
	}
    // Close the file
    fclose(file);
	
	file = fopen(outpfile2, "r"); 
	if (file == NULL){
		perror("Error opening file!");
		return 1; 
	}
	
	for(cnt = 0; cnt < N*n_u; cnt++) fscanf(file, "%f", &exp_prod_Mw[cnt]); 
	for(cnt = 0; cnt < N*n_u; cnt++) fscanf(file, "%f", &exp_zhat[cnt]);

	// Print out matrices to verify correct loading 
	printf("\n The expected product M_G * w_v: \n"); 
	for (int i = 0; i < N*n_u; i++){
		printf("%f\n", exp_prod_Mw[i]); 
	}
	
	printf("\n The expected output z_hat: \n");
	for (int i = 0; i < N*n_u; i++){
		printf("%f\n", exp_zhat[i]); 
	}
	
	// Close the file 
	fclose(file); 
	#endif
	
	#ifdef ENABLE_STEP3
	
	#endif 
	
	// Fetching data for step 4
	#ifdef ENABLE_STEP4
	char inpfile4[256];
	char outpfile4[256];
	#ifdef ENABLE_FLATTEN_MATRICES
		snprintf(inpfile4, sizeof(inpfile4), "step4/%s/input.txt", subfolder_number);
		snprintf(outpfile4, sizeof(outpfile4), "step4/%s/output.txt", subfolder_number);
	#elifdef ENABLE_FLIPPED_MATRICES
		snprintf(inpfile4, sizeof(inpfile4), "step4/%s_flipped/input.txt", subfolder_number);
		snprintf(outpfile4, sizeof(outpfile4), "step4/%s_flipped/output.txt", subfolder_number);
	#else
		snprintf(inpfile4, sizeof(inpfile4), "step4/%s_unflat/input.txt", subfolder_number);
		snprintf(outpfile4, sizeof(outpfile4), "step4/%s_unflat/output.txt", subfolder_number);
	#endif 
	
	file = fopen(inpfile4, "r"); 
	if (file == NULL){
		perror("Error opening file!"); 
		return 1; 
	}
	
	if (fscanf(file, "%d %d %d", &n_u, &N, &m) != 3) {
        perror("Error reading data");
        fclose(file);
        return 1;
    }
	
	// Dynamically allocate variables we need  
	float *w_v = (float*)calloc(m, sizeof(float)); 
	float *zhat_v = (float*)calloc(N*n_u, sizeof(float)); 
	float *p_D = (float*)calloc(m, sizeof(float)); 
	float *y_vp1 = (float*)calloc(m, sizeof(float));
	float *prod_Gz = (float*)calloc(m, sizeof(float)); 
	float *exp_prod_Gz = (float*)calloc(m, sizeof(float)); 
	float *exp_sum = (float*)calloc(m, sizeof(float)); 
	float *exp_y_vp1 = (float*)calloc(m, sizeof(float));
	for(cnt = 0; cnt < m; cnt++) fscanf(file, "%f", &w_v[cnt]);
	for(cnt = 0; cnt < N*n_u; cnt++) fscanf(file, "%f", &zhat_v[cnt]);
	for(cnt = 0; cnt < m; cnt++) fscanf(file, "%f", &p_D[cnt]); 
    #ifdef ENABLE_FLATTEN_MATRICES
		float *G_L = (float*)calloc(N*m, sizeof(float));
		for(cnt = 0; cnt < N*m; cnt++) fscanf(file, "%f", &G_L[cnt]);
	#else
		float *G_L = (float*)calloc(N*n_u*m, sizeof(float));
		for(cnt = 0; cnt < N*n_u*m; cnt++) fscanf(file, "%f", &G_L[cnt]);
	#endif 
	
	// Close the file 
	fclose(file);
	//printVector(w_v, m, "w_v"); 
	//printVector(zhat_v, N * n_u, "zhat_v"); 
	//printVector(p_D, m, "p_D"); 
	//printMatrix(G_L, m, N * n_u, "G_L");
	
	file = fopen(outpfile4, "r"); 
	if (file == NULL){
		perror("Error opening file!");
		return 1; 
	}
	
	for(cnt = 0; cnt < m; cnt++) fscanf(file, "%f", &exp_prod_Gz[cnt]); 
	for(cnt = 0; cnt < m; cnt++) fscanf(file, "%f", &exp_sum[cnt]);
	for(cnt = 0; cnt < m; cnt++) fscanf(file, "%f", &exp_y_vp1[cnt]);
	// Print out matrices to verify correct loading 
	printf("\n The expected product y_vp1: \n"); 
	for (int i = 0; i < m; i++){
		printf("%.12f\n", exp_y_vp1[i]); 
	}
	
	// Close the file 
	fclose(file);
	#endif
	
	
	
	
	
	
	
    // Print the values to verify
    printf("n_u = %d, N = %d, m = %d\n", n_u, N, m);
	
	
	
	
	
	// Write algorithm here! 
	
	// STEP 1: (INSERT HERE)
	
	// STEP 2: zhat_v <-- M_G * w_v - g_P
	#ifdef ENABLE_STEP2
	
	#ifdef ENABLE_FLATTEN_MATRICES
		StepTwoGPADFlatSequential(M_G, w_v, g_P, zhat_v, N, n_u, m);
	#else 
		StepTwoGPADSequential(M_G, w_v, g_P, zhat_v, N, n_u, m); 
	#endif 
	
	int status_success = 0; 
	int fail_index = 0; 
	printf("\n Computed result for Step 2 : \n"); 
	for (int i = 0; i < N*n_u; i++){
		printf("%.12f\n", zhat_v[i]); 
		if (status_success == 0 && abs(zhat_v[i] - exp_zhat[i]) > EPSILON){
			status_success = 1; 
			fail_index = i; 
		}
	}
	
	if(status_success){
		printf("\nStatus: FAILED!\n");
		printf("Failed at index %d\n", fail_index); 
	}
	else{
		printf("\nStatus: SUCCESS!\n"); 
	}
	#endif
	
	// STEP 3: (INSERT HERE)
	
	// STEP 4: y_{v+1} <-- [w_v + G_L * zhat_v + p_D]_+ (INSERT HERE)
	#ifdef ENABLE_STEP4 
	
	// GPU memory allocation and transfer
	#ifdef ENABLE_GPU
	float* dG_L;
	float* dy_vp1; 
	float* dw_v; 
	float* dp_D; 
	float* dzhat_v; 
	#ifdef ENABLE_FLATTEN_MATRICES
		cudaMalloc((void**)&dG_L, N*m*sizeof(float)); 
		cudaMemcpy(dG_L, G_L, N*m*sizeof(float), cudaMemcpyHostToDevice);
	#else 
		cudaMalloc((void**)&dG_L, N*n_u*m*sizeof(float)); 
		cudaMemcpy(dG_L, G_L, N*n_u*m*sizeof(float), cudaMemcpyHostToDevice);
	#endif 
	cudaMalloc((void**)&dy_vp1, m*sizeof(float)); 
	cudaMalloc((void**)&dw_v, m*sizeof(float)); 
	cudaMalloc((void**)&dp_D, m*sizeof(float)); 
	cudaMalloc((void**)&dzhat_v, N*n_u*sizeof(float)); 	
	cudaMemcpy(dw_v, w_v, m*sizeof(float), cudaMemcpyHostToDevice); 
	cudaMemcpy(dp_D, p_D, m*sizeof(float), cudaMemcpyHostToDevice); 
	cudaMemcpy(dzhat_v, zhat_v, N*n_u*sizeof(float), cudaMemcpyHostToDevice); 
	struct timeval gpu_start, gpu_finish; 
	// Grid and block sizes 
	//dim3 bd((int)min(256.0, (float)m), 1, 1); 
	//dim3 gd((int)ceil((float)m/256.0), 1, 1); 
	//printf("Block Dimensions: (%d, %d, %d)\n", bd.x, bd.y, bd.z); 
	//printf("Grid Dimensions: (%d, %d, %d)\n", gd.x, gd.y, gd.z);
	
	file = fopen("block_times.txt", "w"); 
	for(cnt = 1; cnt <= max_num_threads; cnt += 2){
	
	dim3 bd((int)min(256.0, (float)cnt), 1, 1); 
	dim3 gd((int)ceil((float)cnt/256.0), 1, 1);
	printf("Number of threads: %d\n", cnt); 
	
	// Kernel launch and synchronize
	long gpu_exectime = 0; 
	unsigned int ovfl_cnt = 0; 
	for (int i = 0; i < num_trials; i++){
		gettimeofday(&gpu_start, NULL);
		#ifdef ENABLE_FLATTEN_MATRICES
			StepFourGPADFlatParRows<<<gd, bd, n_u*N*sizeof(float)>>>(dG_L, dy_vp1, dw_v, dp_D, dzhat_v, N, n_u, m);
		#elifdef ENABLE_FLIPPED_MATRICES
			StepFourGPADFlippedParRows<<<gd, bd, n_u*N*sizeof(float)>>>(dG_L, dy_vp1, dw_v, dp_D, dzhat_v, N, n_u, m, cnt);
		#else
			StepFourGPADParRows<<<gd, bd, n_u*N*sizeof(float)>>>(dG_L, dy_vp1, dw_v, dp_D, dzhat_v, N, n_u, m, cnt);
			//StepFourGPADDynamicParRows<<<gd, bd, n_u*N*sizeof(float)>>>(dG_L, dy_vp1, dw_v, dp_D, dzhat_v, N, n_u, m);
		#endif 			
		cudaDeviceSynchronize();
		gettimeofday(&gpu_finish, NULL);
		gpu_exectime += abs((long)gpu_finish.tv_usec - (long)gpu_start.tv_usec);
		
	}
	cudaMemcpy(y_vp1, dy_vp1, m*sizeof(float), cudaMemcpyDeviceToHost);
	fprintf(file, "%lu\n", gpu_exectime/(num_trials)); 
	//printf("Avg. GPU Execution Time over %d trial(s) = %lu usec\n", num_trials - ovfl_cnt, gpu_exectime/(num_trials - ovfl_cnt)); 
	#endif 
	
	
	int status_success = 0; 
	int fail_index = 0; 
	//printf("\n Computed y_vp1 for Step 4 : \n"); 
	for (int i = 0; i < m; i++){
		//printf("%.12f\n", abs(y_vp1[i] - exp_y_vp1[i])); 
		//printf("%.12f\n", y_vp1[i]);
		if (status_success == 0 && abs(y_vp1[i] - exp_y_vp1[i]) > EPSILON){
			status_success = 1; 
			fail_index = i; 
		}
	}
	if(status_success){
		printf("\nStatus: FAILED!\n");
		printf("Failed at index %d\n", fail_index); 
	}
	else{
		printf("\nStatus: SUCCESS!\n"); 
	}
	
	struct timeval cpu_start, cpu_finish; 
	gettimeofday(&cpu_start, NULL);
	for (int i = 0; i < num_trials; i++){
		#ifdef ENABLE_FLATTEN_MATRICES
			StepFourGPADFlatSequential(G_L, y_vp1, w_v, p_D, zhat_v, N, n_u, m);
		#else 
			StepFourGPADSequential(G_L, y_vp1, w_v, p_D, zhat_v, N, n_u, m);
		#endif 
	}
	gettimeofday(&cpu_finish, NULL); 
	
	printf("Avg. GPU Execution Time over %d trial(s) = %lu usec\n", num_trials - ovfl_cnt, gpu_exectime/(num_trials - ovfl_cnt));
	printf("Avg. CPU Execution Time over %d trial(s) = %lu usec\n", num_trials, (cpu_finish.tv_usec - cpu_start.tv_usec)/num_trials); 

	#endif 
	// end algorithm 
	}
	fclose(file); 
	
	// Free memory from the heap
	#ifdef ENABLE_STEP2
	free(M_G); 
	free(w_v); 
	free(g_P);
	free(zhat_v); 
	free(prod_Mw); 
	free(exp_prod_Mw); 
	free(exp_zhat);
	#endif 
	
	#ifdef ENABLE_STEP4
	free(w_v); 
	free(zhat_v); 
	free(p_D); 
	free(G_L); 
	free(y_vp1); 
	free(prod_Gz); 
	free(exp_prod_Gz); 
	free(exp_y_vp1);
	#ifdef ENABLE_GPU	
		cudaFree(dG_L); 
		cudaFree(dy_vp1); 
		cudaFree(dw_v); 
		cudaFree(dp_D); 
		cudaFree(dzhat_v);
	#endif 
	#endif 
	
	return 0; 
}