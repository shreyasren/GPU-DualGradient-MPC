#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define EPSILON 1e-6
//#define STEP2
#define STEP4 

__global__ void StepTwoGPADParallelCols(const float* __restrict__ M_G, const float* __restrict__ g_P, float* w_v, float* zhat_v, int N, int n_u, int m){
	
	// insert parallel code here
}

__global__ void StepFourGPADParallelCols(const float* __restrict__ G_L, float* y_vp1, float* w_v, const float* __restrict__ p_D, float* zhat_v, int N, int n_u, int m){
	
	// insert parallel code here
}

// Sequential implementation of Step 2
void StepTwoGPADSequential(const float* M_G, float* w_v, const float* g_P, float* zhat_v, const int N, const int n_u, const int m){
	
	for (int i = 0; i < N; i++){
		for (int j = 0; j < n_u; j++){ 
			float sum = 0.0; 
			for (int k = j; k < 4*n_u*N; k += n_u){
				sum += M_G[i*m + k]*w_v[k];  
			}
			for (int k = 4*n_u*N; k < m; k++){
				sum += M_G[i*m + k]*w_v[k]; 
			}
			zhat_v[i*n_u + j] = sum - g_P[i*n_u + j];
		}
	}

}

// Sequential implementation of Step 4
void StepFourGPADSequential(const float* G_L, float* y_vp1, float* w_v, const float* p_D, float* zhat_v, const int N, const int n_u, const int m){
	
	for (int i = 0; i < m; i++){
		float sum = 0.0f; 
		for (int j = 0; j < N; j++){
			if (i < 4*n_u*N){
				sum += G_L[i*N + j]*zhat_v[j*n_u + (i%n_u)]; 
			}
			else{
				for(int k = 0; k < n_u; k++){
					sum += G_L[i*N + j]*zhat_v[j*n_u + k]; 
				}
			}
		}
		y_vp1[i] = sum + w_v[i] + p_D[i]; 
	}
	
	for(int i = 0; i < m; i++){
		if(y_vp1[i] < 0) y_vp1[i] = 0;
	}
}


int main(int argc, char *argv[]){
	
	if (argc != 2) {
        fprintf(stderr, "Usage: %s <subfolder number>\n", argv[0]);
        return 1;
    }
	
	int n_u, N, m; 
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
	#ifdef STEP2	
	char inpfile2[256];
	char outpfile2[256];
	snprintf(inpfile2, sizeof(inpfile2), "step2/%s/input.txt", subfolder_number);
	snprintf(outpfile2, sizeof(outpfile2), "step2/%s/output.txt", subfolder_number);
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
	float *M_G = (float*)calloc(N*m, sizeof(float)); 
	float *w_v = (float*)calloc(m, sizeof(float)); 
	float *g_P = (float*)calloc(N*n_u, sizeof(float));
	float *zhat_v = (float*)calloc(N*n_u, sizeof(float)); 
	float *prod_Mw = (float*)calloc(N*n_u, sizeof(float)); 
	float *exp_prod_Mw = (float*)calloc(N*n_u, sizeof(float)); 
	float *exp_zhat = (float*)calloc(N*n_u, sizeof(float));
	
	// Populate input matrices 
	for(cnt = 0; cnt < N*m; cnt++) fscanf(file, "%f", &M_G[cnt]); 
	for(cnt = 0; cnt < m; cnt++) fscanf(file, "%f", &w_v[cnt]);
	for(cnt = 0; cnt < N*n_u; cnt++) fscanf(file, "%f", &g_P[cnt]);
	
	// Print out matrices to verify correct loading
	printf("\nM_G:\n"); 
	for(int i = 0; i < N; i++){
		for(int j = 0; j < m; j++){
			printf("%f ", M_G[i*m + j]); 
		}
		printf("\n"); 
	}
	
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
	
	// Fetching data for step 4
	#ifdef STEP4
	char inpfile4[256];
	char outpfile4[256];
	snprintf(inpfile4, sizeof(inpfile4), "step4/%s/input.txt", subfolder_number);
	snprintf(outpfile4, sizeof(outpfile4), "step4/%s/output.txt", subfolder_number);
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
	float *G_L = (float*)calloc(N*m, sizeof(float));
	float *y_vp1 = (float*)calloc(m, sizeof(float));
	float *prod_Gz = (float*)calloc(m, sizeof(float)); 
	float *exp_prod_Gz = (float*)calloc(m, sizeof(float)); 
	float *exp_sum = (float*)calloc(m, sizeof(float)); 
	float *exp_y_vp1 = (float*)calloc(m, sizeof(float));
	
	// Populate input matrices 
	for(cnt = 0; cnt < m; cnt++) fscanf(file, "%f", &w_v[cnt]);
	for(cnt = 0; cnt < N*n_u; cnt++) fscanf(file, "%f", &zhat_v[cnt]);
	for(cnt = 0; cnt < m; cnt++) fscanf(file, "%f", &p_D[cnt]); 
	for(cnt = 0; cnt < N*m; cnt++) fscanf(file, "%f", &G_L[cnt]);
	
	// Print out matrices to verify correct loading
	printf("\nG_L:\n"); 
	for(int i = 0; i < m; i++){
		for(int j = 0; j < N; j++){
			printf("%.12f ", G_L[i*N + j]); 
		}
		printf("\n"); 
	}
	
	printf("\np_D:\n"); 
	for(int j = 0; j < m; j++){
		printf("%.12f\n", p_D[j]); 
	}
    // Close the file
    fclose(file);
	
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
	#ifdef STEP2
	StepTwoGPADSequential(M_G, w_v, g_P, zhat_v, N, n_u, m);
	
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
	#ifdef STEP4 
	StepFourGPADSequential(G_L, y_vp1, w_v, p_D, zhat_v, N, n_u, m);
	
	int status_success = 0; 
	int fail_index = 0; 
	printf("\n Computed difference for Step 4 : \n"); 
	for (int i = 0; i < m; i++){
		printf("%.12f\n", abs(y_vp1[i] - exp_y_vp1[i])); 
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
	
	#endif 
	// end algorithm 
	
	
	// Free memory from the heap
	#ifdef STEP2
	free(M_G); 
	free(w_v); 
	free(g_P);
	free(zhat_v); 
	free(prod_Mw); 
	free(exp_prod_Mw); 
	free(exp_zhat);
	#endif 
	
	#ifdef STEP4
	free(w_v); 
	free(zhat_v); 
	free(p_D); 
	free(G_L); 
	free(y_vp1); 
	free(prod_Gz); 
	free(exp_prod_Gz); 
	free(exp_y_vp1);	
	#endif 
	
	return 0; 
}