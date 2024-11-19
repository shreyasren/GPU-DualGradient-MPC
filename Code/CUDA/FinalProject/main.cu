#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#define EPSILON 1e-6
#define COMP_EPSILON 1e-8
//#define FLATTEN_MATRICES
#define ENABLE_GPU
//#define STEP2
#define STEP4
#define num_states 900

#ifdef FLATTEN_MATRICES
__global__ void StepTwoGPADParallelCols(const float* __restrict__ M_G, const float* __restrict__ g_P, float* w_v, float* zhat_v, int N, int n_u, int m){
	
	// insert parallel code here
}

__global__ void StepFourGPADFlatParRows(const float* __restrict__ G_L, float* y_vp1, float* w_v, const float* __restrict__ p_D, float* zhat_v, const int N, const int n_u, const int m){
	
	// launch m threads to compute a unique element of the output vector 
	
	extern __shared__ float zhat_vs[]; 
	int tx = threadIdx.x; 
	int bx = blockIdx.x; 
	int index = tx + bx*blockDim.x; 
	
	// collaborate in loading the shared memory with m threads
	int i = 0; 
	while(tx + i < num_states){
		zhat_vs[tx + i] = zhat_v[tx + i]; // coalesced memory accesses 
		i += blockDim.x; 
	}
	__syncthreads(); 
	
	// handle out of bounds 
	if(index < m){
		float sum = 0.0f; 
		float yraw = 0.0f; 
		for (int j = 0; j < N; j++){
			if (index < 4*n_u*N){
				sum += G_L[index*N + j]*zhat_vs[j*n_u + (index % n_u)]; 
			}
			else{
				for(int k = 0; k < n_u; k++){
					sum += G_L[index*N + j]*zhat_vs[j*n_u + k]; 
				}
			}
		}
		yraw = sum + w_v[index] + p_D[index]; // sum 
		
		y_vp1[index] = (yraw < COMP_EPSILON) ? 0 : yraw; // projection onto nonnegative orthant
	}

}

/*
__global__ void StepFourGPADParElements(const float* __restrict__ G_L, float* y_vp1, float* w_v, const float* __restrict__ p_D, float* zhat_v, const int N, const int n_u, const int m){
	
	// launch m*N threads to compute a unique element of the output vector 
	__shared__ float zhat_vs[blockDim.x]; 
	int tx = threadIdx.x; int ty = threadIdx.y; 
	int bx = blockIdx.x; int by = blockIdx.y; 
	int row = by*blockDim.y + ty; 
	int col = bx*blockDim.x + tx; 
	
	// collaborate in loading the shared memory with m threads
	int i = 0; 
	while(tx + i < num_states){
		zhat_vs[tx + i] = zhat_v[tx + i]; // coalesced memory accesses 
		i += blockDim.x; 
	}
	__syncthreads(); 
	
	// handle out of bounds 
	if(index < m){
		float sum = 0.0f; 
		float yraw = 0.0f; 
		for (int j = 0; j < N; j++){
			if (index < 4*n_u*N){
				sum += G_L[index*N + j]*zhat_vs[j*n_u + (index % n_u)]; 
			}
			else{
				for(int k = 0; k < n_u; k++){
					sum += G_L[index*N + j]*zhat_vs[j*n_u + k]; 
				}
			}
		}
		yraw = sum + w_v[index] + p_D[index]; // sum 
		
		y_vp1[index] = (yraw < COMP_EPSILON) ? 0 : yraw; // projection onto nonnegative orthant
	}

}
*/
__global__ void ParPositiveProjection(float* vec, int size){
	
	int index = threadIdx.x + blockDim.x*blockIdx.x; 
	if (index < size){
		if (vec[index] < COMP_EPSILON) vec[index] = 0.0f; 
	}
	
}

// Sequential implementation of Step 2
void StepTwoGPADFlatSequential(const float* M_G, float* w_v, const float* g_P, float* zhat_v, const int N, const int n_u, const int m){
	
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
void StepFourGPADFlatSequential(const float* G_L, float* y_vp1, float* w_v, const float* p_D, float* zhat_v, const int N, const int n_u, const int m){
	
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
#endif 

#ifndef FLATTEN_MATRICES
	// Sequential implementation of Step 2 over unflattened matrices 
	void StepTwoGPADSequential(const float* M_G, float* w_v, const float* g_P, float* zhat_v, const int N, const int n_u, const int m){
		
		int numrows_M_G = n_u*N;
		int numcols_M_G = m; 
		for (int i = 0; i < numrows_M_G; i++){
			float sum = 0.0f; 
			for(int j = 0; j < numcols_M_G; j++){
				sum += M_G[i*numcols_M_G + j]*w_v[j]; 
			}
			zhat_v[i] = sum - g_P[i]; 
		}

	}

	// Sequential implementation of Step 4 over unflattened matrices 
	void StepFourGPADSequential(const float* G_L, float* y_vp1, float* w_v, const float* p_D, float* zhat_v, const int N, const int n_u, const int m){
		
		int numrows_G_L = m;
		int numcols_G_L = n_u*N;
		for (int i = 0; i < numrows_G_L; i++){
			float sum = 0.0f; 
			for (int j = 0; j < numcols_G_L; j++){
				sum += G_L[i*numcols_G_L + j]*zhat_v[j]; 
			}
			sum += w_v[i] + p_D[i]; 
			if (sum < COMP_EPSILON) { y_vp1[i] = 0.0f; }
			else { y_vp1[i] = sum; } 
		}
	}
	
	
	__global__ void StepFourGPADParRows(const float* G_L, float* y_vp1, float* w_v, const float* __restrict__ p_D, float* zhat_v, const int N, const int n_u, const int m){
		
		// launch m threads to compute a unique element of the output vector 
		
		extern __shared__ float zhat_vs[];
		int tx = threadIdx.x; 
		int bx = blockIdx.x; 
		int index = tx + bx*blockDim.x; 
		int numcols_G_L = n_u*N; 
		
		// collaborate in loading the shared memory with m threads
		for(int i = tx; i < numcols_G_L; i+= blockDim.x){
			zhat_vs[i] = zhat_v[i]; // coalesced memory accesses
		}
		__syncthreads(); 
		
		// handle out of bounds 
		if(index < m){
			float sum = 0.0f; 
			for (int j = 0; j < numcols_G_L; j++){
				sum += G_L[index*numcols_G_L + j]*zhat_vs[j]; 
			}
			sum += w_v[index] + p_D[index]; // sum 
			y_vp1[index] = (sum < COMP_EPSILON) ? 0 : sum; // projection onto nonnegative orthant
		}

	}
	
	__global__ void ParDotProduct(float* matrix, float* vector, float* result, int row_index, int size){
		
		extern __shared__ float prod[]; 
		int index = threadIdx.x + blockIdx.x*blockDim.x; 
		int tx = threadIdx.x; 
		
		float sum = 0.0f; 
		int i = 0; 
		// coalesced memory accesses by the threads 
		while (index + i < size){
			sum += matrix[row_index + index + i]*vector[index + i];
			i += blockDim.x; 
		}
		// store the sum computed by each thread in the shared memory and synch the threads 
		prod[tx] = sum; 
		__syncthreads(); 
		
		// reduction step 
		for (unsigned int stride = blockDim.x >> 1; stride >= 1; stride >>= 1){
			if (tx < stride) prod[tx] += prod[tx + stride]; 
			__syncthreads(); 
		}
		
		// write result to output 
		if (tx == 0){
			result[row_index] = prod[0]; 
		}
	}
	
	__global__ void StepFourGPADDynamicParRows(const float* G_L, float* y_vp1, float* w_v, const float* __restrict__ p_D, float* zhat_v, const int N, const int n_u, const int m){
		
		int tx = threadIdx.x; 
		int bx = blockIdx.x; 
		int index = tx + bx*blockDim.x; 
		int numcols_G_L = n_u*N; 
		const int block_size = (int)(min(1024.0, (float)n_u*N)); 
		float sum = 0.0f; 
		
		// handle out of bounds 
		if(index < m){
			ParDotProduct<<<1, block_size, block_size*sizeof(float)>>>(G_L, zhat_v, y_vp1, index*numcols_G_L, numcols_G_L); 
		}
		cudaDeviceSynchronize(); // wait for all child kernels to complete execution 
		
		sum = y_vp1[index] + w_v[index] + p_D[index]; 
		y_vp1[index] = (sum < COMP_EPSILON) ? 0 : sum; // projection onto nonnegative orthant
	}
	
	/*
	__global__ void StepFourGPADParChunks(const float* G_L, float* y_vp1, float* w_v, const float* __restrict__ p_D, float* zhat_v, const int N, const int n_u, const int m){
		
		// launch 32*m threads to compute a unique element of the output vector 
		
		extern __shared__ float zhat_vs[];
		int tx = threadIdx.x; int ty = threadIdx.y; 
		int bx = blockIdx.x; int by = blockIdx.y; 
		int col = tx + bx*blockDim.x; 
		int row = ty + by*blockDim.y; 
		int numrows_G_L = m; 
		int numcols_G_L = n_u*N; 
		
		// collaborate in loading the shared memory with m threads
		for(int i = tx; i < numcols_G_L; i+= blockDim.x){
			zhat_vs[i] = zhat_v[i]; // coalesced memory accesses
		}
		__syncthreads(); 
		
		// handle out of bounds 
		if(row < numrows_G_L && col < numcols_G_L){
			float result = 0.0f;
			for(int j = col; j < numcols_G_L; j += blockDim.x){
				result += G_L[row*numcols_G_L + j]*zhat_vs[j]; 
			}
			atomicAdd(&y_vp1[row], result); 
			__syncthreads(); 
			
			if(col == 0){
				sum_results += 
			}
			sum += w_v[index] + p_D[index]; // sum 
			y_vp1[index] = (sum < COMP_EPSILON) ? 0 : sum; // projection onto nonnegative orthant
		}

	}
	*/
	
#endif 

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

int main(int argc, char *argv[]){
	
	if (argc != 2) {
        fprintf(stderr, "Usage: %s <subfolder number>\n", argv[0]);
        return 1;
    }
	
	int n_u, N, m; 
	int num_trials = 100; 
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
	#ifdef FLATTEN_MATRICES
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
	#ifdef FLATTEN_MATRICES
		float *M_G = (float*)calloc(N*m, sizeof(float));
		for(cnt = 0; cnt < N*m; cnt++) fscanf(file, "%f", &M_G[cnt]);
	#else
		float *M_G = (float*)calloc(N*n_u*m, sizeof(float));
		for(cnt = 0; cnt < N*n_u*m; cnt++) fscanf(file, "%f", &M_G[cnt]);
	#endif  
	float *w_v = (float*)calloc(m, sizeof(float)); 
	float *g_P = (float*)calloc(N*n_u, sizeof(float));
	float *zhat_v = (float*)calloc(N*n_u, sizeof(float)); 
	float *prod_Mw = (float*)calloc(N*n_u, sizeof(float)); 
	float *exp_prod_Mw = (float*)calloc(N*n_u, sizeof(float)); 
	float *exp_zhat = (float*)calloc(N*n_u, sizeof(float));
	
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
	
	// Fetching data for step 4
	#ifdef STEP4
	char inpfile4[256];
	char outpfile4[256];
	#ifdef FLATTEN_MATRICES
		snprintf(inpfile4, sizeof(inpfile4), "step4/%s/input.txt", subfolder_number);
		snprintf(outpfile4, sizeof(outpfile4), "step4/%s/output.txt", subfolder_number);
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
    #ifdef FLATTEN_MATRICES
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
	#ifdef STEP2
	
	#ifdef FLATTEN_MATRICES
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
	#ifdef STEP4 
	
	// GPU memory allocation and transfer
	#ifdef ENABLE_GPU
	float* dG_L;
	float* dy_vp1; 
	float* dw_v; 
	float* dp_D; 
	float* dzhat_v; 
	#ifdef FLATTEN_MATRICES
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
	dim3 bd((int)min(1024.0, (float)m), 1, 1); 
	dim3 gd((int)ceil((float)m/1024.0), 1, 1); 
	printf("Block Dimensions: (%d, %d, %d)\n", bd.x, bd.y, bd.z); 
	printf("Grid Dimensions: (%d, %d, %d)\n", gd.x, gd.y, gd.z);
	
	// Kernel launch and synchronize
	unsigned long gpu_exectime = 0; 
	unsigned int ovfl_cnt = 0; 
	for (int i = 0; i < num_trials; i++){
		gettimeofday(&gpu_start, NULL);
		#ifdef FLATTEN_MATRICES
			StepFourGPADFlatParRows<<<gd, bd, n_u*N*sizeof(float)>>>(dG_L, dy_vp1, dw_v, dp_D, dzhat_v, N, n_u, m);
		#else
			//StepFourGPADParRows<<<gd, bd, n_u*N*sizeof(float)>>>(dG_L, dy_vp1, dw_v, dp_D, dzhat_v, N, n_u, m);
			//StepFourGPADDynamicParRows<<<gd, bd, n_u*N*sizeof(float)>>>(dG_L, dy_vp1, dw_v, dp_D, dzhat_v, N, n_u, m);
		#endif 			
		cudaDeviceSynchronize();
		gettimeofday(&gpu_finish, NULL);
		if (gpu_finish.tv_usec - gpu_start.tv_usec > 1e8){
			ovfl_cnt++; 
		}
		else{
			gpu_exectime += gpu_finish.tv_usec - gpu_start.tv_usec;
		}
		
	}
	cudaMemcpy(y_vp1, dy_vp1, m*sizeof(float), cudaMemcpyDeviceToHost);
	printf("Avg. GPU Execution Time over %d trial(s) = %lu usec\n", num_trials - ovfl_cnt, gpu_exectime/(num_trials - ovfl_cnt)); 
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
		#ifdef FLATTEN_MATRICES
			StepFourGPADFlatSequential(G_L, y_vp1, w_v, p_D, zhat_v, N, n_u, m);
		#else 
			StepFourGPADSequential(G_L, y_vp1, w_v, p_D, zhat_v, N, n_u, m);
		#endif 
	}
	gettimeofday(&cpu_finish, NULL); 
	
	printf("Avg. CPU Execution Time over %d trial(s) = %lu usec\n", num_trials, (cpu_finish.tv_usec - cpu_start.tv_usec)/num_trials); 
	
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	printf("Compute Capability: %d.%d\n", prop.major, prop.minor);

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