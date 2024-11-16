#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define EPSILON 1e-7
#define STEP1
//#define STEP2
//#define STEP4

#define CUDA_CHECK(ans){ gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void CheckNullError(void *ptr, const char *string, const char *file, int line);
void PrintInputFileDataSeqStep1(double *y_vec_in, double *y_vec_minus_1_in, const int n_u, const int N, const int m, const double beta_v);
void CheckOutputDataStep1(double *reference_out, double *calculate_out, const int m);
__global__ void KernelStep1(double *y_vec_in, double *y_vec_minus_1_in, double *w_vec_out, double beta_v, int m);

int main(int argc, char *argv[])
{
    int n_u;
    int N;
    int m;
	int cnt = 0;
    char inpfile1[256];
	char inpfile3[256];
	char outpfile1[256]; 
	char outpfile3[256];
    double beta_v;            /* STEP1: size: 1x1 matrix */
    double *y_vec_in;         /* STEP1: size: 1x1 matrix */
    double *y_vec_minus_1_in; /* STEP1: size: mx1 matrix */
    double *w_vec_out;        /* STEP1: size: mx1 matrix */
    double *w_vec_out_ref;    /* STEP1 */ 
	FILE *file; 
    char *subfolder_number;
    
    /*****************/
    /* CMD LINE ARGS */
    /*****************/
	if (argc != 2)
    {
        fprintf(stderr, "Usage: %s <subfolder number>\n", argv[0]);
        exit(EXIT_FAILURE);;
    }
    else
    {
        subfolder_number = argv[1];
        snprintf(inpfile1, sizeof(inpfile1), "FinalProject/build/step1/%s/input.txt", subfolder_number);
        snprintf(outpfile1, sizeof(outpfile1), "FinalProject/build/step1/%s/output.txt", subfolder_number);
        snprintf(inpfile3, sizeof(inpfile3), "step3/%s/input.txt", subfolder_number);
        snprintf(outpfile3, sizeof(outpfile3), "step3/%s/output.txt", subfolder_number);
    }
	
    // Fetch Data from step 1
#ifdef STEP1    
    file = fopen(inpfile1, "r");
    CheckNullError(file, "OPENING_INPUT_FILE step1", __FILE__, __LINE__);
    
    if (fscanf(file, "%d %d %d %lf", &n_u, &N, &m, &beta_v) != 4)
    {
        perror("Error reading data");
        fclose(file);
    }

    // Dym mem allocation
    y_vec_in         = (double *)calloc(m, sizeof(double));
    y_vec_minus_1_in = (double *)calloc(m, sizeof(double));
    w_vec_out        = (double *)calloc(m, sizeof(double));
    w_vec_out_ref    = (double *)calloc(m, sizeof(double));

    CheckNullError(y_vec_in        , "ALLOCATION y_vec_in"        , __FILE__, __LINE__);
    CheckNullError(y_vec_minus_1_in, "ALLOCATION y_vec_minus_1_in", __FILE__, __LINE__);
    CheckNullError(w_vec_out       , "ALLOCATION w_vec_out"       , __FILE__, __LINE__);
    CheckNullError(w_vec_out_ref   , "ALLOCATION w_vec_out_ref"   , __FILE__, __LINE__);

    // Get vector data from input file
    for (cnt = 0; cnt < m; cnt++) fscanf(file, "%lf", &y_vec_in[cnt]);
    for (cnt = 0; cnt < m; cnt++) fscanf(file, "%lf", &y_vec_minus_1_in[cnt]);

    //PrintInputFileDataSeqStep1(y_vec_in, y_vec_minus_1_in, n_u, N, m, beta_v);
    fclose(file);

    file = fopen(outpfile1, "r");
    CheckNullError(file, "OPENING_OUTPUT_FILE step1", __FILE__, __LINE__);
    
    // Get vector data from output file
    for (cnt = 0; cnt < m; cnt++) fscanf(file, "%lf", &w_vec_out_ref[cnt]);
    
    fclose(file);
#endif /* STEP1 */
	
	
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
	float *w_v4 = (float*)calloc(m, sizeof(float)); 
	float *zhat_v4 = (float*)calloc(N*n_u, sizeof(float)); 
	float *p_D = (float*)calloc(m, sizeof(float)); 
	float *G_L = (float*)calloc(N*m, sizeof(float));
	float *y_vp1 = (float*)calloc(m, sizeof(float));
	float *prod_Gz = (float*)calloc(m, sizeof(float)); 
	float *exp_prod_Gz = (float*)calloc(m, sizeof(float)); 
	float *exp_sum = (float*)calloc(m, sizeof(float)); 
	float *exp_y_vp1 = (float*)calloc(m, sizeof(float));
	
	// Populate input matrices 
	for(cnt = 0; cnt < m; cnt++) fscanf(file, "%f", &w_v4[cnt]);
	for(cnt = 0; cnt < N*n_u; cnt++) fscanf(file, "%f", &zhat_v4[cnt]);
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
#ifdef STEP1    
    // Seq
    for (int i = 0; i < m; i++)
    {
        w_vec_out[i] = y_vec_in[i] + beta_v * (y_vec_in[i] - y_vec_minus_1_in[i]);
    }
    CheckOutputDataStep1(w_vec_out_ref, w_vec_out, m);

    // Parallel
    double *device_y_vec_in;
    double *device_y_vec_minus_1_in;
    double *device_w_vec_out;

    CUDA_CHECK(cudaMalloc((void **)&device_y_vec_in, m * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&device_y_vec_minus_1_in, m * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&device_w_vec_out, m * sizeof(double)));
    
    CUDA_CHECK(cudaMemcpy(device_y_vec_in, y_vec_in, m * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_y_vec_minus_1_in, y_vec_minus_1_in, m * sizeof(double), cudaMemcpyHostToDevice));

    dim3 gridDimStep1(ceil((float)m / (float)32), 1 , 1);
    dim3 blockDimStep1(32, 1, 1);

    KernelStep1<<<gridDimStep1, blockDimStep1>>>(device_y_vec_in, device_y_vec_minus_1_in, device_w_vec_out, beta_v, m);

    cudaDeviceSynchronize();

    CUDA_CHECK(cudaMemcpy(w_vec_out, device_w_vec_out, m * sizeof(double), cudaMemcpyDeviceToHost));

    CheckOutputDataStep1(w_vec_out_ref, w_vec_out, m);
   
#endif /* STEP1 */
	
	// STEP 2: zhat_v <-- M_G * w_v - g_P
	#ifdef STEP2
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
	for (int i = 0; i < m; i++){
		float sum = 0.0f; 
		for (int j = 0; j < N; j++){
			if (i < 4*n_u*N){
				sum += G_L[i*N + j]*zhat_v4[j*n_u + (i%n_u)]; 
			}
			else{
				for(int k = 0; k < n_u; k++){
					sum += G_L[i*N + j]*zhat_v4[j*n_u + k]; 
				}
			}
		}
		y_vp1[i] = sum + w_v4[i] + p_D[i]; 
	}
	
	for(int i = 0; i < m; i++){
		if(y_vp1[i] < 0) y_vp1[i] = 0;
	}
	
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
#ifdef STEP1
    free(w_vec_out_ref);
    free(w_vec_out);
    free(y_vec_minus_1_in);
    free(y_vec_in);
#endif STEP1 /* STEP1 */
    
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
	free(w_v4); 
	free(zhat_v4); 
	free(p_D); 
	free(G_L); 
	free(y_vp1); 
	free(prod_Gz); 
	free(exp_prod_Gz); 
	free(exp_y_vp1);	
	#endif 
	
	return 0; 
}

void CheckNullError(void *ptr, const char *string, const char *file, int line)
{
    if (ptr == NULL)
    {
        printf("error: %s, file: %s, line num: %d\n", string, file, line);
        exit(EXIT_FAILURE);
    }
    else
    {
        /* no error, do nothing */
    }
}

void PrintInputFileDataSeqStep1(double *y_vec_in, double *y_vec_minus_1_in, const int n_u, const int N, const int m, const double beta_v)
{
    printf("INPUT DATA: STEP 1\n");
    printf("%d ", n_u);
    printf("%d ", N);
    printf("%d ", m);
    printf("%0.8lf\n", beta_v);

    for (int i = 0; i < m; i++)
    {
        printf("%0.8lf\n", y_vec_in[i]);
    }

    for (int i = 0; i < m; i++)
    {
        printf("%0.8lf\n", y_vec_minus_1_in[i]);
    }

    printf("\n");
}

void CheckOutputDataStep1(double *reference_out, double *calculate_out, const int m)
{
    unsigned int cnt_good;
    unsigned int cnt_bad;

    cnt_good = 0;
    cnt_bad = 0;
    
    printf("OUTPUT RESULTS\n");
    
    for (int i = 0; i < m; i++)
    {
        if (fabs(reference_out[i] - calculate_out[i]) <= EPSILON)
        {
            cnt_good++;

            if (cnt_good == m) printf("result: correct :)\n");
        }
        else
        {
            if (cnt_bad == 0) printf("result: incorrect :(\n");
            
            printf("idx: %d, actual: %0.8lf, calculated: %0.8lf\n", i, reference_out[i], calculate_out[i]);
            cnt_bad++;
        }
    }
}

__global__ void KernelStep1(double *y_vec_in, double *y_vec_minus_1_in, double *w_vec_out, double beta_v, int m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < m)
    {
        w_vec_out[i] = y_vec_in[i] + beta_v * (y_vec_in[i] - y_vec_minus_1_in[i]);
    }
    else
    {
        /* extra thread, do nothing */
    }  
}
