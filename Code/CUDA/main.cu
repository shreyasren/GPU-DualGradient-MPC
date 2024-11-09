#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]){
	
	if (argc != 3) {
        fprintf(stderr, "Usage: %s <step number> <subfolder number>\n", argv[0]);
        return 1;
    }
	
	int n_u, N, m; 
	int cnt = 0; 
	FILE *file; 
	char *step_number = argv[1];
    char *subfolder_number = argv[2];
	char inpfile[256];
	char outpfile[256]; 
    snprintf(inpfile, sizeof(inpfile), "%s/%s/input.txt", step_number, subfolder_number);
	snprintf(outpfile, sizeof(inpfile), "%s/%s/output.txt", step_number, subfolder_number);
	
	file = fopen(inpfile, "r"); 
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
	for(int i = 0; i < N; i++){
		for(int j = 0; j < m; j++){
			printf("%.8f ", M_G[i*m + j]); 
		}
		printf("\n"); 
	}
    // Close the file
    fclose(file);
	
	file = fopen(outpfile, "r"); 
	if (file == NULL){
		perror("Error opening file!");
		return 1; 
	}
	
	for(cnt = 0; cnt < N*n_u; cnt++) fscanf(file, "%f", &exp_prod_Mw[cnt]); 
	for(cnt = 0; cnt < N*n_u; cnt++) fscanf(file, "%f", &exp_zhat[cnt]);

	// Print out matrices to verify correct loading 
	printf("\n The expected product M_G * w_v: \n"); 
	for (int i = 0; i < N*n_u; i++){
		printf("%.8f\n", exp_prod_Mw[i]); 
	}
	
	printf("\n The expected output z_hat: \n");
	for (int i = 0; i < N*n_u; i++){
		printf("%.8f\n", exp_zhat[i]); 
	}
	
	// Close the file 
	fclose(file); 

    // Print the values to verify
    printf("n_u = %d, N = %d, m = %d\n", n_u, N, m);
	
	// Write algorithm here! 
	
	
	
	
	
	// end algorithm 
	
	
	// Free memory from the heap
	free(M_G); 
	free(w_v); 
	free(g_P);
	free(zhat_v); 
	free(prod_Mw); 
	free(exp_prod_Mw); 
	free(exp_zhat); 
	
	return 0; 
}