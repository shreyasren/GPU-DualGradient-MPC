#include <stdlib.h>
#include "seq_functions.h"

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
		y_vp1[i] = (sum + abs(sum))/2; 
	}
}