#include <stdlib.h>
#include "kernel_functions.h"
#define COMP_EPSILON 1e-8

__global__ void update_z(float theta, float* z_vm1, float* zhat_v, float* z_v, int length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) {
        z_v[i] = (1 - theta) * z_vm1[i] + theta * zhat_v[i];
    }
}

__global__ void StepFourGPADFlatParRows(const float* __restrict__ G_L, float* y_vp1, float* w_v, const float* __restrict__ p_D, float* zhat_v, const int N, const int n_u, const int m){
	
	// launch m threads to compute a unique element of the output vector 
	
	extern __shared__ float zhat_vs[]; 
	int tx = threadIdx.x; 
	int bx = blockIdx.x; 
	int index = tx + bx*blockDim.x; 
	
	// collaborate in loading the shared memory with m threads
	int i = 0; 
	while(tx + i < n_u*N){
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

__global__ void StepFourGPADParRows(const float* __restrict__ G_L, float* y_vp1, float* w_v, const float* p_D, float* zhat_v, const int N, const int n_u, const int m, int max_threads){
		
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
		
	if (index < max_threads){	
		// handle out of bounds 
		for (int i = index; i < m; i += gridDim.x*blockDim.x){
			float sum = 0.0f; 
			for (int j = 0; j < numcols_G_L; j++)
			{
				sum += G_L[i*numcols_G_L + j]*zhat_vs[j]; 
			}
			sum += w_v[i] + p_D[i]; // sum 
			y_vp1[i] = (sum + abs(sum))/2; // max without control divergence
		}
	}

}
	
__global__ void StepFourGPADFlippedParRows(const float* __restrict__ G_L, float* y_vp1, float* w_v, const float* p_D, float* zhat_v, const int N, const int n_u, const int m, int max_threads){
	
	// launch m threads to compute a unique element of the output vector 
	
	extern __shared__ float zhat_vs[];  
	int index = threadIdx.x + blockIdx.x*blockDim.x; 
	int numcols_G_L = n_u*N;
	
	// collaborate in loading the shared memory with m threads
	for(int i = threadIdx.x; i < numcols_G_L; i+= blockDim.x){
		zhat_vs[i] = zhat_v[i]; // coalesced memory accesses
	}
	__syncthreads(); 
		
	if (index < max_threads){	
		// handle out of bounds 
		for (int i = index; i < m; i += gridDim.x*blockDim.x){
			float sum = 0.0f; 
			for (int j = 0; j < numcols_G_L; j++)
			{
				sum += G_L[j*m + i]*zhat_vs[j]; // flipping the matrices results in coalesced memory accesses
			}
			sum += w_v[i] + p_D[i]; // sum 
			y_vp1[i] = (sum + abs(sum))/2; // max without control divergence
			//y_vp1[i] = (sum < COMP_EPSILON) ? 0 : sum;
		}
	}
}

