#include <stdlib.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "kernel_functions.h"
#define COMP_EPSILON 1e-8


__global__ void StepOneGPADKernel(float *y_vec_in, float *y_vec_minus_1_in, float *w_vec_out, float beta_v, int m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m)
    {
        w_vec_out[i] = y_vec_in[i] + beta_v * (y_vec_in[i] - y_vec_minus_1_in[i]);
    }
}

__global__ void StepTwoGPADKernel(const float* __restrict__ M_G, float* w_v, float* g_P, float* zhat, int N, int n_u, int m) 
{
	extern __shared__ float w_vs[]; 
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int total_elements = N * n_u;
	int index = 0; 
	while(threadIdx.x + index < m){
		w_vs[threadIdx.x + index] = w_v[threadIdx.x + index]; // coalesced memory accesses 
		index += blockDim.x; 
	}
	__syncthreads(); 
	
    if (i < total_elements) {
        float sum = 0.0;
        for (int j = 0; j < m; j++) {
            sum += M_G[j + i * m] * w_vs[j];
        }
        zhat[i] = sum - g_P[i];
    }
}

__global__ void StepThreeGPADKernel(float theta, float* zhat_v, float* z_v, int length) {
    
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) {
        z_v[i] = z_v[i] + theta * (zhat_v[i] - z_v[i]);
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
			y_vp1[i] = 0.5*(sum + abs(sum)); // max without control divergence
		}
	}

}

/*
__global__ void StepFourGPADParChunks(const float* __restrict__ G_L, float* y_vp1, float* w_v, const float* p_D, float* zhat_v, const int N, const int n_u, const int m, int max_threads){
		
	// launch m threads to compute a unique element of the output vector 
	
	extern __shared__ float zhat_vs[];
	int row = threadIdx.y + blockDim.y * blockIdx.y; 
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
			y_vp1[i] = 0.5*(sum + abs(sum)); // max without control divergence
		}
	}

}


__global__ void StepFourGPADParElements(const float* __restrict__ G_L, float* y_vp1, float* w_v, float* zhat_v, const int N, const int n_u, const int m) {

    // launch threads to compute a unique element of the output vector 
    extern __shared__ float zhat_vs[];
    extern __shared__ float sums[]; // dynamically allocate based on blockDim.y
    
    int row = (threadIdx.y + blockIdx.y * blockDim.y) << 1; 
    int col = (threadIdx.x + blockIdx.x * blockDim.x) << 1; 
    int numcols_G_L = n_u * N;
    
    // Each block has its own chunk of zhat_v in shared memory to reduce memory overhead 
    if (col < numcols_G_L) {
        zhat_vs[threadIdx.x] = zhat_v[col]; // coalesced memory accesses
		zhat_vs[threadIdx.x + 1] = zhat_v[col + 1]; 
    }
    __syncthreads(); 
    
    // handle out of bounds
    if (col < numcols_G_L && row < m) {
		float sum = G_L[row * numcols_G_L + col] * zhat_vs[threadIdx.x << 1] + G_L[row * numcols_G_L + col] * zhat_vs[threadIdx.x << 1];
        atomicAdd(&sums[threadIdx.y], G_L[row * numcols_G_L + col] * zhat_vs[threadIdx.x]); // each thread computes the product and adds it to the shared variable
    }
    __syncthreads(); 
    
    if (threadIdx.x == 0 && row < m) { 
        atomicAdd(&y_vp1[row], sums[threadIdx.y]); 
    }
}
*/
	
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

__global__ void DeviceArrayCopy(float* dest, float* src, int size){
	
	int index = threadIdx.x + blockIdx.x * blockDim.x; 
	if (index < size){ dest[index] = src[index]; }
}

