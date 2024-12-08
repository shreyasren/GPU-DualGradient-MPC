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

__global__ void StepTwoGPADKernel(
	const float* 
	#ifdef ENABLE_CONST_CACHE 
	__restrict__ 
	#endif 
	M_G, 
	float* w_v, 
	float* g_P, 
	float* zhat, 
	int N, 
	int n_u, 
	int m) 
{
	#ifdef ENABLE_SHARED_MEM
	extern __shared__ float w_vs[]; 
	#endif 
	
    int index = blockIdx.x * blockDim.x + threadIdx.x; 
    int total_elements = N * n_u;
	int i; 
	
	#ifdef ENABLE_SHARED_MEM 
	for(i = threadIdx.x; i < m; i+= blockDim.x){
		w_vs[i] = w_v[i]; // coalesced memory accesses
	}
	__syncthreads();  
	#endif 
	
	for (i = index; i < total_elements; i += gridDim.x*blockDim.x){
		float sum = 0.0f; 
		for (int j = 0; j < m; j++)
		{	
			#ifdef ENABLE_SHARED_MEM
			#ifdef ENABLE_FLIPPING
			sum += M_G[j*total_elements + i] * w_vs[j]; // flipping the matrices results in coalesced memory accesses
			#else 
			sum += M_G[j + i * total_elements] * w_vs[j];
			#endif 
			#else 
			#ifdef ENABLE_FLIPPING
			sum += M_G[j*total_elements + i] * w_v[j];
			#else
			sum += M_G[j + i * total_elements] * w_v[j];	
			#endif 
			#endif 
		}
		zhat[i] = sum - g_P[i];
	}
}

__global__ void StepThreeGPADKernel(float theta, float* zhat_v, float* z_v, int length) {
    
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) {
        z_v[i] = (1 - theta)*z_v[i] + theta * (zhat_v[i]);
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
	
__global__ void StepFourGPADFlippedParRows(
	const float* 
	#ifdef ENABLE_CONST_CACHE 
	__restrict__ 
	#endif 
	G_L, 
	float* y_vp1, 
	float* w_v, 
	const float* p_D, 
	float* zhat_v, 
	const int N, 
	const int n_u, 
	const int m, 
	int max_threads){
	
	// launch m threads to compute a unique element of the output vector 
	#ifdef ENABLE_SHARED_MEM
	extern __shared__ float zhat_vs[];  
	#endif 
	int index = threadIdx.x + blockIdx.x*blockDim.x; 
	int numcols_G_L = n_u*N;
	int i; 
	
	#ifdef ENABLE_SHARED_MEM
	// collaborate in loading the shared memory with m threads
	for(i = threadIdx.x; i < numcols_G_L; i+= blockDim.x){
		zhat_vs[i] = zhat_v[i]; // coalesced memory accesses
	}
	__syncthreads(); 
	#endif 
	
	// handle out of bounds 
	for (i = index; i < m; i += gridDim.x*blockDim.x){
		float sum = 0.0f; 
		for (int j = 0; j < numcols_G_L; j++)
		{	
			#ifdef ENABLE_SHARED_MEM
			#ifdef ENABLE_FLIPPING
			sum += G_L[j*m + i]*zhat_vs[j]; // flipping the matrices results in coalesced memory accesses
			#else 
			sum += G_L[i*numcols_G_L + j]*zhat_vs[j];
			#endif 
			#else 
			#ifdef ENABLE_FLIPPING
			sum += G_L[j*m + i]*zhat_v[j];
			#else
			sum += G_L[i*numcols_G_L + j]*zhat_v[j];
			#endif 
			#endif 
		}
		sum += w_v[i] + p_D[i]; // sum 
		
		#ifdef ENABLE_TH_CONVERGENCE
		y_vp1[i] = 0.5*(sum + abs(sum)); // max without control divergence
		#else
		y_vp1[i] = (sum < COMP_EPSILON) ? 0 : sum;
		#endif 
	}
}

inline __device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void mv_warpReduceNoRollOverGranular(const float* __restrict__ a, const float* __restrict__ b, float* c, int m, int n) {
    // designed for datasets with more then 1024 columns
    extern __shared__ float warpSum[]; // Dynamic shared memory for warp sums

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int rowLength = ((n + blockDim.x - 1) / blockDim.x) * blockDim.x;
    int row = (blockIdx.x / ((n + blockDim.x - 1) / blockDim.x)) * 4;
    int col = tid % (rowLength);
    int lane = tid % 32;   // Lane index within the warp
    int warpIdx = (col / 32); // Warp index within the block
    int warpIdxLocal = (warpIdx % 32) % (blockDim.x / 32); // warp index within a block 0 - 31

    float bLocal = b[col];
    for (int phase = 0; phase < 4; phase++){
        if (row + phase > m) return; // Bounds check for rows

        float localVal = 0.0f;

        if (tid < 32) warpSum[tid] = 0.0f;

        if (col < n) {
            localVal += a[(row + phase) * n + col] * bLocal;
        }

        // Warp-level reduction
        localVal = warpReduceSum(localVal);

        // Store results in shared memory
        if (lane == 0) warpSum[warpIdxLocal] = localVal;

        __syncthreads();

        // Reduce warp results to a single value
        if (warpIdxLocal == 0 ) {
            float sum = 0.0f;
            localVal = (lane < 32) * warpSum[lane];
            sum = warpReduceSum(localVal);

            if (lane == 0) atomicAdd(&c[row + phase], sum);
        }
    }
}

__global__ void vvs(float* a, float* b, float* c, int n) {

    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    if (tid < n) {
        c[tid] = a[tid] - b[tid];
    }
}

__global__ void DeviceArrayCopy(float* dest, float* src, int size){
	
	int index = threadIdx.x + blockIdx.x * blockDim.x; 
	if (index < size){ dest[index] = src[index]; }
}

