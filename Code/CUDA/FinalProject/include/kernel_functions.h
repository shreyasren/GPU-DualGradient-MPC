#ifndef KERNEL_FUNCTIONS_H
#define KERNEL_FUNCTIONS_H
#define ENABLE_SHARED_MEM
#define ENABLE_CONST_CACHE
#define ENABLE_TH_CONVERGENCE
#define ENABLE_FLIPPING
//#define ENABLE_OPTIMIZED

__global__ void StepOneGPADKernel(float *y_vec_in, float *y_vec_minus_1_in, float *w_vec_out, float beta_v, int m);
__global__ void StepTwoGPADKernel(const float* 
                                    #ifdef ENABLE_CONST_CACHE
                                    __restrict__ 
                                    #endif
                                    M_G, 
                                    float* w_v, 
                                    float* g_P, 
                                    float* zhat, 
                                    int N, 
                                    int n_u, 
                                    int m);
__global__ void StepThreeGPADKernel(float theta, float* zhat_v, float* z_v, int length);
__global__ void StepFourGPADFlatParRows(const float* __restrict__ G_L, float* y_vp1, float* w_v, const float* __restrict__ p_D, float* zhat_v, const int N, const int n_u, const int m);
__global__ void StepFourGPADParRows(const float* __restrict__ G_L, float* y_vp1, float* w_v, const float* p_D, float* zhat_v, const int N, const int n_u, const int m, int max_threads);
__global__ void StepFourGPADParElements(const float* __restrict__ G_L, float* y_vp1, float* w_v, float* zhat_v, const int N, const int n_u, const int m);
__global__ void StepFourGPADAddMax(float* y_vp1, float* w_v, const float* __restrict__ p_D, const int m);
__global__ void StepFourGPADFlippedParRows(const float* 
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
                                            int max_threads);
__global__ void DeviceArrayCopy(float* dest, float* src, int size);
__global__ void mv_warpReduceNoRollOverGranular(const float* __restrict__ a, const float* __restrict__ b, float* c, int m, int n);
__global__ void vvs(float* a, float* b, float* c, int n);

#endif 