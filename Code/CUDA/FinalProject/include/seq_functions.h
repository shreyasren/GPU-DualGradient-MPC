#ifndef SEQ_FUNCTIONS_H
#define SEQ_FUNCTIONS_H

#ifdef __cplusplus
extern "C" {
#endif 

void StepTwoGPADFlatSequential(const float* M_G, float* w_v, const float* g_P, float* zhat_v, const int N, const int n_u, const int m);
void StepFourGPADFlatSequential(const float* G_L, float* y_vp1, float* w_v, const float* p_D, float* zhat_v, const int N, const int n_u, const int m);
void StepTwoGPADSequential(const float* M_G, float* w_v, const float* g_P, float* zhat_v, const int N, const int n_u, const int m);
void StepFourGPADSequential(const float* G_L, float* y_vp1, float* w_v, const float* p_D, float* zhat_v, const int N, const int n_u, const int m);

#ifdef __cplusplus
}
#endif 

#endif // SEQ_FUNCTIONS_H