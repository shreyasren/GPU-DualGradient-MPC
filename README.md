# GPU-Accelerated Dual Gradient Projection Algorithm for Embedded Linear Model Predictive Control

This repository contains a GPU-accelerated implementation of the **Gradient Projection Algorithm for Dual (GPAD)** for solving embedded **Linear Model Predictive Control (MPC)** problems. Our approach builds upon the work of Patrinos' and Bemporads' "Simple and Certifiable Quadratic Programming Algorithms for Embedded Linear Model Predictive Control" by leveraging parallel computation on GPUs to achieve significant speedups compared to a sequential CPU implementation. 

## 🚀 Features
- **GPU-accelerated first-order optimization** for real-time embedded MPC applications.
- **Comparison with a CPU implementation** to evaluate computational performance.
- **CUDA-based parallelization** of the GPAD algorithm.

## 📖 Background
Model Predictive Control (MPC) is widely used in embedded systems for real-time decision-making under constraints. The GPAD method is a first-order gradient-based approach that efficiently solves the dual problem of MPC formulations. By offloading computations to a GPU, our implementation reduces solution time, making MPC more feasible for resource-constrained embedded systems.

## 🛠️ Implementation Details
- **Programming Languages:** CUDA C/C++
- **Optimization Approach:** First-order dual gradient projection
- **Target Hardware:** NVIDIA GPUs (tested on [specific hardware, if relevant])

## 📂 Repository Structure
