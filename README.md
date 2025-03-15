# GPU-Accelerated Dual Gradient Projection Algorithm for Embedded Linear Model Predictive Control

This repository contains a GPU-accelerated implementation of the **Gradient Projection Algorithm for Dual (GPAD)** for solving embedded **Linear Model Predictive Control (MPC)** problems. Our approach leverages parallel computation on GPUs to achieve significant speedups compared to a sequential CPU implementation. 

This work is based on our research, which has been accepted for presentation at an upcoming conference.

## 🚀 Features
- **GPU-accelerated first-order optimization** for real-time embedded MPC applications.
- **Comparison with a CPU implementation** to evaluate computational performance.
- **CUDA-based parallelization** of the GPAD algorithm.
- **Optimized memory management** to enhance efficiency on embedded platforms.

## 📖 Background
Model Predictive Control (MPC) is widely used in embedded systems for real-time decision-making under constraints. The GPAD method is a first-order gradient-based approach that efficiently solves the dual problem of MPC formulations. By offloading computations to a GPU, our implementation reduces solution time, making MPC more feasible for resource-constrained embedded systems.

## 🛠️ Implementation Details
- **Programming Languages:** CUDA, C/C++, Python (for benchmarking and analysis)
- **Optimization Approach:** First-order dual gradient projection
- **Target Hardware:** NVIDIA GPUs (tested on [specific hardware, if relevant])
- **Comparison Metrics:** Solution time, convergence rate, numerical stability

## 🔬 Performance Comparison
We compare our GPU-based GPAD solver against a sequential CPU implementation, evaluating:
- Speedup factors achieved through GPU parallelization
- Scalability with problem size
- Convergence characteristics

## 📂 Repository Structure
