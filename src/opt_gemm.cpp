#include<cstdio>

#include "opt_gemm.h"
#include "cuda/cuda_backend.h"

CudaBackend cudaBackend;

void gemm(int M, int N, int K,
          float alpha, float beta,
          const void* A, int ldA,
          const void* B, int ldB,
          void* C, int ldC) {
  cudaBackend.gemm(M, N, K, 
                   alpha, beta,
                   A, ldA,
                   B, ldB,
                   C, ldC);
}