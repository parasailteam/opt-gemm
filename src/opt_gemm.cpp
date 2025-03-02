#include<cstdio>

#include "opt_gemm.h"
#include "cuda/cuda_backend.h"

CudaBackend cudaBackend;

void gemm(int M, int N, int K,
          float alpha, float beta,
          const void* A, int ldA, OptGemmOp opA,
          const void* B, int ldB, OptGemmOp opB,
          void* C, int ldC) {
  cudaBackend.gemm(M, N, K, 
                   alpha, beta,
                   A, ldA, opA,
                   B, ldB, opB,
                   C, ldC);
}