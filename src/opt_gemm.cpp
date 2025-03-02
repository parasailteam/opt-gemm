#include<cstdio>
#include<unordered_map>

#include "opt_gemm.h"
#include "cuda/cuda_backend.h"

CudaBackend cudaBackend;

std::size_t std::hash<GemmShape>::operator()(const GemmShape& problem) const {
  std::size_t h = hash<int>()(problem.m()) ^ hash<int>()(problem.k()) ^ 
                  hash<int>()(problem.n());
  return h;
}

void gemm(int M, int N, int K,
          float alpha, float beta,
          const void* A, int ldA, OptGemmOp opA,
          const void* B, int ldB, OptGemmOp opB,
          void* C, int ldC) {
  cudaBackend.gemm(GemmShape(M, N, K), 
                   alpha, beta,
                   A, ldA, opA,
                   B, ldB, opB,
                   C, ldC);
}