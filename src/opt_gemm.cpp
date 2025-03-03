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

std::string strOfOptGemmElemType(OptGemmElemType elemType) {
  if (elemType == OptGemmFloat) {
    return "f32";
  } else if (elemType == OptGemmHalf) {
    return "f16";
  }
}

std::string strOfOptGemmOp(OptGemmOp op) {
  if (op == OptGemmOp_N) {
    return "N";
  } else if (op == OptGemmOp_T) {
    return "T";
  }
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