#include "cuda/ampere/kernels/kernel_decl.h"

#include "cuda/cuda_backend.h"

void CudaBackend::gemm(int M, int N, int K,
                       float alpha, float beta,
                       const void* A, int ldA, OptGemmOp opA,
                       const void* B, int ldB, OptGemmOp opB,
                       void* C, int ldC) {
  return AllAmpereKernels[0]->launch(M, N, K,
                                     alpha, beta,
                                     A, ldA,
                                     B, ldB,
                                     C, ldC,
                                     nullptr);
}