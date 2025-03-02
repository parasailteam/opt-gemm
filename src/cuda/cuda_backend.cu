#include "cuda/ampere/kernels/kernel_decl.h"

#include "cuda/cuda_backend.h"

void CudaBackend::gemm(GemmShape gemmShape,
                       float alpha, float beta,
                       const void* A, int ldA, OptGemmOp opA,
                       const void* B, int ldB, OptGemmOp opB,
                       void* C, int ldC) {
  CudaGemmKernel* kernel;

  auto it = GemmShapeToAmpereKernel.find(gemmShape);
  if (it == GemmShapeToAmpereKernel.end()) {
    it = GemmShapeToAmpereKernel.find(GemmShape());
  }

  return it->second->launch(gemmShape.m(), gemmShape.n(), gemmShape.k(),
                                     alpha, beta,
                                     A, ldA,
                                     B, ldB,
                                     C, ldC,
                                     nullptr);
}