#include "cuda/ampere/kernels/kernel_decl.h"
#include "cuda/cuda_backend.h"
#include "utils/logger.h"

void CudaBackend::gemm(GemmShape gemmShape,
                       float alpha, float beta,
                       const void* A, int ldA, OptGemmOp opA,
                       const void* B, int ldB, OptGemmOp opB,
                       void* C, int ldC) {
  CudaGemmKernel* kernel;

  auto it = GemmShapeToAmpereKernel.find(gemmShape);
  if (it == GemmShapeToAmpereKernel.end()) {
    it = GemmShapeToAmpereKernel.find(GemmShape(opA, opB, OptGemmOp_N));
  }

  Logger(LogLevel::Info) << "Launching " << it->second->str() << std::endl;

  return it->second->launch(gemmShape.m(), gemmShape.n(), gemmShape.k(),
                                     alpha, beta,
                                     A, ldA,
                                     B, ldB,
                                     C, ldC,
                                     nullptr);
}