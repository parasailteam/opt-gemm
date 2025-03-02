#pragma once

enum OptGemmOp {
  OptGemmOp_N,
  OptGemmOp_T
};

enum OptGemmElemType {
  OptGemmHalf,
  OptGemmFloat,
};

void gemm(int M, int N, int K,
          float alpha, float beta,
          const void* A, int ldA,
          const void* B, int ldB,
          void* C, int ldC);