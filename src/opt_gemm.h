#pragma once

enum OptGemmOp {
  OptGemmOp_N,
  OptGemmOp_T
};

enum OptGemmElemType {
  OptGemmHalf,
  OptGemmFloat,
};

extern "C" {
void gemm(int M, int N, int K,
          float alpha, float beta,
          const void* A, int ldA, OptGemmOp opA,
          const void* B, int ldB, OptGemmOp opB,
          void* C, int ldC);
}