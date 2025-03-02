
class CudaBackend {
public:
  CudaBackend() {}

  void gemm(GemmShape shape,
            float alpha, float beta,
            const void* A, int ldA, OptGemmOp opA,
            const void* B, int ldB, OptGemmOp opB,
            void* C, int ldC);
};