
class CudaBackend {
public:
  CudaBackend() {}

  void gemm(int M, int N, int K,
            float alpha, float beta,
            const void* A, int ldA,
            const void* B, int ldB,
            void* C, int ldC);
};