#include <unordered_map>
#include <string>

#pragma once

enum OptGemmOp {
  OptGemmOp_N,
  OptGemmOp_T
};

enum OptGemmElemType {
  OptGemmHalf,
  OptGemmFloat,
};

class GemmShape {
  int M, N, K;
public:
  GemmShape(int M, int N, int K) : M(M), N(N), K(K) {}
  GemmShape() : M(0), N(0), K(0) {}

  int m() const {return M;}
  int n() const {return N;}
  int k() const {return K;}

  bool operator==(const GemmShape& shape2) const {
    return m() == shape2.m() &&
           n() == shape2.n() &&
           k() == shape2.k();
  }
};

template<>
struct std::hash<GemmShape> {
  std::size_t operator()(const GemmShape& k) const;
};

std::string strOfOptGemmElemType(OptGemmElemType elemType);
std::string strOfOptGemmOp(OptGemmOp op);

extern "C" {
void gemm(int M, int N, int K,
          float alpha, float beta,
          const void* A, int ldA, OptGemmOp opA,
          const void* B, int ldB, OptGemmOp opB,
          void* C, int ldC);
}