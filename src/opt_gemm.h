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
  OptGemmOp OpA;
  OptGemmOp OpB;
  OptGemmOp OpC;

public:
  GemmShape(int M, int N, int K, OptGemmOp OpA, OptGemmOp OpB, OptGemmOp OpC) :
    M(M), N(N), K(K), OpA(OpA), OpB(OpB), OpC(OpC) {}
  GemmShape(OptGemmOp OpA, OptGemmOp OpB, OptGemmOp OpC) :
    M(0), N(0), K(0), OpA(OpA), OpB(OpB), OpC(OpC) {}

  int m() const {return M;}
  int n() const {return N;}
  int k() const {return K;}
  OptGemmOp opA() const {return OpA;}
  OptGemmOp opB() const {return OpB;}
  OptGemmOp opC() const {return OpC;}

  bool operator==(const GemmShape& shape2) const {
    return m() == shape2.m() &&
           n() == shape2.n() &&
           k() == shape2.k() &&
           opA() == shape2.opA() &&
           opB() == shape2.opB() &&
           opC() == shape2.opC();
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