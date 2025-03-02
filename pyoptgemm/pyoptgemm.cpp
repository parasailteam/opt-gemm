#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "opt_gemm.h"

namespace py = pybind11;

#define THROW_ERROR(err) {\
  if ((err) != fastKronSuccess) {\
    throw std::runtime_error(std::string("FastKronError: ") + std::string(fastKronGetErrorString(err)));\
  }\
}

PYBIND11_MODULE(OptGemm, m)
{
  m.doc() = "";
  // OptGemmElemType

  py::enum_<OptGemmOp>(m, "Op", py::module_local())
    .value("OpN", OptGemmOp_N)
    .value("OpT", OptGemmOp_T)
    .export_values();

  m.def("hgemm", [](int M, int N, int K,
                    float alpha, float beta,
                    long A, int ldA, OptGemmOp opA,
                    long B, int ldB, OptGemmOp opB,
                    long C, int ldC) {
    gemm(M, N, K, alpha, beta,
         (const void*)A, ldA, opA,
         (const void*)B, ldB, opB,
         (void*)C, ldC);
  }, "HGEMM");
}