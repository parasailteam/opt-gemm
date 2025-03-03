#include "gemm_kernel.h"

enum CudaArch {
  CudaArchAmpere = 80,
  CudaArchHopper = 90
};

class CudaGemmKernel : public GemmKernel {
  CudaArch arch;

public:
  CudaGemmKernel(CudaArch arch,
                 OptGemmElemType elemA, OptGemmElemType elemB,
                 OptGemmElemType elemC, OptGemmElemType elemAccum,
                 OptGemmOp opA, OptGemmOp opB, OptGemmOp opC,
                 int CtaM, int CtaN, int CtaK,
                 int WarpM, int WarpN, int WarpK,
                 int InstM, int InstN, int InstK,
                 int NumStages, int SplitKSlices) :
  arch(arch),
  GemmKernel(BackendCUDA, elemA, elemB, elemC, elemAccum,
             opA, opB, opC,
             CtaM, CtaN, CtaK,
             WarpM, WarpN, WarpK,
             InstM, InstN, InstK,
             NumStages, SplitKSlices) {}

  virtual std::string str() {
    std::string archStr = "";
    if (arch == CudaArchAmpere)      archStr = "ampere";
    else if (arch == CudaArchHopper) archStr = "hopper";

    return archStr + GemmKernel::str();
  }
};