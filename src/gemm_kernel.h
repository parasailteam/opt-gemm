#include <stddef.h>
#include <stdint.h>
#include <sstream>

#include "opt_gemm.h"

#pragma once

enum Backend {
  BackendCUDA
};

class GemmKernel {
protected:
  Backend backend;

  OptGemmElemType elemA;
  OptGemmElemType elemB;
  OptGemmElemType elemC;
  OptGemmElemType elemAccum;

  OptGemmOp opA; OptGemmOp opB; OptGemmOp opC;

  int CtaM;  int CtaN;  int CtaK;
  int WarpM; int WarpN; int WarpK;
  int InstM; int InstN; int InstK;
  
  int NumStages;
  int SplitKSlices;

public:
  GemmKernel(Backend backend,
             OptGemmElemType elemA, OptGemmElemType elemB,
             OptGemmElemType elemC, OptGemmElemType elemAccum,
             OptGemmOp opA, OptGemmOp opB, OptGemmOp opC,
             int CtaM, int CtaN, int CtaK,
             int WarpM, int WarpN, int WarpK,
             int InstM, int InstN, int InstK,
             int NumStages, int SplitKSlices) :
  backend(backend), elemA(elemA), elemB(elemB), elemC(elemC), elemAccum(elemAccum),
  opA(opA), opB(opB), opC(opC),
  CtaM(CtaM),   CtaN(CtaN),   CtaK(CtaK),
  WarpM(WarpM), WarpN(WarpN), WarpK(WarpK),
  InstM(InstM), InstN(InstN), InstK(InstK),
  NumStages(NumStages), SplitKSlices(SplitKSlices) {}

  virtual void launch(int M, int N, int K,
                      float alpha, float beta,
                      const void* A, int ldA,

                      const void* B, int ldB,
                      void* C, int ldC,
                      void* workspace) = 0;

  virtual size_t workspace_size(int M, int N, int K,
                                float alpha, float beta) = 0;
  
  virtual std::string str() {
    std::stringstream info;
    info << "_" << strOfOptGemmElemType(elemA) 
         << "_" << strOfOptGemmElemType(elemAccum)
         << "_" << CtaM << "x" << CtaN << "x" << CtaK 
         << "_" << WarpM << "x" << WarpN << "x" << WarpK
         << "_" << InstM << "x" << InstN << "x" << InstK 
         << "_" << NumStages << "_" << SplitKSlices << "_"
         << strOfOptGemmOp(opA) << strOfOptGemmOp(opB) << strOfOptGemmOp(opC);
    return info.str();;
  } 
};