#include <type_traits>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#include "cuda/cuda_kernel.h"

#pragma once

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = float;                   // <- data type of accumulator
using ElementComputeEpilogue = float;  // <- data type of epilogue operations
using ElementInputA = cutlass::half_t;                        // <- data type of elements in input matrix A
using ElementInputB = cutlass::half_t;                        // <- data type of elements in input matrix B
using ElementOutput = cutlass::half_t;                        // <- data type of elements in output matrix D

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock =
//128x256x32 provides a little better perf
    cutlass::gemm::GemmShape<256, 128, 32>;  // <- threadblock tile M = 128, N = 128, K = 16 
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;  // <- warp tile M = 64, N = 64, K = 16
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;  // <- MMA Op tile M = 16, N = 8, K = 8

// This code section describes how threadblocks are scheduled on GPU

// This code section describes the epilogue part of the kernel

template<OptGemmElemType ElementType>
using conditional_type = typename std::conditional<ElementType == OptGemmHalf, cutlass::half_t, float>::type;

template<OptGemmElemType OptGemmElemA, OptGemmElemType OptGemmElemB,
         OptGemmElemType OptGemmElemC, OptGemmElemType OptGemmElemAccum,
         OptGemmOp OpA, OptGemmOp OpB, OptGemmOp OpC,
         int kCtaM, int kCtaN, int kCtaK,
         int kWarpM, int kWarpN, int kWarpK, 
         int kInstM, int kInstN, int kInstK,
         int kNumStages, int kSplitKSlices = 1>
class AmpereGemmKernel : public CudaGemmKernel {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using ElementA = conditional_type<OptGemmElemA>;
  using ElementB = conditional_type<OptGemmElemB>;
  using ElementC = conditional_type<OptGemmElemC>;
  using ElementAccumulator = conditional_type<OptGemmElemAccum>;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementC,                                     // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- the number of elements per vectorized
                                                       // memory access. For a byte, it's 16
                                                       // elements. This becomes the vector width of
                                                       // math instructions in the epilogue too
    ElementAccumulator,                                // <- data type of accumulator
    ElementAccumulator>;  // <- data type for alpha/beta in linear combination function

  using Gemm = cutlass::gemm::device::Gemm<ElementA,
                                          LayoutA,
                                          ElementB,
                                          LayoutB,
                                          ElementC,
                                          LayoutC,
                                          ElementAccumulator,
                                          cutlass::arch::OpClassTensorOp,
                                          cutlass::arch::Sm80,
                                          cutlass::gemm::GemmShape<kCtaM, kCtaN, kCtaK>,
                                          cutlass::gemm::GemmShape<kWarpM, kWarpN, kWarpK>,
                                          cutlass::gemm::GemmShape<kInstM, kInstN, kInstK>,
                                          EpilogueOp,
                                          cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
                                          kNumStages, 8, 8, (kSplitKSlices > 1)>;

  using TensorRefA = cutlass::TensorRef<ElementA, LayoutA>;
  using TensorRefB = cutlass::TensorRef<ElementB, LayoutB>;
  using TensorRefC = cutlass::TensorRef<ElementC, LayoutC>;
  
  Gemm gemm_op;

public:
  AmpereGemmKernel() :
    CudaGemmKernel(CudaArchAmpere,
                   OptGemmElemA, OptGemmElemB, OptGemmElemC, OptGemmElemAccum,
                   OpA, OpB, OpC,
                   kCtaM, kCtaN, kCtaK, kWarpM, kWarpN, kWarpK,
                   kInstM, kInstN, kInstK, kNumStages, kSplitKSlices) {}

  typename Gemm::Arguments arguments(uint M, uint N, uint K,
                                     ElementAccumulator alpha,
                                     ElementAccumulator beta,
                                     ElementA* A, uint ldA,
                                     ElementB* B, uint ldB,
                                     ElementC* C, uint ldC) {
    cutlass::gemm::GemmCoord problem_size = cutlass::gemm::GemmCoord(M, N, K);
    typename Gemm::Arguments args{problem_size,
                                  TensorRefA(A, LayoutA(ldA)),
                                  TensorRefA(B, LayoutA(ldB)),
                                  TensorRefA(C, LayoutA(ldC)),
                                  TensorRefA(C, LayoutA(ldC)),
                                  {alpha, beta},
                                  SplitKSlices};
    return args;
  }

  virtual void launch(int M, int N, int K,
                         float alpha,
                         float beta,
                         const void* A, int ldA,
                         const void* B, int ldB,
                         void* C, int ldC, void* workspace) {
    auto args = arguments(M, N, K, alpha, beta,
                          (ElementA*)A, ldA,
                          (ElementB*)B, ldB,
                          (ElementC*)C, ldC);
    
    // cutlass::Status status = gemm_op.can_implement(args);
    // if (status != cutlass::Status::kSuccess) return status;

    gemm_op.initialize(args, workspace);
    // if (status != cutlass::Status::kSuccess) return status;

    // Launch initialized CUTLASS kernel
    gemm_op();
  }

  virtual size_t workspace_size(int M, int N, int K,
                            float alpha,
                            float beta) {
    return Gemm::get_workspace_size(arguments(M, N, K, alpha, beta, 
                                              nullptr, K, nullptr, N,
                                              nullptr, N));
  }
};