#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#include "src/opt_gemm.h"

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

template<typename ElementA, typename ElementB, typename ElementC, typename ElementAccumulator,
         OptGemmOp OpA, OptGemmOp OpB, OptGemmOp OpC,
         int CtaM, int CtaN, int CtaK,
         int WarpM, int WarpN, int WarpK, 
         int InstM, int InstN, int InstK,
         int NumStages, int SplitKSlices = 1>
struct AmpereGemmKernel {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;

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
                                          cutlass::gemm::GemmShape<CtaM, CtaN, CtaK>,
                                          cutlass::gemm::GemmShape<WarpM, WarpN, WarpK>,
                                          cutlass::gemm::GemmShape<InstM, InstN, InstK>,
                                          EpilogueOp,
                                          cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
                                          NumStages, 8, 8, true>;

  using TensorRefA = cutlass::TensorRef<ElementA, LayoutA>;
  using TensorRefB = cutlass::TensorRef<ElementB, LayoutB>;
  using TensorRefC = cutlass::TensorRef<ElementC, LayoutC>;
  
  typename Gemm::Arguments arguments(uint M, uint N, uint K,
                                     ElementAccumulator alpha,
                                     ElementAccumulator beta,
                                     const void* A, uint ldA,
                                     const void* B, uint ldB,
                                     void* C, uint ldC) {
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

  cutlass::Status launch(uint M, uint N, uint K,
                         ElementAccumulator alpha,
                         ElementAccumulator beta,
                         const void* A, uint ldA,
                         const void* B, uint ldB,
                         void* C, uint ldC, void* workspace) {
    Gemm gemm_op;
    
    auto args = arguments(M, N, K, alpha, beta, A, ldA, B, ldB, C, ldC);
    
    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) return status;

    status = gemm_op.initialize(args, workspace);
    if (status != cutlass::Status::kSuccess) return status;

    // Launch initialized CUTLASS kernel
    return gemm_op();
  }

  size_t get_workspace_size(uint M, uint N, uint K,
                            ElementAccumulator alpha,
                            ElementAccumulator beta) {
    return Gemm::get_workspace_size(arguments(M, N, K, alpha, beta, 
                                              nullptr, K, nullptr, N,
                                              nullptr, N));
  }
};