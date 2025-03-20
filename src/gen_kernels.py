import argparse
import os

def cutlass_type(ty):
  if ty == "float": return "OptGemmFloat"
  if ty == "half": return "OptGemmHalf"
  assert False, f"Unknown type {ty}"

def cutlass_shape(shape):
  return f"{shape[0]}, {shape[1]}, {shape[2]}"

class GemmShape:
  def __init__(self, m=0, n=0, k=0, opA="N", opB="N", opC="N"):
    self.m = m
    self.k = k
    self.n = n
    self.opA = opA
    self.opB = opB
    self.opC = opC

  def __str__(self):
    return f"{m}x{n}x{k}_{self.opA}{self.opB}{self.opC}"

  def cpp_object(self):
    if self.m == 0 and self.n == 0 and self.k == 0:
      return f"GemmShape(OptGemmOp::OptGemmOp_{self.opA}, OptGemmOp::OptGemmOp_{self.opB}, OptGemmOp::OptGemmOp_{self.opC})"
    return f"GemmShape({self.m}, {self.n}, {self.k}, OptGemmOp::OptGemmOp_{self.opA}, OptGemmOp::OptGemmOp_{self.opB}, OptGemmOp::OptGemmOp_{self.opC})"

class KernelConfig:
  def __init__(self, elemA, elemB, elemC, elemAccum, opA, opB, opC, arch, cta, warp, mma, stages, split_k_slices):
    self.elemA = elemA
    self.elemB = elemB
    self.elemC = elemC
    self.elemAccum = elemAccum
    self.opA = opA
    self.opB = opB
    self.opC = opC
    self.arch = arch
    self.cta = cta
    self.warp = warp
    self.mma = mma
    self.stages = stages
    self.split_k_slices = split_k_slices
  
  def template_decl(self):
    return f"AmpereGemmKernel<{cutlass_type(self.elemA)}, {cutlass_type(self.elemB)}, {cutlass_type(self.elemC)}, {cutlass_type(self.elemAccum)}, OptGemmOp_{self.opA}, OptGemmOp_{self.opB}, OptGemmOp_{self.opC}, {cutlass_shape(self.cta)}, {cutlass_shape(self.warp)}, {cutlass_shape(self.mma)}, {self.stages}, {self.split_k_slices}>"

  def object_name(self):
    return f"ampere_{self.elemA}_{self.elemB}_{self.elemC}_{self.cta[0]}x{self.cta[1]}x{self.cta[2]}_{self.warp[0]}x{self.warp[1]}x{self.warp[2]}_{self.stages}_{self.split_k_slices}_{self.opA}{self.opB}{self.opC}"

AmpereKernels = {
  #Default Kernel
  GemmShape(opA="N", opB="T", opC="N")  : KernelConfig("half", "half", "half", "float", "N", "T", "N", 80, [256,128,32], [64,64,32], [16,8,16], 4, 1),
  GemmShape(opA="N", opB="N", opC="N")  : KernelConfig("half", "half", "half", "float", "N", "N", "N", 80, [256,128,32], [64,64,32], [16,8,16], 4, 1),

  #Phi-3-mini
  GemmShape(2048, 16384, 3072, opB="T") : KernelConfig("half", "half", "half", "float", "N", "T", "N", 80, [128,128,32], [64,64,32], [16,8,16], 4, 1),
  GemmShape(2048, 16384, 3072, opB="N") : KernelConfig("half", "half", "half", "float", "N", "N", "N", 80, [128,128,32], [64,64,32], [16,8,16], 4, 1),

  #Phi-3-mini-C-23%
  GemmShape(2048, 16384, 1792, opB="T") : KernelConfig("half", "half", "half", "float", "N", "T", "N", 80, [128,128,32], [64,64,32], [16,8,16], 4, 1),
  GemmShape(2048, 1792,  3072, opB="T") : KernelConfig("half", "half", "half", "float", "N", "T", "N", 80, [128,128,32], [64,64,32], [16,8,16], 4, 1),

  GemmShape(2048, 16384, 1792, opB="N") : KernelConfig("half", "half", "half", "float", "N", "N", "N", 80, [128,128,32], [64,64,32], [16,8,16], 4, 1),
  GemmShape(2048, 1792,  3072, opB="N") : KernelConfig("half", "half", "half", "float", "N", "N", "N", 80, [128,128,32], [64,64,32], [16,8,16], 4, 1),
}

def generate_kernels():
  kernel_out_dir = os.path.join(os.path.dirname(__file__), "cuda/ampere/kernels/")
  kernel_decl_file = os.path.join(kernel_out_dir, "kernel_decl.h")

  if not os.path.exists(kernel_out_dir):
    os.makedirs(kernel_out_dir, exist_ok=True)

  kernel_includes = ['#include "cuda/ampere/ampere_gemm_kernel.h"', '#include <unordered_map>']
  kernel_decls = list(kernel_includes)
  kernel_array = ["CudaGemmKernel* AllAmpereKernels[] = {"]
  kernel_cmake = []
  shape_to_kernel_map = ["std::unordered_map<GemmShape, CudaGemmKernel*> GemmShapeToAmpereKernel = {"]
  kernel_generated = set()

  for shape,kernel in AmpereKernels.items():
    shape_to_kernel_map += [f"{{{shape.cpp_object()}, &{kernel.object_name()}}}"]
    if kernel.object_name() not in kernel_generated:
      kernel_generated.add(kernel.object_name())
    else:
      continue
    kernel_decls += ["extern " + kernel.template_decl() + " " + kernel.object_name() + ";"]
    kernel_array += ["&"+kernel.object_name()]

    with open(os.path.join(kernel_out_dir, kernel.object_name()+".cu"), "w") as f:
      kernel_object_init = kernel_includes + [kernel.template_decl() + " " + kernel.object_name() + ";"]
      f.write("\n".join(kernel_object_init))

    kernel_cmake += [f"${{CUDA}}/ampere/kernels/{kernel.object_name()+'.cu'}"]

  kernel_array += ["};"]
  shape_to_kernel_map += ["};"]

  with open(kernel_decl_file, "w") as decl_file:
    decl_file.write("\n".join(kernel_decls) + "\n\n" + kernel_array[0] + ",\n".join(kernel_array[1:]) + "\n")
    decl_file.write(shape_to_kernel_map[0] + ",\n".join(shape_to_kernel_map[1:]))

  with open(os.path.join(kernel_out_dir, "kernels.cmake"), "w") as f:
    f.write(f'set(CUDA_KERNELS {" ".join(kernel_cmake)})')

if __name__ == "__main__":
  from argparse import RawTextHelpFormatter

  parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter,
                                   description="Generate kernels for a GeKMM problem.")
  # parser.add_argument('archs', required=False, type=str, nargs="+", action='append')
  # parser.add_argument('cta', required=False, type=str, nargs="+", action='append')

  args = parser.parse_args()

  generate_kernels()
