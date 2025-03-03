import pyoptgemm
import torch

def run_gemm(m, n, k):
  a = torch.randn((m, k), dtype=torch.float16).cuda()
  b = torch.randn((k, n), dtype=torch.float16).cuda()
  ref = a@b

  c = pyoptgemm.mm(a, b)
  return torch.allclose(c, ref)

def test_phi3_mini():
  assert run_gemm(2048, 16384, 3072)

def test_phi3_mini_c23():
  assert run_gemm(2048, 3072, 1792)
  assert run_gemm(2048, 16384, 1792)

