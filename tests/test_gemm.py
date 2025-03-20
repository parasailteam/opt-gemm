import pyoptgemm
import torch

def run_gemm(m, n, k, trb):
  a = torch.ones((m, k), dtype=torch.float16).cuda()
  if trb:
    b = torch.randn((n, k), dtype=torch.float16)
  else:
    b = torch.randn((k, n), dtype=torch.float16)
  b = b.cuda()
  if trb:
    ref = a@b.mT
  else:
    ref = a@b

  c = pyoptgemm.mm(a, b, trb=trb)
  return torch.allclose(c, ref)

  # print([(i, torch.sum(b[i,:]).item()) for i in range(10)])
  # print([(i, torch.sum(b[i,:]).item()) for i in range(b.shape[0]-10,b.shape[0])])

  # print(ref)

  # print(c)

  # for i in range(ref.shape[0]):
  #   for j in range(ref.shape[1]):
  #     if not torch.allclose(c[i][j], ref[i][j]):
  #       print(i,j, c[i][j], ref[i][j])
  #       assert False

def test_phi3_mini():
  assert run_gemm(2048, 16384, 3072, False)

def test_phi3_mini_c23():
  assert run_gemm(2048, 3072, 1792, True)
  assert run_gemm(2048, 16384, 1792, True)

  assert run_gemm(2048, 3072, 1792, False)
  assert run_gemm(2048, 16384, 1792, False)
