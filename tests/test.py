import pyoptgemm
import torch

a = torch.randn((2048, 3072), dtype=torch.float16).cuda()
b = torch.randn((3072, 16384), dtype=torch.float16).cuda()
c1 = a@b

c = pyoptgemm.mm(a, b)

print(torch.allclose(c, c1))