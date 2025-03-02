import pyoptgemm
import torch

a = torch.randn((2048, 512), dtype=torch.float16).cuda()
b = torch.randn((512, 1024), dtype=torch.float16).cuda()
c1 = a@b

c = pyoptgemm.mm(a, b)

print(torch.allclose(c, c1))