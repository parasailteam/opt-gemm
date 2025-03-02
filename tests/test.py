import OptGemm
import torch

a = torch.randn((2048, 512), dtype=torch.float16).cuda()
b = torch.randn((512, 1024), dtype=torch.float16).cuda()
c1 = a@b

c = torch.zeros((a.shape[0], b.shape[1]), dtype=torch.float16).cuda()

OptGemm.hgemm(c.shape[0], c.shape[1], a.shape[1], 1.0, 0.0,
              a.data_ptr(), a.shape[1],
              b.data_ptr(), b.shape[1],
              c.data_ptr(), c.shape[1])

print(torch.allclose(c, c1))