import OptGemm
import torch

def mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  c = torch.empty(size=(a.shape[0],b.shape[1]),
                  dtype=a.dtype).to(a.device)
  
  OptGemm.hgemm(c.shape[0], c.shape[1], a.shape[1], 1.0, 0.0,
                a.data_ptr(), a.shape[1], 
                b.data_ptr(), b.shape[1],
                c.data_ptr(), c.shape[1])
  return c