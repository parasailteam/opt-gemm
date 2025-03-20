import OptGemm
import torch

def mm(a: torch.Tensor, b: torch.Tensor, tra = False, trb = False) -> torch.Tensor:
  m = a.shape[0]
  k = a.shape[1]
  OpA = OptGemm.OpN
  if tra:
    m = a.shape[1]
    k = a.shape[0]
    OpA = OptGemm.OpT

  n = b.shape[1]
  OpB = OptGemm.OpN
  if trb:
    n = b.shape[0]
    OpB = OptGemm.OpT

  if trb:
    size=(a.shape[0],b.shape[0])
  else:
    size=(a.shape[0],b.shape[1])
  c = torch.empty(size,
                  dtype=a.dtype,
                  device=a.device)

  ldb = b.shape[1]
  
  OptGemm.hgemm(m, n, k, 1.0, 0.0,
                a.data_ptr(), a.shape[1], OpA,
                b.data_ptr(), ldb, OpB,
                c.data_ptr(), c.shape[1])
  return c