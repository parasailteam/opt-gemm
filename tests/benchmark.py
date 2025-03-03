import pyoptgemm
import torch

def runtime(f, runs, warmup):
  start_event = torch.cuda.Event(enable_timing=True)
  end_event = torch.cuda.Event(enable_timing=True)

  for w in range(warmup):
    c = f()
  torch.cuda.synchronize()
  
  start_event.record()
  for r in range(runs):
    c = f()
  end_event.record()
  torch.cuda.synchronize()

  return start_event.elapsed_time(end_event)/runs

def benchmark(m, n, k):
  a = torch.randn((m, k), dtype=torch.float16).cuda()
  b = torch.randn((k, n), dtype=torch.float16).cuda()

  torch_time = runtime(lambda: a@b, 100, 10)
  torch_gflops = (2*m*n*k)/(torch_time/1e3)/1e9
  pyoptgemm_time = runtime(lambda: pyoptgemm.mm(a,b), 100, 10)
  pyoptgemm_gflops = (2*m*n*k)/(pyoptgemm_time/1e3)/1e9
  print(torch_gflops, pyoptgemm_gflops, torch_time/pyoptgemm_time)

cases = [
  (2048, 16384, 3072),
  # (2048, 1792,  3072),
  # (2048, 16384, 1792)
]

for mnk in cases:
  benchmark(*mnk)