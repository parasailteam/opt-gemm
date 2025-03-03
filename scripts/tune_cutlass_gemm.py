import subprocess
import os
import csv

def run_command(c):
  (s,o) = subprocess.getstatusoutput(c)
  if s != 0:
    print(f"Error running {c}. Output: {o}")
    assert False
  return o

def find_best_config(csvfile):
  sortedfile = sorted(csvfile, key = lambda row1: float(row1[-1]))
  print(", ".join(sortedfile[-1]))

def run(m, n, k):
  print(f"---- Running for {m}x{n}x{k} ----")
  cutlass_profiler = os.path.join(os.getcwd(), "../../cutlass/build", "tools/profiler/cutlass_profiler")
  cutlass_command = cutlass_profiler + f" --operation=Gemm  --op_class=tensorop --m={m} --n={n} --k={k} --A=f16:* --B=f16:* --C=f16:* --accum=f32 --min_cc=80 --cta_k=32"
  cutlass_command = cutlass_command + f" --output=output.csv"
  cutlass_output = run_command(cutlass_command)
  with open('output.gemm.csv', newline='') as csvfile:
    outputcsv = csv.reader(csvfile, delimiter=',')
    rows = []
    for row in outputcsv:
      rows += [row]

    find_best_config(rows[1:])

for b in range(10, 12):
  run(2**b, 16384, 3072)
  run(2**b, 1792, 3072)
  run(2**b, 16384, 1792)

  
