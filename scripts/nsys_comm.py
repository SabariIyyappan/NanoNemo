"""Share of GPU kernel time in NCCL kernels, from an `nsys stats --report cuda_gpu_kern_sum` CSV.

    python scripts/nsys_comm.py results/nsys/t4-x2/ddp_cuda_gpu_kern_sum.csv

This is communication's share of all kernel time, summed over GPUs; it does not say how much
of it overlapped compute. For that, compare step time against the 1-GPU run at the same
per-GPU work, or look at the timeline.
"""

import csv
import sys
from collections import defaultdict

rows = list(csv.DictReader(open(sys.argv[1])))
total = sum(float(r["Total Time (ns)"]) for r in rows)
comm = defaultdict(float)
for r in rows:
    if "nccl" in r["Name"].lower():
        comm[r["Name"].split("(")[0][:60]] += float(r["Total Time (ns)"])
print(f"kernel time {total / 1e6:,.1f} ms, NCCL {sum(comm.values()) / 1e6:,.1f} ms "
      f"= {sum(comm.values()) / total:.1%} of GPU kernel time")
for name, t in sorted(comm.items(), key=lambda kv: -kv[1]):
    print(f"  {t / 1e6:9,.1f} ms  {name}")
