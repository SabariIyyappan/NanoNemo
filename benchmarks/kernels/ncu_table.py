"""Print the metrics CSV that ncu.sh exports as a table: one row per kernel launch."""

import csv
import sys

with open(sys.argv[1]) as f:
    rows = [r for r in csv.reader(f) if r]
header = next(i for i, r in enumerate(rows) if "Kernel Name" in r)
cols, data = rows[header], rows[header + 2:]  # header + 1 units row
keep = ["Kernel Name", "gpu__time_duration.sum", "dram__bytes_read.sum", "dram__bytes_write.sum",
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active", "launch__registers_per_thread", "launch__grid_size"]
short = {k: k.replace("__", ".").split(".avg")[0].split(".sum")[0] for k in keep}
keep = [k for k in keep if k in cols]  # older CSVs lack the tensor-pipe column
idx = [cols.index(k) for k in keep]
units = rows[header + 1]
print("  ".join(f"{short[k]}" for k in keep))
for r in data:
    print("  ".join(f"{r[i][:40]}" for i in idx))
print("units:", ", ".join(f"{short[k]}={units[i]}" for k, i in zip(keep[1:], idx[1:])))
