"""Print the multi-GPU table (markdown) from results/dist/<gpu>-x<N>/*.json.

    python scripts/dist_table.py [results/dist]

Speedup and scaling efficiency are against the 1-GPU row (<gpu>-x1/single.json) at the same
global batch: efficiency = tokens/s / (N x single tokens/s).
"""

import json
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/dist")
rows = [(p, json.loads(p.read_text())) for p in sorted(root.glob("*/*.json")) if not p.name.endswith(".bench.json")]
single = {p.parent.name.rsplit("-x", 1)[0]: r["tokens_per_s"]["median"] for p, r in rows if p.parent.name.endswith("-x1")}
print("| GPU | Config | GPUs | Tokens/s (min-max) | Speedup | Scaling eff. | MFU/GPU | Peak mem/GPU | Flags |")
print("|---|---|---|---|---|---|---|---|---|")
for p, r in rows:
    gpu, n = p.parent.name.rsplit("-x", 1)
    n = int(n)
    t = r["tokens_per_s"]
    base = single.get(gpu)
    speed = f"{t['median'] / base:.2f}x" if base else "-"
    eff = f"{t['median'] / (n * base):.0%}" if base else "-"
    mfu = r["mfu"]["fp16_tc"]["median"] if r.get("mfu") else None
    mem = r["peak_mem_gb"]["allocated"] if r.get("peak_mem_gb") else None
    print(f"| {gpu} | {p.stem} | {n} | {t['median']:,.0f} ({t['min']:,.0f}-{t['max']:,.0f}) | {speed} | {eff} | "
          f"{f'{mfu:.1%}' if mfu else '-'} | {f'{mem:.2f} GB' if mem else '-'} | {len(r.get('throttle_flags', []))} |")
