"""Print the optimization-ladder table (markdown) from results/ladder/<gpu>/*.json.

    python scripts/ladder_table.py rtx-2070-super
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gpu", help="results/ladder subfolder, e.g. rtx-2070-super or t4")
    args = parser.parse_args()

    rows = []
    for path in sorted(Path("results/ladder", args.gpu).glob("s*.json")):
        r = json.loads(path.read_text())
        tps = r["tokens_per_s"]
        clocks = [t["gpu"]["sm_mhz"]["median"] for t in r["trials"] if t["gpu"]["n_samples"]]
        rows.append({
            "stage": path.stem,
            "tps": tps,
            "mfu": r["mfu"]["fp16_tc"]["median"] if r["mfu"] else None,
            # CUDA graphs keep their memory in a pool reserved during warmup, which the
            # post-warmup "allocated" peak doesn't see; reserved is comparable across stages.
            "mem": r["peak_mem_gb"],
            "loss": r["trials"][-1]["loss_last"],
            "steps": r["bench"]["warmup_steps"] + r["bench"]["steps"] * r["bench"]["trials"],
            "warmup_s": r["warmup_s"],
            "clock": min(clocks) if clocks else None,
            "flags": r["throttle_flags"],
        })

    print("| Stage | Tokens/s (min–max) | MFU | Peak mem alloc / reserved | Loss @ step | Δ vs previous | Warmup | SM clock (lowest trial median) |")
    print("|---|---|---|---|---|---|---|---|")
    prev = None
    for row in rows:
        tps = row["tps"]
        delta = f"{tps['median'] / prev:.2f}×" if prev else "—"
        mfu = f"{row['mfu']:.1%}" if row["mfu"] is not None else "n/a"
        mem = f"{row['mem']['allocated']:.2f} / {row['mem']['reserved']:.2f} GB" if row["mem"] else "n/a"
        clock = f"{row['clock']:.0f} MHz" if row["clock"] else "n/a"
        flag = " ⚠" if row["flags"] else ""
        print(
            f"| {row['stage']}{flag} | {tps['median']:,.0f} ({tps['min']:,.0f}–{tps['max']:,.0f}) | {mfu} | {mem} "
            f"| {row['loss']:.3f} @ {row['steps']} | {delta} | {row['warmup_s']:.0f} s | {clock} |"
        )
        prev = tps["median"]
    for row in rows:
        for flag in row["flags"]:
            print(f"\n⚠ {row['stage']}: {flag}")


if __name__ == "__main__":
    main()
