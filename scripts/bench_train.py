"""Benchmark training throughput for one run config and save the result as JSON.

    scripts/run_container.sh python scripts/bench_train.py --config configs/stages/s0.yaml

Writes results/ladder/<gpu>/<config name>.json unless --out is given.
"""

import argparse
import json
from pathlib import Path

from nanonemo.train.bench import benchmark
from nanonemo.train.config import load_config
from nanonemo.utils.env import gpu_slug


def print_summary(r: dict) -> None:
    tps, ms = r["tokens_per_s"], r["step_ms"]
    print(f"warmup         {r['warmup_s']:.1f} s ({r['bench']['warmup_steps']} steps, excluded)")
    for i, t in enumerate(r["trials"]):
        gpu = t["gpu"]
        clocks = (
            f"SM {gpu['sm_mhz']['median']:.0f} MHz (min {gpu['sm_mhz']['min']}), {gpu['temp_c']['max']} C max"
            if gpu["n_samples"] else "no GPU samples"
        )
        print(f"trial {i}        {t['tokens_per_s']:,.0f} tok/s | {clocks}")
    print(f"tokens/s       {tps['median']:,.0f}  (min {tps['min']:,.0f}, max {tps['max']:,.0f})")
    print(f"step           {ms['median']:,.1f} ms  (min {ms['min']:,.1f}, max {ms['max']:,.1f})")
    print(f"achieved       {r['achieved_tflops']:.2f} TFLOP/s  ({r['flops_per_token'] / 1e6:.1f} MFLOP/token)")
    if r["mfu"]:
        print("MFU            " + ", ".join(f"{m['median']:.1%} of {k} peak" for k, m in r["mfu"].items()))
    if r["peak_mem_gb"]:
        mem = r["peak_mem_gb"]
        print(f"peak memory    {mem['allocated']:.2f} GB allocated, {mem['reserved']:.2f} GB reserved")
    for flag in r["throttle_flags"]:
        print(f"WARNING        {flag}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=50, help="timed optimizer steps per trial")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5, help="untimed steps before the first trial")
    parser.add_argument("--out", help="output JSON path")
    parser.add_argument("--data-dir", help="override train.data_dir (e.g. the Kaggle input path)")
    args = parser.parse_args()

    model_cfg, train_cfg = load_config(args.config)
    if args.data_dir:
        train_cfg.data_dir = args.data_dir
    result = benchmark(model_cfg, train_cfg, steps=args.steps, trials=args.trials, warmup_steps=args.warmup)
    result["config_path"] = args.config

    out = Path(args.out or f"results/ladder/{gpu_slug()}/{Path(args.config).stem}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print_summary(result)
    print(f"saved          {out}")


if __name__ == "__main__":
    main()
