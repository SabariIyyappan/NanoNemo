"""Train NanoNemo from a YAML run config.

    scripts/run_container.sh python scripts/train.py --config configs/smoke.yaml
    torchrun --nproc_per_node=2 scripts/train.py --config configs/dist/ddp.yaml     # multi-GPU
    ... --set train.resume_from=runs/x/ckpt train.stop_after_min=690                # overrides
"""

import argparse
import dataclasses

import yaml

from nanonemo import distributed
from nanonemo.train.config import load_config
from nanonemo.train.trainer import train


def print_record(r: dict) -> None:
    if "loss" in r:
        line = (
            f"step {r['step']:5d} | loss {r['loss']:.4f} | lr {r['lr']:.2e} | "
            f"gnorm {r['grad_norm']:.3f} | {r['tokens_per_s']:,.0f} tok/s"
        )
        if "peak_mem_gb" in r:
            line += f" | {r['peak_mem_gb']:.2f} GB"
        if "mfu" in r:
            line += f" | MFU {r['mfu']:.1%}"
        print(line, flush=True)
    elif "val_loss" in r:
        print(f"step {r['step']:5d} | val_loss {r['val_loss']:.4f}", flush=True)
    else:
        print(r, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="SECTION.KEY=VALUE",
                        help="override config values, e.g. train.data_dir=/kaggle/input/... (values parsed as YAML)")
    args = parser.parse_args()
    model_cfg, train_cfg = load_config(args.config)
    for item in args.set:
        key, value = item.split("=", 1)
        section, name = key.split(".", 1)
        target = {"model": model_cfg, "train": train_cfg}[section]
        if name not in {f.name for f in dataclasses.fields(target)}:
            raise SystemExit(f"unknown {section} key {name!r}")
        target = dataclasses.replace(target, **{name: yaml.safe_load(value)})
        if section == "model":
            model_cfg = target
        else:
            train_cfg = target

    sink = None
    if train_cfg.wandb_project and distributed.init().main:
        from nanonemo.train.wandb_sink import WandbSink

        sink = WandbSink(model_cfg, train_cfg)

    def log_fn(record: dict) -> None:
        print_record(record)
        if sink:
            sink(record)

    try:
        train(model_cfg, train_cfg, log_fn=log_fn)
    finally:
        if sink:
            sink.finish()
        distributed.destroy()


if __name__ == "__main__":
    main()
