"""Export a NanoNemo training checkpoint to Hugging Face Llama format (for lm-eval).

    scripts/run_container.sh python scripts/export_hf.py --ckpt runs/main/ckpt --out exports/nanonemo

--ckpt is a checkpoint dir or a ckpt root (its `latest` is used); the model config is read
from the run's config.yaml next to the ckpt root unless --config is given.
"""

import argparse
from pathlib import Path

from nanonemo.eval.hf_export import export, load_checkpoint
from nanonemo.train.checkpoint import find_latest
from nanonemo.train.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", help="run config (default: <run dir>/config.yaml)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    path = find_latest(args.ckpt)
    if path is None:
        raise SystemExit(f"no checkpoint under {args.ckpt}")
    config = Path(args.config) if args.config else path.parent.parent / "config.yaml"
    model_cfg, _ = load_config(config)
    out = export(load_checkpoint(path, model_cfg), args.out)
    print(f"exported {path} -> {out}")


if __name__ == "__main__":
    main()
