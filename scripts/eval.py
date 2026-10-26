"""Zero-shot lm-evaluation-harness on one model, with the settings shared by every row of the
quality table (spec: HellaSwag, ARC-Easy, PIQA, LAMBADA).

    scripts/run_container.sh python scripts/eval.py --model gpt2 --name gpt2-124m
    scripts/run_container.sh python scripts/eval.py --model exports/nanonemo --name nanonemo

--model is a Hugging Face id or a local export (scripts/export_hf.py). Both go through
lm-eval's `hf` backend in FP32 with the GPT-2 tokenizer, 0-shot, no BOS token, the same
batch size, so the only difference between rows is the weights.
Writes results/eval/<name>.json.
"""

import argparse
import json
import time
from pathlib import Path

TASKS = ["hellaswag", "arc_easy", "piqa", "lambada_openai"]
METRICS = {"hellaswag": "acc_norm,none", "arc_easy": "acc_norm,none", "piqa": "acc_norm,none",
           "lambada_openai": "acc,none"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, help="examples per task (smoke tests only)")
    args = parser.parse_args()

    import lm_eval

    t0 = time.time()
    out = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={args.model},dtype=float32",
        tasks=args.tasks,
        num_fewshot=0,
        batch_size=args.batch_size,
        device="cuda",
        limit=args.limit,
        random_seed=0, numpy_random_seed=0, torch_random_seed=0, fewshot_random_seed=0,
    )
    scores = {}
    for task in args.tasks:
        r = out["results"][task]
        metric = METRICS.get(task, "acc,none")
        scores[task] = {"metric": metric.split(",")[0], "value": r[metric],
                        "stderr": r.get(metric.replace(",", "_stderr,"))}
        print(f"{task:16s} {metric.split(',')[0]:9s} {r[metric]:.4f}")
    result = {
        "name": args.name, "model": args.model, "scores": scores, "limit": args.limit,
        "settings": {"backend": "hf", "dtype": "float32", "num_fewshot": 0, "batch_size": args.batch_size},
        "versions": out.get("versions"), "n_samples": out.get("n-samples"),
        "lm_eval": lm_eval.__version__, "seconds": time.time() - t0,
        "raw": out["results"],
    }
    path = Path(f"results/eval/{args.name}{'-limit' if args.limit else ''}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
