"""Regenerate every results table and figure from results/ (nothing is typed in by hand).

    scripts/run_container.sh python scripts/make_figures.py

Writes figures/*.png and docs/results.md, and replaces the block between
<!-- results:start --> and <!-- results:end --> in README.md with the headline tables.
A result that doesn't exist yet renders as TBD.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS, FIGS = Path("results"), Path("figures")
GPUS = {"rtx-2070-super": "RTX 2070 Super (laptop)", "t4": "Tesla T4"}
STAGES = {
    "s0_fp32": "S0 eager FP32", "s1_fp16": "S1 FP16 AMP", "s2_mem_efficient": "S2 + mem-efficient SDPA",
    "s3_compile": "S3 + torch.compile", "s4_cudagraphs": "S4 + CUDA graphs", "s5_triton": "S5 + Triton kernels",
    "s6_cuda_norm": "S6 + CUDA C++ RMSNorm", "s7_cuda_attn": "S7 + CUDA C++ FlashAttention",
    "s8_fused_ce": "S8 + fused linear-CE",
}
DIST = {"single": "1 GPU", "ddp": "DDP (default buckets)", "ddp_tuned": "DDP (100 MB buckets, grad views)",
        "fsdp": "FSDP2", "fsdp_prefetch": "FSDP2 + prefetch", "fsdp_compile_blocks": "FSDP2 + prefetch + per-block compile",
        "tp": "TP=2 + SP"}
EVAL_TASKS = [("hellaswag", "HellaSwag (acc_norm)"), ("arc_easy", "ARC-Easy (acc_norm)"),
              ("piqa", "PIQA (acc_norm)"), ("lambada_openai", "LAMBADA (acc)")]
# Reference categorical palette (dataviz skill), fixed order.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TBD = "TBD"


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def ladder(gpu: str) -> tuple[str, list[tuple[str, float | None]]]:
    rows, bars, prev = [], [], None
    for stem, name in STAGES.items():
        r = load(RESULTS / "ladder" / gpu / f"{stem}.json")
        if r is None:
            rows.append([name] + [TBD] * 5)
            bars.append((name, None))
            continue
        tps = r["tokens_per_s"]["median"]
        flags = " ⚠ clock-limited" if r["throttle_flags"] else ""
        mem = r["peak_mem_gb"]
        rows.append([
            name, f"{tps:,.0f}{flags}", f"{r['mfu']['fp16_tc']['median']:.1%}" if r["mfu"] else "-",
            f"{mem['allocated']:.2f} / {mem['reserved']:.2f} GB" if mem else "-",
            f"{r['trials'][-1]['loss_last']:.3f}", f"{tps / prev:.2f}×" if prev else "—",
        ])
        bars.append((name, tps))
        prev = tps
    md = table(["Stage", "Tokens/s", "MFU", "Peak mem alloc / reserved", "Loss (last step)", "Δ vs previous"], rows)
    return md, bars


def ladder_figure(all_bars: dict[str, list]) -> None:
    names = list(STAGES.values())
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=150)
    h = 0.38
    for i, (gpu, color) in enumerate(zip(all_bars, (BLUE, ORANGE))):
        vals = dict(all_bars[gpu])
        ys = [k + (i - 0.5) * h for k in range(len(names))]
        xs = [vals.get(n) or 0 for n in names]
        ax.barh(ys, xs, height=h - 0.04, color=color, label=GPUS[gpu])
        for y, n in zip(ys, names):
            v = vals.get(n)
            ax.text((v or 0) + 200, y, f"{v:,.0f}" if v else TBD, va="center", fontsize=7, color=MUTED)
    ax.set_yticks(range(len(names)), names, fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("training tokens/s (higher is better)", fontsize=8, color=MUTED)
    ax.tick_params(axis="x", labelsize=7, colors=MUTED)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Optimization ladder: same model, seq 1024, fixed global batch", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(FIGS / "ladder_tokens_per_s.png")
    plt.close(fig)


def kernels(gpu: str) -> str:
    rows = []
    for path in sorted((RESULTS / "kernels" / gpu).glob("*.json")):
        r = load(path)
        for row in r["rows"]:
            rows.append([r["kernel"], row["case"], row["provider"], f"{row['fwd_ms']['median'] * 1e3:,.1f}",
                         f"{row['fwd_bwd_ms']['median'] * 1e3:,.1f}",
                         f"{row['pct_peak_fwd']:.0%}", f"{row['pct_peak_fwd_bwd']:.0%}" + (" ⚠" if row["flag"] else "")])
    if not rows:
        return TBD
    return table(["Kernel", "Case", "Provider", "fwd µs", "fwd+bwd µs", "fwd % of DRAM peak", "fwd+bwd % of peak"], rows)


def dist() -> str:
    rows = []
    for gpu_dir in sorted((RESULTS / "dist").glob("*-x*")) if (RESULTS / "dist").exists() else []:
        gpu, n = gpu_dir.name.rsplit("-x", 1)
        base = load(RESULTS / "dist" / f"{gpu}-x1" / "single.json")
        for stem, name in DIST.items():
            r = load(gpu_dir / f"{stem}.json")
            if r is None:
                continue
            tps = r["tokens_per_s"]["median"]
            comm = _nccl_share(RESULTS / "nsys" / gpu_dir.name / f"{stem}_cuda_gpu_kern_sum.csv")
            rows.append([gpu, name, n, f"{tps:,.0f}",
                         f"{tps / (int(n) * base['tokens_per_s']['median']):.0%}" if base else TBD,
                         f"{r['peak_mem_gb']['allocated']:.2f} GB", comm])
    return table(["GPU", "Strategy", "GPUs", "Tokens/s", "Scaling eff. vs 1 GPU", "Peak mem/GPU", "NCCL % of kernel time"],
                 rows) if rows else TBD


def _nccl_share(path: Path) -> str:
    if not path.exists():
        return TBD
    import csv

    rows = list(csv.DictReader(path.open()))
    total = sum(float(r["Total Time (ns)"]) for r in rows)
    comm = sum(float(r["Total Time (ns)"]) for r in rows if "nccl" in r["Name"].lower())
    return f"{comm / total:.1%}"


def quality() -> str:
    main = load(RESULTS / "main" / "summary.json")
    rows = []
    for name, label, tokens in (("gpt2-124m", "GPT-2 124M (HF `gpt2`, evaluated by us)", "~10B+ (published)"),
                                ("nanonemo", "NanoNemo-114M (ours)", f"{main['tokens']:,}" if main else "~2.4B")):
        r = load(RESULTS / "eval" / f"{name}.json")
        scores = [f"{r['scores'][t]['value']:.1%} ± {r['scores'][t]['stderr']:.1%}" if r else TBD for t, _ in EVAL_TASKS]
        val = "n/a" if name.startswith("gpt2") else (f"{main['val_loss']:.3f}" if main else TBD)
        rows.append([label, tokens, *scores, val])
    return table(["Model", "Tokens trained", *(l for _, l in EVAL_TASKS), "Val loss"], rows)


def loss_figure() -> str:
    path = RESULTS / "main" / "metrics.jsonl"
    if not path.exists():
        return TBD
    recs = [json.loads(line) for line in path.read_text().splitlines()]
    train = [(r["tokens"] / 1e9, r["loss"]) for r in recs if "loss" in r]
    val, resumes = [], []
    tok_per_step = train[1][0] / max(1, [r for r in recs if "loss" in r][1]["step"] + 1) if len(train) > 1 else None
    for r in recs:
        if "val_loss" in r and tok_per_step:
            val.append(((r["step"] + 1) * tok_per_step, r["val_loss"]))
        if r.get("event") == "start" and r.get("resumed_from") and tok_per_step:
            resumes.append(r["start_step"] * tok_per_step)
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    ax.plot(*zip(*train), color=BLUE, linewidth=1.2, label="train loss")
    if val:
        ax.plot(*zip(*val), color=ORANGE, linewidth=2, marker="o", markersize=3, label="val loss")
    for x in resumes:
        ax.axvline(x, color=MUTED, linewidth=0.8, linestyle="--")
        ax.text(x, ax.get_ylim()[1], " resume", fontsize=7, color=MUTED, va="top")
    ax.set_xlabel("tokens (billions)", fontsize=8, color=MUTED)
    ax.set_ylabel("cross-entropy (nats)", fontsize=8, color=MUTED)
    ax.tick_params(labelsize=7, colors=MUTED)
    ax.grid(color=GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Main run on Kaggle 2x T4 (dashed: session resume points)", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(FIGS / "main_loss.png")
    plt.close(fig)
    return "![Main-run loss](../figures/main_loss.png)"


def main() -> None:
    FIGS.mkdir(exist_ok=True)
    sections, bars = [], {}
    for gpu, name in GPUS.items():
        md, bars[gpu] = ladder(gpu)
        sections.append(f"### Optimization ladder: {name}\n\n{md}")
    ladder_figure(bars)
    sections.append("![Ladder tokens/s](../figures/ladder_tokens_per_s.png)")
    sections.append(f"### Multi-GPU (Kaggle 2x T4)\n\n{dist()}")
    sections.append("### Precision (AWS L4)\n\n" + table(["Precision", "Tokens/s", "MFU", "Peak mem", "Loss after short run"],
                                                           [[p, TBD, TBD, TBD, TBD] for p in
                                                            ("FP16 AMP", "BF16 AMP", "BF16 + Transformer Engine", "FP8 + Transformer Engine")]))
    sections.append(f"### Model quality (zero-shot, lm-evaluation-harness 0.4.9, identical settings)\n\n{quality()}")
    sections.append(f"### Main run\n\n{loss_figure()}")
    for gpu, name in GPUS.items():
        k = kernels(gpu)
        if k != TBD:
            sections.append(f"### Kernel microbenchmarks: {name}\n\n{k}")
    body = "\n\n".join(sections) + "\n"
    Path("docs/results.md").write_text("# Results\n\nGenerated by `scripts/make_figures.py` from `results/`.\n\n" + body)

    readme = Path("README.md")
    if readme.exists():
        text = readme.read_text()
        start, end = "<!-- results:start -->", "<!-- results:end -->"
        if start in text and end in text:
            head = "\n\n".join(s.replace("../figures/", "figures/") for s in sections[:7]) + "\n"
            text = text[: text.index(start) + len(start)] + "\n" + head + text[text.index(end):]
            readme.write_text(text)
    print("wrote docs/results.md, figures/, README.md results block")


if __name__ == "__main__":
    main()
