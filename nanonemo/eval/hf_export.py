"""Export NanoNemo to Hugging Face Llama format, so lm-eval runs it through the same `hf`
backend, tokenizer handling and batching as GPT-2.

The architectures match exactly: pre-norm RMSNorm (eps 1e-5), rotate-half RoPE with base
10000, SwiGLU (gate/up/down), GQA where query head h reads KV head h // group (HF's
repeat_kv), 1/sqrt(head_dim) scaling, no biases, tied input/output embeddings. The weights
are renamed, not changed; tests/eval/test_hf_export.py checks the logits agree.
"""

from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp

from nanonemo.model.config import ModelConfig
from nanonemo.model.model import NanoNemo

EOT = 50256  # GPT-2's <|endoftext|>, the document separator in our data

LAYER_KEYS = {
    "norm_attn.weight": "input_layernorm.weight",
    "attn.wq.weight": "self_attn.q_proj.weight",
    "attn.wk.weight": "self_attn.k_proj.weight",
    "attn.wv.weight": "self_attn.v_proj.weight",
    "attn.wo.weight": "self_attn.o_proj.weight",
    "norm_mlp.weight": "post_attention_layernorm.weight",
    "mlp.w_gate.weight": "mlp.gate_proj.weight",
    "mlp.w_up.weight": "mlp.up_proj.weight",
    "mlp.w_down.weight": "mlp.down_proj.weight",
}


def llama_config(cfg: ModelConfig):
    from transformers import LlamaConfig

    return LlamaConfig(
        vocab_size=cfg.vocab_size, hidden_size=cfg.d_model, intermediate_size=cfg.d_ff,
        num_hidden_layers=cfg.n_layers, num_attention_heads=cfg.n_heads, num_key_value_heads=cfg.n_kv_heads,
        head_dim=cfg.head_dim, max_position_embeddings=cfg.max_seq_len, rms_norm_eps=cfg.norm_eps,
        rope_theta=cfg.rope_base, hidden_act="silu", attention_bias=False, mlp_bias=False,
        tie_word_embeddings=True, bos_token_id=EOT, eos_token_id=EOT, torch_dtype="float32",
    )


def hf_state_dict(model: NanoNemo) -> dict[str, torch.Tensor]:
    sd = {"model.embed_tokens.weight": model.tok_emb.weight, "model.norm.weight": model.norm_out.weight,
          "lm_head.weight": model.lm_head.weight}
    for i, block in enumerate(model.blocks):
        for ours, theirs in LAYER_KEYS.items():
            sd[f"model.layers.{i}.{theirs}"] = block.get_parameter(ours)
    return {k: v.detach().float().cpu().contiguous() for k, v in sd.items()}


def to_hf(model: NanoNemo):
    from transformers import LlamaForCausalLM

    hf = LlamaForCausalLM(llama_config(model.cfg))
    hf.load_state_dict(hf_state_dict(model), strict=True)
    return hf.eval()


def load_checkpoint(path: str | Path, cfg: ModelConfig) -> NanoNemo:
    """Model weights from a DCP training checkpoint (any layout that wrote it), on the CPU."""
    model = NanoNemo(cfg)
    state = {"model": model.state_dict()}
    dcp.load(state, checkpoint_id=str(path))  # reads only the model part; the optimizer state is skipped
    model.load_state_dict(state["model"])
    return model.eval()


def export(model: NanoNemo, out_dir: str | Path, tokenizer: bool = True) -> Path:
    out_dir = Path(out_dir)
    to_hf(model).save_pretrained(out_dir, safe_serialization=True)
    if tokenizer:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained("gpt2").save_pretrained(out_dir)  # the tokenizer we trained with
    return out_dir
