"""GPT-2 BPE tokenizer (via tiktoken) with document separators.

Token IDs are < 65,536, so token streams are stored on disk as uint16.
"""

import numpy as np
import tiktoken

TOKENIZER_NAME = "gpt2"
VOCAB_SIZE = 50257  # real vocabulary; the model pads its embedding to 50304 (a multiple of 64)

_enc = tiktoken.get_encoding(TOKENIZER_NAME)
EOT = _enc.eot_token  # <|endoftext|>, id 50256
assert VOCAB_SIZE == _enc.n_vocab and VOCAB_SIZE <= np.iinfo(np.uint16).max + 1


def encode_doc(text: str) -> np.ndarray:
    """Tokenize one document and append the end-of-text separator."""
    # encode_ordinary ignores special-token strings inside the text, so a web page that
    # literally contains "<|endoftext|>" can't inject a fake document boundary.
    ids = _enc.encode_ordinary(text)
    ids.append(EOT)
    return np.array(ids, dtype=np.uint16)


def encode_batch(texts: list[str], num_threads: int = 8) -> list[np.ndarray]:
    """Tokenize many documents in parallel (tiktoken releases the GIL)."""
    batches = _enc.encode_ordinary_batch(texts, num_threads=num_threads)
    return [np.array(ids + [EOT], dtype=np.uint16) for ids in batches]


def decode(ids) -> str:
    return _enc.decode([int(i) for i in ids])
