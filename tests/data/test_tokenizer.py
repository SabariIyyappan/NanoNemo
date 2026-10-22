import numpy as np

from nanonemo.data.tokenizer import EOT, VOCAB_SIZE, decode, encode_batch, encode_doc


def test_round_trip_and_separator():
    text = "NanoNemo trains a small language model. Ünïcödé and emoji 🚀 survive too."
    ids = encode_doc(text)
    assert ids.dtype == np.uint16
    assert ids[-1] == EOT
    assert decode(ids[:-1]) == text


def test_special_token_text_is_not_a_boundary():
    ids = encode_doc("before <|endoftext|> after")
    # Only the separator we append is EOT; the literal string is tokenized as plain text.
    assert (ids == EOT).sum() == 1


def test_batch_matches_single():
    texts = ["first document", "second one, a bit longer", ""]
    for batched, text in zip(encode_batch(texts), texts):
        np.testing.assert_array_equal(batched, encode_doc(text))


def test_ids_fit_uint16():
    assert VOCAB_SIZE - 1 <= np.iinfo(np.uint16).max
