import numpy as np
from typing import List
import random
import math

random.seed(1234)

import pdb


def positional_encoding(lens: List[int], emb_dim: int):
    """Constant input tensors."""
    pos_encoding = []
    for seq_len in lens:
        # Compute the positional encodings once in log space.
        pe = np.zeros((seq_len, emb_dim))
        position = np.expand_dims(np.arange(0, seq_len), axis=1)
        div_term = np.exp(
            np.arange(0, emb_dim, 2) * -(math.log(10000.0) / emb_dim))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pos_encoding.append(pe)
    return pos_encoding


def gen_seq_batch(batch_size, max_seq_len, vocab_size):
    """Generate a random sequence batch."""
    seq = []
    lens = []
    for i in range(batch_size):
        seq_len = random.randint(1, max_seq_len)
        seq.append([random.randint(0, vocab_size - 1) for _ in range(seq_len)])
        lens.append(seq_len)
    return seq, lens


def gen_seq_batch2(batch_size, max_seq_len):
    seq = []
    lens = []
    for i in range(batch_size):
        seq_len = random.randint(1, max_seq_len)
        seq.append([random.randint(0, vocab_size - 1) for _ in range(seq_len)])
        lens.append(seq_len)
    return seq, lens


if __name__ == "__main__":
    gen_seq_batch2(4, 13)
