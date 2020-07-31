#!/usr/bin/env python3

import numpy as np
import pdb

import sys
from pprint import pprint
from typing import List
from typing import Tuple

import context

from utils import *

vocab_size = 5000
model_dim = 512
ff_inner_dim = 2048
n_heads = 8

drop_rate = 0.5


class Tensor(np.ndarray):
    pass


# ===================  begin learnable paramters =========================

embedding = np.random.rand(vocab_size, model_dim)

encoder_depth = 6

# learnable parameters in encdder
query_proj = []
key_proj = []
value_proj = []

layer_norm_scale = []
layer_norm_bias = []

ff_proj_mat1 = []
ff_proj_bias1 = []
ff_proj_mat2 = []
ff_proj_bias2 = []

# Tensor Array
for i in range(encoder_depth):
    query_proj.append(np.random.rand(model_dim, model_dim))
    key_proj.append(np.random.rand(model_dim, model_dim))
    value_proj.append(np.random.rand(model_dim, model_dim))

    layer_norm_scale.append(np.random.rand(1))
    layer_norm_bias.append(np.random.rand(1))

    ff_proj_mat1.append(np.random.rand(model_dim, ff_inner_dim))
    ff_proj_bias1.append(np.random.rand(1, ff_inner_dim))
    ff_proj_mat2.append(np.random.rand(ff_inner_dim, model_dim))
    ff_proj_bias2.append(np.random.rand(1, model_dim))

# ===================  end learnable paramters =========================


def softmax_func(x: Tensor):
    """primary softmax function defined for a vector."""
    assert x.ndim == 1

    reduced_max = np.max(x)
    shiftx = x - reduced_max
    exps = np.exp(shiftx)
    reduced_sum = np.sum(exps)
    res = exps / reduced_sum
    return res


def norm_func(x: Tensor):
    """primary normalization function defined for a vector."""
    assert x.ndim == 1
    epsilon = 1e-6
    reduced_mean = np.mean(x)
    std = np.sqrt(np.mean(np.abs(x - np.mean(x))**2))
    res = (x - reduced_mean) / std
    return res


def relu(x: Tensor):
    """primary activation function defined for a scalar."""
    return max(0, x)


def dropout(x: Tensor, rate=0.5):
    """primary dropout defined for a scalar."""
    if np.random.rand() > rate:
        return x
    else:
        return 0.


def multihead_attention(
        query: Tensor,  # a single sequence is representated by a tensor.
        key: Tensor,
        value: Tensor,
        query_proj: Tensor,  # learnable parameter
        key_proj: Tensor,  # learnable parameter
        value_proj: Tensor,  # learnable parameter
        n_heads: int,
        mask=None,  # for decoder
):
    query_projected = np.matmul(query,
                                query_proj)  # [sequence_len x model_dim]
    key_projected = np.matmul(key, key_proj)  # [sequence_len x model_dim]
    value_projected = np.matmul(value,
                                value_proj)  # [sequence_len x model_dim]

    hidden_dim = model_dim // n_heads

    # TODO: It is better to distinguish runtime physically layout
    # transformation and logical view_as.
    query_reshape = query_projected.reshape(-1, hidden_dim, n_heads)
    key_reshape = key_projected.reshape(-1, hidden_dim, n_heads)
    value_reshape = value_projected.reshape(-1, hidden_dim, n_heads)

    pre_softmax = np.einsum('ijk, mjk -> imk', query_reshape, key_reshape)
    # pre_softmax: [seq_len, seq_len, n_heads]
    pre_softmax_scaled = pre_softmax * math.sqrt(model_dim)
    attn_weights = np.apply_along_axis(softmax_func, 0, pre_softmax_scaled)
    attn = np.einsum('mnk, npk -> mpk', attn_weights, value_reshape)
    res = attn.reshape(-1, model_dim)
    return res


def stacked_encoder(
        src_ids_batch: List[int],  # Immutable tensor array
        pos_enc: List[Tensor],  # Immutable tensor array
        encder_depth: int):
    batch_size = len(src_ids_batch)
    # foreach/map: iterate over sample in a batch without padding

    output_i = []  # a nested tensor array whose element is a tensor array.
    for i in range(batch_size):
        # a tensor array whose element is a tensor with a shape []
        output_j = []
        for j in range(encoder_depth):
            # Step 1: get positional encoding.

            # select rows or columns
            if j == 0:
                embed = embedding[src_ids_batch[i], :]
            else:
                input = output_j[j - 1]

            scaled_embed = embed * math.sqrt(model_dim) + pos_enc[i]
            # shape of embed: [seq_len, model_dim]

            # Step 2: multi-head attention.
            encoding = multihead_attention(scaled_embed, scaled_embed,
                                           scaled_embed, query_proj[j],
                                           key_proj[j], value_proj[j], n_heads)
            # shape of encoder: [seq_len, model_dim]

            # Add & Norm
            # skip connection, add positional embedding.
            positional_enc = encoding + pos_enc[i]
            # layer normalization.
            enc_normalized = np.apply_along_axis(norm_func, 1, positional_enc)
            layer_normed = layer_norm_scale[j] * enc_normalized + layer_norm_bias[j]

            # positional feedforward
            ff_proj1 = np.matmul(layer_normed,
                                 ff_proj_mat1[j]) + ff_proj_bias1[j]
            # element-wise apply
            ff_proj1_act = np.vectorize(relu)(ff_proj1)
            ff_proj1_dropout = np.vectorize(dropout)(ff_proj1_act)

            ff_proj2 = np.matmul(ff_proj1_dropout,
                                 ff_proj_mat2[j]) + ff_proj_bias2[j]
            output_j.append(ff_proj2)
        output_i.append(output_j)
    return output_i


if __name__ == "__main__":
    batch_size = 4
    max_src_len = 20

    seq, lens = gen_seq_batch(batch_size, max_src_len, vocab_size)
    pos_enc = positional_encoding(lens, model_dim)

    enc_vectors = stacked_encoder(seq, pos_enc, encoder_depth)
