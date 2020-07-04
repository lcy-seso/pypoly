import pdb

import collections
import numpy as np
from typing import List

import torch
from torch.nn.parameter import Parameter

import context

import pypoly
from pypoly import VanillaRNNCell
from pypoly import ImmutableTensor
from pypoly import MutableArray
from pypoly import ImmutableArray

from pypoly import meta
from pypoly import compound
from pypoly import functional as F

from data_utils import *

# ============  hyper parameters constants independent of data ============== #
embed_dim = 6
word_num = 25
proj_dim = 6
hidden_dim = 6
depth = 3
# ============  hyper parameters constants independent of data ============== #

# =====  Learnable parameters that have static shapes and global liveness. ==#
# Learnable parameters are mutable. They will be updated by optimizer.
embed_table = Parameter(torch.randn(word_num, embed_dim))
proj_matrix = Parameter(torch.randn(embed_dim, proj_dim))

in2h_proj = []
h2h_proj = []
for i in range(depth):
    in2h_proj.append(Parameter(torch.randn(proj_dim, hidden_dim)))
    h2h_proj.append(Parameter(torch.randn(hidden_dim, hidden_dim)))

sen_in2h = Parameter(torch.randn(proj_dim, hidden_dim))
sen_h2h = Parameter(torch.randn(hidden_dim, hidden_dim))

# =====  Learnable parameters that have static shapes and global liveness. ==#


# user defined computations that is consisted of meta-operators is pure.
def cell(input: ImmutableTensor, h: ImmutableTensor, in2h: ImmutableTensor,
         h2h: ImmutableTensor) -> ImmutableTensor:
    in2h_proj = compound.MatMul(input, in2h)
    h2h_proj = compound.MatMul(h, h2h)
    return in2h_proj + h2h_proj


def model1(dataset):
    # FIXME: to hidden the declaration of the output buffere. It can be infered
    # from the computation to avoid error.
    output = MutableArray(
        array_shape=(10, 10, 10, 10, depth), tensor_shape=(1, proj_dim))

    for i, batch in enumerate(dataset):
        for j, passage in enumerate(batch):
            for k, sentence in enumerate(passage):
                for m, word in enumerate(sentence):

                    embed = meta.SelectRows([word], embed_table)
                    proj = compound.MatMul(embed, proj_matrix)

                    for d in range(depth):
                        if d == 0:
                            x = proj
                        else:
                            x = output[i][j][k][m][d - 1]

                        if m == 0:
                            h_prev = torch.zeros(1, hidden_dim)
                        else:
                            h_prev = output[i][j][k][m - 1][d]

                        h = cell(x, h_prev, in2h_proj[d], h2h_proj[d])
                        h = torch.tanh(h)
                        output[i][j][k][m][d] = h

            sentence_out = output[i][j]

    return output


def model2(dataset):
    output_i = []
    for i, batch in enumerate(dataset):
        output_j = []
        for j, passage in enumerate(batch):
            output_k = []
            for k, sentence in enumerate(passage):
                output_m1 = []
                output_m2 = []
                for m, word in enumerate(sentence):  # iterate over a sequence

                    embed = meta.SelectRows([word], embed_table)
                    proj = compound.MatMul(embed, proj_matrix)

                    output_d = []
                    for d in range(depth):  # depth
                        if d == 0:
                            x = proj
                        else:
                            x = output_d[d - 1]

                        if m == 0:
                            h_prev = torch.zeros(1, hidden_dim)
                        else:
                            h_prev = output_m1[m - 1][d]

                        h = cell(x, h_prev, in2h_proj[d], h2h_proj[d])
                        h = torch.tanh(h)  # activation.

                        # line 104 to 109 canbe warped into a functional "for"
                        output_d.append(h)

                    last = output_d[-1]

                    output_m1.append(output_d)
                    output_m2.append(last)

                output_m3 = []
                for n, last in enumerate(output_m2):
                    if n == 0:
                        h = torch.zeros(1, hidden_dim)
                    else:
                        h = output_m3[n - 1]

                    h = cell(output_m2[n], h, sen_in2h, sen_h2h)
                    output_m3.append(h)

                last = output_m3[-1]
                output_k.append(last)
            output_j.append(output_k)
        output_i.append(output_j)
    return output_i


if __name__ == "__main__":
    dataset = gen_random_dataset()
    # output = model1(dataset)
    output = model1(dataset)
