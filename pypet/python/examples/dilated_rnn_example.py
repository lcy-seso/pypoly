from pprint import pprint
from typing import List, Tuple
import random

import torch
import torch.nn as nn
from torch import Tensor
from torch._utils_internal import get_source_lines_and_file

import context

import pypet
from pypet import VanillaRNNCell
from pypet import ReadWriteTensorArray
from pypet import ReadTensorArray


class DilatedRNN(nn.Module):
    def __init__(self, hidden_size, cells):
        super(DilatedRNN, self).__init__()

        self.register_buffer('init_state', torch.zeros((1, hidden_size)))

        self.cell1 = cells[0]
        self.cell2 = cells[1]
        self.cell3 = cells[2]
        self.cell4 = cells[3]

        self.cells = [self.cell1, self.cell2, self.cell3, self.cell4]

    def forward(self, input: ReadTensorArray, batch_size: int,
                seq_lens: List[int], depth: int, output: ReadWriteTensorArray):
        for i in range(batch_size):
            link_len = 1  # dilation rate
            for j in range(depth):
                seq_len = seq_lens[i]
                for k in range(seq_len):
                    if j == 0:
                        x = input[i][k]
                    else:
                        x = output[i][j - 1][k]

                    if k < link_len:
                        h_prev = self.init_state
                    else:
                        h_prev = output[i][j][k - link_len]

                    output[i][j][k] = self.cells[j](x, h_prev)
                link_len = 2 * link_len


def get_data(batch_size, input_size):
    min_len = 5
    max_len = 20

    seq_batch = []
    seq_lens = [random.randint(min_len, max_len) for _ in range(batch_size)]
    for l in seq_lens:
        a_seq = ReadTensorArray(
            [torch.randn(1, input_size, device=device) for _ in range(l)])
        seq_batch.append(a_seq)
    return ReadTensorArray(seq_batch), seq_lens


if __name__ == '__main__':
    random.seed(5)
    torch.manual_seed(5)

    device = 'cpu'

    batch_size = 4
    input_size = 16
    hidden_size = 16
    depth = 4

    cells = ReadTensorArray([
        VanillaRNNCell(input_size, hidden_size).to(device)
        for _ in range(depth)
    ])

    m = DilatedRNN(hidden_size, cells).to(device)
    seq_batch, seq_lens = get_data(batch_size, input_size)

    # Initialize output buffer. BUT do not use this way to declare array in
    # future, since it is hard to check whether the declaration is consistent
    # with loop computations.
    # TODO(Ying): provide a better interface to declare arrays.
    outputs = []
    for i in range(batch_size):
        output_j = []
        for j in range(depth):
            seq_len = seq_lens[i]
            for k in range(seq_len):
                output_k = ReadWriteTensorArray(
                    length=seq_len, tensor_shape=(1, hidden_size))
            output_j.append(output_k)
        outputs.append(ReadWriteTensorArray(input=output_j))
    outputs = ReadWriteTensorArray(outputs)

    m(seq_batch, batch_size, seq_lens, depth, outputs)
    parsed = pypet.scop(m)
