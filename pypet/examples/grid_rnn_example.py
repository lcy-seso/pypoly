import os
import sys
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from pprint import pprint
from typing import List, Tuple
import random

import torch
import torch.nn as nn
from torch import Tensor
from torch._utils_internal import get_source_lines_and_file

import pypet
from pypet.cells import VanillaRNNCell
from pypet import ReadWriteTensorArray, ReadTensorArray


class GridRNN(nn.Module):
    def __init__(self, hidden_size, cells_x, cells_y):
        super(GridRNN, self).__init__()

        self.register_buffer('init_state', torch.zeros((1, hidden_size)))

        self.cell_x1 = cells_x[0]
        self.cell_x2 = cells_x[1]
        self.cell_x3 = cells_x[2]
        self.cells_x = [self.cell_x1, self.cell_x2, self.cell_x3]

        self.cell_y1 = cells_y[0]
        self.cell_y2 = cells_y[1]
        self.cell_y3 = cells_y[2]
        self.cells_y = [self.cell_y1, self.cell_y2, self.cell_y3]

    def forward(
            self,
            src_seq_batch: ReadTensorArray,
            src_seq_lens: List[int],
            trg_seq_batch: ReadWriteTensorArray,
            trg_seq_lens: List[int],
            batch_size: int,
            depth: int,
            output: ReadWriteTensorArray,
    ):
        for n in range(batch_size):
            for d in range(depth):

                src_len = src_lens[n]
                trg_len = trg_lens[n]
                for i in range(src_len):
                    for j in range(trg_len):
                        if d == 0:
                            x_t = src_seq_batch[n][i]
                            y_t = trg_seq_batch[n][j]
                        else:
                            x_t = output[n][d - 1][i][j * 2]
                            y_t = output[n][d - 1][i][j * 2 + 1]

                        if i == 0:
                            state_x = self.init_state
                        else:
                            state_x = output[n][d][i - 1][(j - 1) * 2]

                        if j == 0:
                            state_y = self.init_state
                        else:
                            state_y = output[n][d][i][(j - 1) * 2 + 1]

                        state = torch.cat([state_x, state_y], dim=1)
                        h_x = self.cells_x[d](x_t, state_x)
                        h_y = self.cells_y[d](y_t, state_y)

                        output[n][d][i][j * 2] = h_x
                        output[n][d][i][j * 2 + 1] = h_y


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
    depth = 3

    cells_x = ReadTensorArray([
        VanillaRNNCell(input_size, hidden_size).to(device)
        for _ in range(depth)
    ])
    cells_y = ReadTensorArray([
        VanillaRNNCell(input_size, hidden_size).to(device)
        for _ in range(depth)
    ])

    m = GridRNN(hidden_size, cells_x, cells_y).to(device)

    src_seq_batch, src_lens = get_data(batch_size, input_size)
    trg_seq_batch, trg_lens = get_data(batch_size, input_size)

    # Initialize output buffer. BUT do not use this way to declare array in
    # future, since it is hard to check whether the declaration is consistent
    # with loop computations.
    # TODO(Ying): provide a better interface to declare arrays.
    outputs = []
    grid_dim = 2
    for n in range(batch_size):
        output_d = []
        for d in range(depth):
            src_len = src_lens[n]
            trg_len = trg_lens[n]
            output_i = []
            for i in range(src_len):
                output_j = ReadWriteTensorArray(
                    length=trg_len * grid_dim, tensor_shape=(1, hidden_size))
                output_i.append(output_j)
            output_d.append(ReadWriteTensorArray(input=output_i))
        outputs.append(ReadWriteTensorArray(input=output_d))

    m(src_seq_batch, src_lens, trg_seq_batch, trg_lens, batch_size, depth,
      outputs)
    parsed = pypet.scop(m)
