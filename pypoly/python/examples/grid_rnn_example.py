import random
from pprint import pprint
from typing import List
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch._utils_internal import get_source_lines_and_file

import context

import pypoly
from pypoly import ScopParser
from pypoly import CellArray
from pypoly import VanillaRNNCell
from pypoly import MutableArray
from pypoly import ImmutableArray


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

    def forward(self, src_seq_batch: ImmutableArray, src_seq_lens: List[int],
                trg_seq_batch: ImmutableArray, trg_seq_lens: List[int],
                batch_size: int, depth: int, output: MutableArray):
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
                            x_t = output[n][d - 1][i][j][0]
                            y_t = output[n][d - 1][i][j][1]

                        if i == 0:
                            state_x = self.init_state
                        else:
                            state_x = output[n][d][i - 1][j][0]

                        if j == 0:
                            state_y = self.init_state
                        else:
                            state_y = output[n][d][i][j - 1][1]

                        state = torch.cat([state_x, state_y], dim=1)
                        h_x = self.cells_x[d](x_t, state)
                        h_y = self.cells_y[d](y_t, state)

                        output[n][d][i][j][0] = h_x
                        output[n][d][i][j][1] = h_y


def get_data(batch_size, input_size):
    min_len = 5
    max_len = 20

    seq_batch = []
    seq_lens = [random.randint(min_len, max_len) for _ in range(batch_size)]
    for l in seq_lens:
        a_seq = [torch.randn(1, input_size, device=device) for _ in range(l)]
        seq_batch.append(a_seq)
    return seq_batch, seq_lens


if __name__ == '__main__':
    random.seed(5)
    torch.manual_seed(5)

    device = 'cpu'

    batch_size = 4
    hidden_size = 16
    input_size = hidden_size
    depth = 3
    grid_dim = 2

    cells_x = CellArray([
        VanillaRNNCell(input_size, hidden_size, grid_dim).to(device)
        for _ in range(depth)
    ])
    cells_y = CellArray([
        VanillaRNNCell(input_size, hidden_size, grid_dim).to(device)
        for _ in range(depth)
    ])

    m = GridRNN(hidden_size, cells_x, cells_y).to(device)

    src_seq_batch, src_lens = get_data(batch_size, input_size)
    trg_seq_batch, trg_lens = get_data(batch_size, input_size)

    # FIXME(Ying): a significant issue in current implementations.
    # The variable declared here outside the `forward` body MUST have exactly
    # the same name as the name of its usage in the body of `forward`.
    src_seq_batch = ImmutableArray(
        src_seq_batch,
        array_shape=[batch_size, max(src_lens)],
        tensor_shape=[1, input_size])

    # FIXME(Ying): a significant issue in current implementations.
    # The variable declared here outside the `forward` body MUST have exactly
    # the same name as the name of its usage in the body of `forward`.
    trg_seq_batch = ImmutableArray(
        trg_seq_batch,
        array_shape=[batch_size, max(trg_lens)],
        tensor_shape=[1, input_size])

    # FIXME(Ying): a significant issue in current implementations.
    # The variable declared here outside the `forward` body MUST have exactly
    # the same name as the name of its usage in the body of `forward`.
    output = MutableArray(
        array_shape=(batch_size, depth, max(src_lens), max(trg_lens),
                     grid_dim),
        tensor_shape=(1, hidden_size))

    with ScopParser(m) as p:
        p.add_int(batch_size, lower=1, upper=16)
        p.add_array(src_lens)
        p.add_array(src_seq_batch)
        p.add_array(trg_lens)
        p.add_array(trg_seq_batch)
        p.add_array(output)

        # uncomment to print the torch AST.
        # p.print_ast()
        # p.print_context()

        m = p.parse()

        m(src_seq_batch, src_lens, trg_seq_batch, trg_lens, batch_size, depth,
          output)
