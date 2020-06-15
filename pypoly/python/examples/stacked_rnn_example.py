import sys
import random
from pprint import pprint
from typing import List
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch._utils_internal import get_source_lines_and_file

import context
from pypoly import ScopParser
from pypoly import VanillaRNNCell
from pypoly import CellArray
from pypoly import ReadWriteTensorArray
from pypoly import ReadTensorArray


class StackedLSTM(nn.Module):
    def __init__(self, hidden_size, cells):
        super(StackedLSTM, self).__init__()

        self.register_buffer('init_state', torch.zeros((1, hidden_size)))

        self.cell1 = cells[0]
        self.cell2 = cells[1]
        self.cell3 = cells[2]

        self.cells = [self.cell1, self.cell2, self.cell3]

    def forward(self, input: ReadTensorArray, batch_size: int,
                seq_lens: List[int], depth: int, output: ReadWriteTensorArray):
        for i in range(batch_size):
            seq_len = seq_lens[i]
            for j in range(seq_len):  # data-dependent loop bound.
                for k in range(depth):
                    if j == 0:
                        h_prev = self.init_state
                    else:
                        h_prev = output[i][j - 1][k]

                    if k == 0:
                        x = input[i][j]
                    else:
                        x = output[i][j][k - 1]
                    h = self.cells[k](x, h_prev)
                    output[i][j][k] = h


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
    input_size = 16
    hidden_size = 16
    depth = 3

    cells = CellArray([
        VanillaRNNCell(input_size, hidden_size).to(device)
        for _ in range(depth)
    ])

    seq_batch, seq_lens = get_data(batch_size, input_size)
    seq_batch = ReadTensorArray(
        seq_batch,
        array_shape=[batch_size, max(seq_lens)],
        tensor_shape=[1, input_size])

    # declare the output buffer.
    outputs = ReadWriteTensorArray(
        array_shape=(batch_size, max(seq_lens), depth),
        tensor_shape=(1, hidden_size))

    m = StackedLSTM(hidden_size, cells).to(device)

    with ScopParser(m) as p:
        p.add_int(batch_size, lower=0, upper=16)
        p.add_array(seq_lens)
        p.add_array(seq_batch)
        p.add_array(outputs)

        # uncomment to print the torch AST.
        # p.print_context()
        # p.print_ast()

        m = p.parse()

        m(seq_batch, batch_size, seq_lens, depth, outputs)
