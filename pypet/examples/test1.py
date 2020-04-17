import pdb

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
from pypet import TensorArray


class MyModule(nn.Module):
    def __init__(self, hidden_size, cells):
        super(MyModule, self).__init__()

        self.register_buffer('init_state', torch.zeros((1, hidden_size)))

        self.cell1 = cells[0]
        self.cell2 = cells[1]
        self.cell3 = cells[2]

        self.cells = [self.cell1, self.cell2, self.cell3]

    def forward(self, input: List[List[Tensor]], batch_size: int,
                seq_lens: List[int], depth: int,
                output_size: int) -> TensorArray:

        # Declare array shape.
        output: TensorArray = TensorArray(batch_size, seq_lens, depth,
                                          output_size)

        for i in range(batch_size):
            seq_len: Tensor = seq_lens[i]
            for j in range(seq_len):  # data-dependent loop bound.
                for k in range(depth):
                    if j == 0:
                        h_prev = self.init_state
                    else:
                        h_prev = output.read(i, j - 1, k)

                    if k == 0:
                        x = input[i][j]
                    else:
                        x = output.read(i, j, k - 1)

                    h = self.cells[k](x, h_prev)
                    output.write(h, i, j, k)


if __name__ == '__main__':
    random.seed(5)
    torch.manual_seed(5)

    device = 'cpu'

    batch_size = 4
    min_len = 5
    max_len = 20

    input_size = 16
    hidden_size = 16

    seq_batch = []
    seq_lens = [random.randint(min_len, max_len) for _ in range(batch_size)]
    for l in seq_lens:
        a_seq = [torch.randn(1, input_size, device=device) for _ in range(l)]
        seq_batch.append(a_seq)

    depth = 3

    cells = [
        VanillaRNNCell(input_size, hidden_size).to(device)
        for _ in range(depth)
    ]

    outputs = []

    m = MyModule(hidden_size, cells).to(device)
    parsed = pypet.scop(m)
    # print(parsed)

    # This example cannot be run now, because of lacking necessary
    # implementations. Do not uncomment the below line.
    # m(seq_batch, batch_size, seq_lens, cells, depth, hidden_size)
