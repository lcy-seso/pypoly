from typing import List, Tuple, Optional

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.nn.init import xavier_normal_, zeros_
from torch.nn import Module

__all__ = [
    'VanillaRNNCell',
]


class VanillaRNNCell(Module):
    """Cell computation of the Vanilla RNN.

    This implementation can be automatically differentiated.
    """

    def __init__(self, input_size, hidden_size, grid_dim=1):
        super(VanillaRNNCell, self).__init__()
        # learnable paramters
        self.W = Parameter(Tensor(input_size, hidden_size))
        self.U = Parameter(Tensor(hidden_size * grid_dim, hidden_size))
        self.b = Parameter(Tensor(1, hidden_size))

        self.init_weights()
        self.register_buffer('init_state', torch.zeros((1, hidden_size)))

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                xavier_normal_(p.data)
            else:
                zeros_(p.data)

    def forward(self, input: Tensor,
                h_prev: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            input, Tensor, input to current time step with a shape of
                [batch_size, input_size].
            h_prev, Tuple, hidden state of previous time step.

        Returns:
            Hidden states of current time step.
        """

        h_prev = self.init_state if h_prev is None else h_prev
        i2h = torch.mm(input, self.W)  # input-to-hidden projection
        h2h = torch.mm(h_prev, self.U)  # hidden-to-hidden_projection
        return torch.tanh(i2h + h2h + self.b)
