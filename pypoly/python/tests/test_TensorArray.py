import os
import sys
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import random
import torch
from tensor_array import ReadWriteTensorArray, ReadTensorArray


def gen_data():
    tensor_shape = (1, 16)
    min_len = 5
    max_len = 15
    batch_size = 4

    data = []
    lens = [random.randint(min_len, max_len) for _ in range(batch_size)]
    print('shape = [[%d], [%s]]' % (batch_size, ', '.join(map(str, lens))))

    for l in lens:
        a_seq = ReadTensorArray(
            [torch.randn(tensor_shape, device=device) for _ in range(l)])
        data.append(a_seq)
    return ReadTensorArray(data)


def test_ReadTensorArray():
    read_buff = ReadTensorArray(gen_data())


if __name__ == '__main__':
    random.seed(5)
    torch.manual_seed(5)

    device = 'cpu'
    test_ReadTensorArray()
