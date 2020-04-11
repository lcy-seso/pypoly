import torch


class TensorArray(object):
    """TensorArray

    Args:
    """

    def __init__(self, *shape, dtype=torch.float32):
        if not (isinstance(shape, tuple) or isinstance(shape, list)):
            raise ValueError('shape should be a tuple or a list.')
            shape = list(shape)

        self.size = shape[-1]
        self.len = 1
        for dim_shape in shape[:-1][::-1]:
            if isinstance(dim_shape, int):
                self.len *= dim_shape
            elif isinstance(dim_shape, tuple) or isinstance(dim_shape, list):
                self.len *= sum(dim_shape)
            else:
                raise TypeError('Unrecognizable dimensional size.')

        # Initialize and allocate memory.

    def size(self):
        return self.size * self.len

    def gather(self, indices):
        pass

    def scatter(self, indices, value):
        pass

    def write(self, value, *indices):
        pass

    def read(self, *indices):
        pass
