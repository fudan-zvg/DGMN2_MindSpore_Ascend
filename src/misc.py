"""
    The Functions in this file is used to replace the functions in timm which cannot be used in mindspore

"""

import mindspore.nn as nn
import mindspore.ops as ops

class DropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None, seed=0):
        super().__init__()
        self.keep_prob = 1 - drop_prob
        seed = min(seed, 0)  # always be 0
        self.rand = ops.UniformReal(seed=seed)  # seed must be 0, if set to other value, it's not rand for multiple call

    def construct(self, x):
        if self.training:
            random_tensor = self.rand((x.shape[0], 1, 1)) + self.keep_prob
            random_tensor = ops.floor(random_tensor)
            x = x / self.keep_prob
            x = x * random_tensor
        return x
