# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Deformable Convolution operator V2"""

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops


class DeformUnfold(nn.Cell):
    """
    Deformable unfold operator

    Args:
        num_samples(int): The number of sampled nodes. Default: 9.
        kernel_size (int): Convolution window. Default: 3.
        stride (int): The distance of kernel moving. Default: 1.
    Returns:
        Tensor, detection of images(bboxes, score, keypoints and category id of each objects)
    """
    def __init__(self, num_samples=9, kernel_size=3, stride=1):
        super().__init__()
        self.stride = stride
        self.num_samples = num_samples
        self.meshgrid = ops.Meshgrid(indexing='ij')
        self.concat = ops.Concat(axis=1)

        if kernel_size % 2 == 0:
            raise ValueError("Only odd number is supported, but current kernel sizeis {}".format(kernel_size))

        # get p_n
        k = int(num_samples ** 0.5)
        self.k = k
        # (k)
        range_pn = nn.Range(-(k - 1) // 2, (k - 1) // 2 + 1)()
        # (k, k), (k, k) -> (2*k, k)
        p_n = ops.Concat(axis=0)(self.meshgrid((range_pn, range_pn)))
        # (2*k, k) -> (1, 2*k*k, 1, 1)
        self.p_n = ops.reshape(p_n, (1, 2 * num_samples, 1, 1))

        self.broadcast_to = ops.BroadcastTo((1, self.num_samples, -1, -1))

    def _get_offset_base(self, offset_shape):
        """
        get base position index from deformable shift of each kernel element.
        """
        # (n, 2*k*k, h, w)
        _, _, h, w = offset_shape

        range_h = nn.Range(self.k // 2, h * self.stride + 1, self.stride)()
        range_w = nn.Range(self.k // 2, w * self.stride + 1, self.stride)()
        # (h, w), (h, w)
        p_0_x, p_0_y = self.meshgrid((range_h, range_w))

        # (h, w) -> (1, k*k, h, w)
        p_0_x = self.broadcast_to(p_0_x)

        # (h, w) -> (1, k*k, h, w)
        p_0_y = self.broadcast_to(p_0_y)

        # (1, k*k, h, w), (1, k*k, h, w) -> (1, 2*k*k, h, w)
        p_0 = self.concat((p_0_x, p_0_y))
        # (1, 2*k*k, h, w) + (1, 2*k*k, 1, 1) -> (1, 2*k*k, h, w)
        p = p_0 + self.p_n

        return p

    def _get_feature_by_index(self, x, p_h, p_w):
        """gather feature by specified index"""
        # x (n, c, h_in, w_in)
        # p_h (n, h, w, k*k)
        # p_w (n, h, w, k*k)
        n, c, h_in, w_in = x.shape
        _, h, w, k2 = p_h.shape
        # (n, c, h_in, w_in) -> (n, h_in, w_in, c)
        x = ops.transpose(x, (0, 2, 3, 1))

        # the following is the opt for:
        # input(n, h_in, w_in, c), index_x/index_y(n, h, w, k*k) -> output(n, h, w, k*k, c)

        # (n, h_in, w_in, c) -> (n*h_in*w_in, c)
        x = ops.reshape(x, (n * h_in * w_in, c))

        # (n)
        idx_0_n = nn.Range(0, n, 1)()
        # (n, h, w, k*k) + (n, h, w, k*k) + (n, 1, 1, 1) -> (n, h, w, k*k)
        index = p_w + p_h * w_in + ops.reshape(idx_0_n, (n, 1, 1, 1)) * w_in * h_in

        # (n*h_in*w_in, c), (n, h, w, k*k) -> (n, h, w, k*k, c)
        x_offset = ops.gather(x, index, 0)
        # (n, h*w*k*k, c) -> (n, h*w, k*k, c)
        x_offset = ops.reshape(x_offset, (n, h * w, k2, c))
        # (n, h*w, k*k, c) -> (n, c, h*w, k*k)
        x_offset = ops.transpose(x_offset, (0, 3, 1, 2))
        # (n, c, h*w, k*k) -> (n, c, h, w, k*k)
        x_offset = ops.reshape(x_offset, (n, c, h, w, k2))

        return x_offset

    def construct(self, x, offset):
        """deformed sampling locations with augmented offsets"""
        # 0 ── h ──x
        # |
        # w
        # |
        # y

        # (n, c, h_in, w_in)
        n, _, h_in, w_in = x.shape

        # get absolute position of each pixel w.r.s to input feature map without offset
        # -> (1, 2*k*k, h, w)
        p_base = self._get_offset_base(offset.shape)
        # (1, 2*k*k, h, w) + (n, 2*k*k, h, w) -> (n, 2*k*k, h, w)
        p = p_base + offset

        # (n, 2*k*k, h, w) -> (n, h, w, 2*k*k)
        p = ops.transpose(p, (0, 2, 3, 1))
        p_lt = ops.cast(ops.floor(p), mindspore.int32)
        p_rb = p_lt + 1

        # (n, h, w, 2*k*k) -> (n, h, w, k*k), (n, h, w, k*k)
        k2 = p.shape[-1] // 2
        p_h = ops.tensor_slice(p, (0, 0, 0, 0), (n, h_in, w_in, k2)).clip(0, h_in - 1)
        p_w = ops.tensor_slice(p, (0, 0, 0, k2), (n, h_in, w_in, k2)).clip(0, w_in - 1)

        # (n, h, w, 2*k*k) -> (n, h, w, k*k), (n, h, w, k*k)
        p_lt_h = ops.tensor_slice(p_lt, (0, 0, 0, 0), (n, h_in, w_in, k2)).clip(0, h_in - 1)
        p_lt_w = ops.tensor_slice(p_lt, (0, 0, 0, k2), (n, h_in, w_in, k2)).clip(0, w_in - 1)

        # (n, h, w, 2*k*k) -> (n, h, w, k*k), (n, h, w, k*k)
        p_rb_h = ops.tensor_slice(p_rb, (0, 0, 0, 0), (n, h_in, w_in, k2)).clip(0, h_in - 1)
        p_rb_w = ops.tensor_slice(p_rb, (0, 0, 0, k2), (n, h_in, w_in, k2)).clip(0, w_in - 1)

        # perform bilinear interpolation
        # (n, h, w, k*k) -> (n, h, w, k*k)
        weight_lt = (1 - (p_h - p_lt_h)) * (1 - (p_w - p_lt_w))
        weight_rb = (p_h - p_lt_h) * (p_w - p_lt_w)
        weight_rt = (1 - (p_h - p_lt_h)) * (p_w - p_lt_w)
        weight_lb = (p_h - p_lt_h) * (1 - (p_w - p_lt_w))

        # (n, c, h_in, w_in), (n, h, w, k*k), (n, h, w, k*k) -> (n, c, h, w, k*k)
        x_p_lt = self._get_feature_by_index(x, p_lt_h, p_lt_w)
        x_p_rb = self._get_feature_by_index(x, p_rb_h, p_rb_w)
        x_p_lb = self._get_feature_by_index(x, p_rb_h, p_lt_w)
        x_p_rt = self._get_feature_by_index(x, p_lt_h, p_rb_w)

        # (n, h, w, k*k) -> (n, 1, h, w, k*k) * (n, c, h, w, k*k) -> (n, c, h, w, k*k)
        x_offset = (ops.expand_dims(weight_lt, 1) * x_p_lt +
                    ops.expand_dims(weight_rb, 1) * x_p_rb +
                    ops.expand_dims(weight_lb, 1) * x_p_lb +
                    ops.expand_dims(weight_rt, 1) * x_p_rt)

        return x_offset
