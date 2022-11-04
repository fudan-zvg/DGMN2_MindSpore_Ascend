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
"""Vision Transformer implementation."""

from importlib import import_module
import math
import numpy as np
from easydict import EasyDict as edict
import mindspore
import mindspore.nn as nn
from mindspore.common.initializer import initializer
import mindspore.ops as ops

from src.misc import DropPath
from src.dcn_v2 import DeformUnfold


class DGMN2Config:
    """
    DGMN2Config
    """

    def __init__(self, configs):
        self.configs = configs

        # network init
        self.network_dropout_rate = 0.1
        self.network = DGMN2

        # body
        self.body_drop_path_rate = 0.1

        # body attention
        self.attention_init = mindspore.common.initializer.TruncatedNormal(sigma=0.02)
        self.attention_activation = mindspore.nn.Softmax()
        self.attention_dropout_rate = 0.0
        self.project_dropout_rate = 0.0
        self.attention = DGMN2Attention

        # body feedforward
        self.feedforward_init = mindspore.common.initializer.TruncatedNormal(sigma=0.02)
        self.feedforward_activation = mindspore.nn.GELU()
        self.feedforward_dropout_rate = 0.0
        self.feedforward = FeedForward

        # head
        # self.head = origin_head
        self.head_init = mindspore.common.initializer.TruncatedNormal(sigma=0.02)
        self.head_dropout_rate = 0.1
        self.head_activation = mindspore.nn.GELU()


class ResidualCell(nn.Cell):
    """Cell which implements x + f(x) function."""
    def __init__(self, cell):
        super().__init__()
        self.cell = cell

    def construct(self, x):
        return self.cell(x) + x


class RelPosEmb(nn.Cell):
    def __init__(self, fmap_size, dim_head, num_samples):
        super().__init__()
        height, width = fmap_size
        scale = dim_head ** -0.5
        self.num_samples = num_samples

        normal = mindspore.common.initializer.Normal(sigma=1.0)
        self.rel_height = mindspore.Parameter(
            initializer(normal, (height + num_samples - 1, dim_head)),
            name="rel_height") * scale
        self.rel_width = mindspore.Parameter(
            initializer(normal, (width + num_samples - 1, dim_head)),
            name="rel_width") * scale

        self.zeros = ops.Zeros()
        self.concat_3 = ops.Concat(3)
        self.concat_2 = ops.Concat(2)

    def rel_to_abs(self, x):
        b, h, l, c = x.shape
        x = self.concat_3((x, self.zeros((b, h, l, 1), x.dtype)))
        x = ops.reshape(x, (b, h, l * (c + 1)))
        x = self.concat_2((x, self.zeros((b, h, self.num_samples - 1), x.dtype)))
        x = ops.reshape(x, (b, h, l + 1, self.num_samples + l - 1))
        x = ops.tensor_slice(x, (0, 0, 0, l - 1), (b, h, l, self.num_samples))
        return x

    def relative_logits_1d(self, q, rel_k):
        logits = ops.matmul(q, ops.transpose(rel_k, (1, 0)))
        b, h, x, y, r = logits.shape
        logits = ops.reshape(logits, (b, h * x, y, r))

        logits = self.rel_to_abs(logits)
        return logits

    def construct(self, q):
        rel_logits_w = self.relative_logits_1d(q, self.rel_width)

        q = ops.transpose(q, (0, 1, 3, 2, 4))
        rel_logits_h = self.relative_logits_1d(q, self.rel_height)

        return rel_logits_w + rel_logits_h


class PatchEmbed_stage1(nn.Cell):
    """ Image to Patch Embedding
    """

    def __init__(self, in_chans=3, embed_dim=768, mid_embed_dim=384):
        super().__init__()
        he_uniform = mindspore.common.initializer.HeUniform(math.sqrt(5))
        self.conv1 = nn.Conv2d(in_channels=in_chans, out_channels=mid_embed_dim, kernel_size=3, stride=2,
                               pad_mode='pad', padding=1, has_bias=False, weight_init=he_uniform)
        self.norm1 = nn.BatchNorm2d(mid_embed_dim)
        self.conv2 = nn.Conv2d(in_channels=mid_embed_dim, out_channels=mid_embed_dim, kernel_size=3, stride=1,
                               pad_mode='pad', padding=1, has_bias=False, weight_init=he_uniform)
        self.norm2 = nn.BatchNorm2d(mid_embed_dim)
        self.conv3 = nn.Conv2d(in_channels=mid_embed_dim, out_channels=embed_dim, kernel_size=3, stride=2,
                               pad_mode='pad', padding=1, has_bias=False, weight_init=he_uniform)
        self.norm3 = nn.BatchNorm2d(embed_dim)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        x = self.relu(self.norm3(self.conv3(x)))

        return x


class PatchEmbed(nn.Cell):
    """ Image to Patch Embedding
    """

    def __init__(self, in_chans=3, embed_dim=768):
        super().__init__()
        he_uniform = mindspore.common.initializer.HeUniform(math.sqrt(5))
        self.conv = nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=3, stride=2, pad_mode='pad',
                              padding=1, has_bias=False, weight_init=he_uniform)
        self.norm = nn.BatchNorm2d(embed_dim)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.relu(self.norm(self.conv(x)))

        return x


class DGMN2Attention(nn.Cell):
    """Attention layer implementation."""

    def __init__(self, dgmn2_config, d_model, num_heads, fea_size):
        super().__init__()
        assert d_model % num_heads == 0, f"dim {d_model} should be divided by num_heads {num_heads}."
        dim_head = d_model // num_heads

        initialization = dgmn2_config.attention_init
        activation = dgmn2_config.attention_activation  # softmax
        attn_drop = dgmn2_config.attention_dropout_rate
        proj_drop = dgmn2_config.project_dropout_rate

        inner_dim = num_heads * dim_head
        self.dim_head = dim_head
        self.num_heads = num_heads
        self.scale = dim_head ** -0.5
        self.H, self.W = fea_size

        # self.to_qkv = nn.Dense(d_model, inner_dim * 3, weight_init=initialization)
        self.to_q = nn.Dense(d_model, inner_dim, weight_init=initialization)
        self.to_k = nn.Dense(d_model, inner_dim, weight_init=initialization)
        self.to_v = nn.Dense(d_model, inner_dim, weight_init=initialization)

        # sample
        self.num_samples = 9
        self.to_offset = nn.Dense(dim_head, self.num_samples * 2, weight_init=initialization)
        self.unfold = DeformUnfold(self.num_samples)

        # relative position
        self.pos_emb = RelPosEmb(fea_size, dim_head, self.num_samples)

        self.proj = nn.Dense(inner_dim, d_model, weight_init=initialization)
        self.attn_drop = nn.Dropout(1 - attn_drop)
        self.proj_drop = nn.Dropout(1 - proj_drop)
        self.activation = activation

        # auxiliary functions
        self.unstack = ops.Unstack(0)
        self.q_matmul_k = ops.BatchMatMul(transpose_b=True)
        self.attn_matmul_v = ops.BatchMatMul()

    def construct(self, x):
        '''x size - BxNxd_model'''
        bs, seq_len, d_model, h, d = x.shape[0], x.shape[1], x.shape[2], self.num_heads, self.dim_head

        # qkv = ops.transpose(ops.reshape(self.to_qkv(x), (bs, seq_len, 3, h, d)), (2, 0, 3, 1, 4))
        # q, k, v = self.unstack(qkv)  # [bs, h, seq_len, d]
        q = ops.transpose(ops.reshape(self.to_q(x), (bs, seq_len, h, d)), (0, 2, 1, 3))
        k = ops.transpose(ops.reshape(self.to_k(x), (bs, seq_len, h, d)), (0, 2, 3, 1)) # [bs, h, d, seq_len]
        v = ops.transpose(ops.reshape(self.to_v(x), (bs, seq_len, h, d)), (0, 2, 3, 1))

        offset = self.to_offset(ops.reshape(x, (bs, seq_len, h, d)))  # [bs, seq_len, h, self.num_samples*2]
        offset = ops.reshape(ops.transpose(offset, (0, 2, 3, 1)), (bs * h, self.num_samples * 2, self.H, self.W))

        k = ops.reshape(k, (bs * h, d, self.H, self.W))
        v = ops.reshape(v, (bs * h, d, self.H, self.W))
        k = ops.reshape(ops.transpose(self.unfold(k, offset), (0, 2, 3, 4, 1)), (bs, h, seq_len, self.num_samples, d))
        v = ops.reshape(ops.transpose(self.unfold(v, offset), (0, 2, 3, 4, 1)), (bs, h, seq_len, self.num_samples, d))

        # q : [bs, h, seq_len, d]  k : [bs, h, seq_len, d, self.num_samples]
        attn = self.q_matmul_k(ops.expand_dims(q, 3), k) * self.scale

        attn_pos = ops.reshape(self.pos_emb(ops.reshape(q, (bs * h, 1, self.H, self.W, d))),
                               (bs, h, seq_len, 1, self.num_samples))
        attn = attn + attn_pos

        attn = self.activation(attn)
        attn = self.attn_drop(attn)  # [bs, h, seq_len, 1, self.num_samples]

        # attn : [bs, h, seq_len, 1, self.num_samples] v : [bs, h, seq_len, d, self.num_samples]
        x = self.attn_matmul_v(attn, v)
        x = ops.reshape(ops.transpose(x, (0, 2, 1, 3, 4)), (bs, seq_len, d_model))

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class FeedForward(nn.Cell):
    """FeedForward layer implementation."""

    def __init__(self, dgmn2_config, d_model, mlp_ratio):
        super().__init__()

        hidden_dim = int(d_model * mlp_ratio)

        initialization = dgmn2_config.feedforward_init
        activation = dgmn2_config.feedforward_activation
        dropout_rate = dgmn2_config.feedforward_dropout_rate

        self.ff1 = nn.Dense(d_model, hidden_dim, weight_init=initialization)
        self.activation = activation
        self.dropout = nn.Dropout(keep_prob=1. - dropout_rate)
        self.ff2 = nn.Dense(hidden_dim, d_model, weight_init=initialization)

    def construct(self, x):
        x = self.ff1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.ff2(x)
        x = self.dropout(x)
        return x


class DGMN2Block(nn.Cell):
    """Deferent Scale of Transformer implementation."""

    def __init__(self, dgmn2_config, d_model, num_heads, fea_size, mlp_ratio, dpr):
        super().__init__()

        attention = dgmn2_config.attention(dgmn2_config, d_model, num_heads, fea_size)
        feedforward = dgmn2_config.feedforward(dgmn2_config, d_model, mlp_ratio)
        if dpr > 0:
            self.layers = nn.SequentialCell([
                ResidualCell(nn.SequentialCell([nn.LayerNorm((d_model,)),
                                                attention,
                                                DropPath(dpr)])),
                ResidualCell(nn.SequentialCell([nn.LayerNorm((d_model,)),
                                                feedforward,
                                                DropPath(dpr)]))
            ])
        else:
            self.layers = nn.SequentialCell([
                ResidualCell(nn.SequentialCell([nn.LayerNorm((d_model,)),
                                                attention])),
                ResidualCell(nn.SequentialCell([nn.LayerNorm((d_model,)),
                                                feedforward]))
            ])

    def construct(self, x):
        return self.layers(x)


class BasicLayer(nn.Cell):

    def __init__(self, dgmn2_config, layer_i, in_chans, embed_dim, d_model, num_heads, fea_size, mlp_ratio, depth, dpr):
        super().__init__()

        # build blocks
        if layer_i == 0:
            self.patch_embed = PatchEmbed_stage1(in_chans=in_chans, embed_dim=embed_dim, mid_embed_dim=embed_dim // 2)
        else:
            self.patch_embed = PatchEmbed(in_chans=in_chans, embed_dim=embed_dim)
        self.blocks = nn.SequentialCell([
            DGMN2Block(dgmn2_config, d_model, num_heads, fea_size, mlp_ratio, dpr[i])
            for i in range(depth)])

    def construct(self, x):
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = ops.transpose(ops.reshape(x, (B, C, H * W)), (0, 2, 1))
        x = self.blocks(x)
        x = ops.reshape(ops.transpose(x, (0, 2, 1)), (B, C, H, W))

        return x


class DGMN2(nn.Cell):

    def __init__(self, dgmn2_config):
        super().__init__()

        d_model = dgmn2_config.configs.d_model
        num_heads = dgmn2_config.configs.num_heads
        mlp_ratios = dgmn2_config.configs.mlp_ratios
        in_chans = dgmn2_config.configs.in_chans
        num_classes = dgmn2_config.configs.num_classes
        initialization = dgmn2_config.head_init

        # stochastic depth
        drop_path_rate = dgmn2_config.body_drop_path_rate
        depths = dgmn2_config.configs.depths
        dpr = [x.item() for x in np.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        in_chans = [in_chans, d_model[0], d_model[1], d_model[2]]
        embed_dims = [d_model[0], d_model[1], d_model[2], d_model[3]]
        self.layers = nn.SequentialCell([BasicLayer(
            dgmn2_config, i, in_chans[i], embed_dims[i],
            d_model[i], num_heads[i],
            (224 // (2 ** (i + 2)), 224 // (2 ** (i + 2))),
            mlp_ratios[i], depths[i],
            dpr[sum(depths[:i]):sum(depths[:i + 1])])
                                         for i in range(len(depths))])

        self.avgpool = ops.AvgPool(pad_mode="valid", kernel_size=7, strides=1)

        # classification head
        self.head = nn.Dense(d_model[3], num_classes, weight_init=initialization) if num_classes > 0 else ops.Identity()

    def construct(self, x):
        x = self.layers(x)

        B, C, _, _ = x.shape
        x = ops.reshape(self.avgpool(x), (B, C))

        x = self.head(x)

        return x


def load_function(func_name):
    """Load function using its name."""
    modules = func_name.split(".")
    if len(modules) > 1:
        module_path = ".".join(modules[:-1])
        name = modules[-1]
        module = import_module(module_path)
        return getattr(module, name)
    return func_name


dgmn2_cfg = edict({
    'd_model': (64, 128, 320, 512),
    'depths': (2, 2, 2, 2),
    'num_heads': (1, 2, 5, 8),
    'mlp_ratios': (8, 8, 4, 4),
    'image_size': 224,
    'in_chans': 3,
    'num_classes': 1000,
})


def dgmn2_tiny(args):
    """dgmn2_tiny"""
    dgmn2_cfg.d_model = (64, 128, 320, 512)
    dgmn2_cfg.depths = (2, 2, 2, 2)
    dgmn2_cfg.num_heads = (1, 2, 5, 8)
    dgmn2_cfg.mlp_ratios = (8, 8, 4, 4)
    dgmn2_cfg.image_size = args.train_image_size
    dgmn2_cfg.in_chans = 3
    dgmn2_cfg.num_classes = args.class_num

    if args.dgmn2_config_path != '':
        print("get dgmn2_config_path")
        dgmn2_config = load_function(args.dgmn2_config_path)(dgmn2_cfg)
    else:
        print("get default_dgmn2_cfg")
        dgmn2_config = DGMN2Config(dgmn2_cfg)

    model = dgmn2_config.network(dgmn2_config)
    return model


def dgmn2_small(args):
    """dgmn2_small"""
    dgmn2_cfg.d_model = (64, 128, 320, 512)
    dgmn2_cfg.depths = (3, 4, 6, 3)
    dgmn2_cfg.num_heads = (1, 2, 5, 8)
    dgmn2_cfg.mlp_ratios = (8, 8, 4, 4)
    dgmn2_cfg.image_size = args.train_image_size
    dgmn2_cfg.in_chans = 3
    dgmn2_cfg.num_classes = args.class_num

    if args.dgmn2_config_path != '':
        print("get dgmn2_config_path")
        dgmn2_config = load_function(args.dgmn2_config_path)(dgmn2_cfg)
    else:
        print("get default_dgmn2_cfg")
        dgmn2_config = DGMN2Config(dgmn2_cfg)

    model = dgmn2_config.network(dgmn2_config)
    return model


def dgmn2_medium(args):
    """dgmn2_medium"""
    dgmn2_cfg.d_model = (64, 128, 320, 512)
    dgmn2_cfg.depths = (3, 4, 18, 3)
    dgmn2_cfg.num_heads = (1, 2, 5, 8)
    dgmn2_cfg.mlp_ratios = (8, 8, 4, 4)
    dgmn2_cfg.image_size = args.train_image_size
    dgmn2_cfg.in_chans = 3
    dgmn2_cfg.num_classes = args.class_num

    if args.dgmn2_config_path != '':
        print("get dgmn2_config_path")
        dgmn2_config = load_function(args.dgmn2_config_path)(dgmn2_cfg)
    else:
        print("get default_dgmn2_cfg")
        dgmn2_config = DGMN2Config(dgmn2_cfg)

    model = dgmn2_config.network(dgmn2_config)
    return model


def dgmn2_large(args):
    """dgmn2_large"""
    dgmn2_cfg.d_model = (64, 128, 320, 512)
    dgmn2_cfg.depths = (3, 8, 27, 3)
    dgmn2_cfg.num_heads = (1, 2, 5, 8)
    dgmn2_cfg.mlp_ratios = (8, 8, 4, 4)
    dgmn2_cfg.image_size = args.train_image_size
    dgmn2_cfg.in_chans = 3
    dgmn2_cfg.num_classes = args.class_num

    if args.dgmn2_config_path != '':
        print("get dgmn2_config_path")
        dgmn2_config = load_function(args.dgmn2_config_path)(dgmn2_cfg)
    else:
        print("get default_dgmn2_cfg")
        dgmn2_config = DGMN2Config(dgmn2_cfg)

    model = dgmn2_config.network(dgmn2_config)
    return model


def get_network(backbone_name, args):
    """get_network"""
    if backbone_name == 'dgmn2_tiny':
        backbone = dgmn2_tiny(args)
    elif backbone_name == 'dgmn2_small':
        backbone = dgmn2_small(args)
    elif backbone_name == 'dgmn2_medium':
        backbone = dgmn2_medium(args)
    elif backbone_name == 'dgmn2_large':
        backbone = dgmn2_large(args)
    else:
        raise NotImplementedError

    return backbone
