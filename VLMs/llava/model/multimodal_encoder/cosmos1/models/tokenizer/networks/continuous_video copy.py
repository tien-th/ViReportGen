# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""The causal continuous video tokenizer with VAE or AE formulation for 3D data.."""
from collections import OrderedDict, namedtuple
import math
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from loguru import logger as logging


from loguru import logger as logging
from torch import nn

from cosmos1.models.tokenizer.modules import Decoder3DType, Encoder3DType

NetworkEval = namedtuple("NetworkEval", ["reconstructions", "posteriors", "latent"])


class CausalContinuousVideoTokenizer(nn.Module):
    def __init__(self, z_channels: int, z_factor: int, latent_channels: int, **kwargs) -> None:
        super().__init__()
        self.name = kwargs.get("name", "CausalContinuousVideoTokenizer")
        self.latent_channels = latent_channels

        encoder_name = kwargs.get("encoder", Encoder3DType.BASE.name)
        self.encoder = Encoder3DType[encoder_name].value(z_channels=z_factor * z_channels, **kwargs)
        decoder_name = kwargs.get("decoder", Decoder3DType.BASE.name)
        self.decoder = Decoder3DType[decoder_name].value(z_channels=z_channels, **kwargs)

        self.quant_conv = CausalConv3d(
            z_factor * z_channels,
            z_factor * latent_channels,
            kernel_size=1,
            padding=0,
        )
        self.post_quant_conv = CausalConv3d(latent_channels, z_channels, kernel_size=1, padding=0)

        formulation_name = kwargs.get("formulation", ContinuousFormulation.AE.name)
        self.distribution = ContinuousFormulation[formulation_name].value()
        logging.info(f"{self.name} based on {formulation_name} formulation, with {kwargs}.")

        num_parameters = sum(param.numel() for param in self.parameters())
        logging.info(f"model={self.name}, num_parameters={num_parameters:,}")
        logging.info(f"z_channels={z_channels}, latent_channels={self.latent_channels}.")

    def encoder_jit(self):
        return nn.Sequential(
            OrderedDict(
                [
                    ("encoder", self.encoder),
                    ("quant_conv", self.quant_conv),
                    ("distribution", self.distribution),
                ]
            )
        )

    def decoder_jit(self):
        return nn.Sequential(
            OrderedDict(
                [
                    ("post_quant_conv", self.post_quant_conv),
                    ("decoder", self.decoder),
                ]
            )
        )

    def last_decoder_layer(self):
        return self.decoder.conv_out

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        return self.distribution(moments)

    def decode(self, z):
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def forward(self, input):
        latent, posteriors = self.encode(input)
        reconstructions = self.decode(latent)
        if self.training:
            return dict(
                reconstructions=reconstructions,
                posteriors=posteriors,
                latent=latent,
            )
        return NetworkEval(
            reconstructions=reconstructions,
            posteriors=posteriors,
            latent=latent,
        )


# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""The model definition for 3D layers

Adapted from: https://github.com/lucidrains/magvit2-pytorch/blob/
9f49074179c912736e617d61b32be367eb5f993a/magvit2_pytorch/magvit2_pytorch.py#L889

[MIT License Copyright (c) 2023 Phil Wang]
https://github.com/lucidrains/magvit2-pytorch/blob/
9f49074179c912736e617d61b32be367eb5f993a/LICENSE
"""






class Decoder3DType(Enum):
    BASE = DecoderBase
    FACTORIZED = DecoderFactorized



class IdentityDistribution(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, parameters):
        return parameters, (torch.tensor([0.0]), torch.tensor([0.0]))


class GaussianDistribution(torch.nn.Module):
    def __init__(self, min_logvar: float = -30.0, max_logvar: float = 20.0):
        super().__init__()
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar

    def sample(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(mean)

    def forward(self, parameters):
        mean, logvar = torch.chunk(parameters, 2, dim=1)
        logvar = torch.clamp(logvar, self.min_logvar, self.max_logvar)
        return self.sample(mean, logvar), (mean, logvar)


class ContinuousFormulation(Enum):
    VAE = GaussianDistribution
    AE = IdentityDistribution


def divisible_by(num: int, den: int) -> bool:
    return (num % den) == 0

def is_odd(n: int) -> bool:
    return not divisible_by(n, 2)

from typing import Any
def cast_tuple(t: Any, length: int = 1) -> Any:
    return t if isinstance(t, tuple) else ((t,) * length)


class CausalConv3d(nn.Module):
    def __init__(
        self,
        chan_in: int = 1,
        chan_out: int = 1,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        pad_mode: str = "constant",
        **kwargs,
    ):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)

        dilation = kwargs.pop("dilation", 1)
        stride = kwargs.pop("stride", 1)
        time_stride = kwargs.pop("time_stride", 1)
        time_dilation = kwargs.pop("time_dilation", 1)
        padding = kwargs.pop("padding", 1)

        self.pad_mode = pad_mode
        time_pad = time_dilation * (time_kernel_size - 1) + (1 - time_stride)
        self.time_pad = time_pad

        self.spatial_pad = (padding, padding, padding, padding)

        stride = (time_stride, stride, stride)
        dilation = (time_dilation, dilation, dilation)
        self.conv3d = nn.Conv3d(
            chan_in,
            chan_out,
            kernel_size,
            stride=stride,
            dilation=dilation,
            **kwargs,
        )

    def _replication_pad(self, x: torch.Tensor) -> torch.Tensor:
        x_prev = x[:, :, :1, ...].repeat(1, 1, self.time_pad, 1, 1)
        x = torch.cat([x_prev, x], dim=2)
        padding = self.spatial_pad + (0, 0)
        return F.pad(x, padding, mode=self.pad_mode, value=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._replication_pad(x)
        return self.conv3d(x)

