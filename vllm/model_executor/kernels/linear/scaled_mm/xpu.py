# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

import torch

from vllm.model_executor.kernels.linear import (  # noqa: E501
    Fp8BlockScaledMMLinearKernel,
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform


class XPUFP8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_xpu():
            return False, "XPUFP8ScaledMM only support on XPU"
        return True, None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        if c.weight_quant_key not in {kFp8StaticChannelSym, kFp8StaticTensorSym}:
            return (
                False,
                "XPUFP8ScaledMM only support per-channel and per-tensor quantization",
            )
        if c.weight_quant_key.dtype not in {torch.float8_e5m2, torch.float8_e4m3fn}:
            return False, "XPUFP8ScaledMM only support FP8 weight dtype"
        return True, None

    def __init__(
        self, c: FP8ScaledMMLinearLayerConfig, layer_param_names: Sequence[str]
    ) -> None:
        assert self.can_implement(c)[0]
        assert self.is_supported()[0]
        self.config = c
        self.layer_param_names = layer_param_names

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        replace_parameter(layer, "weight", layer.weight.data.t())

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = layer.weight
        weight_scale = layer.weight_scale
        return torch.ops._xpu_C.fp8_gemm_w8a16(x, weight, weight_scale, bias)

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        pass


class XPUFp8BlockScaledMMKernel(Fp8BlockScaledMMLinearKernel):
    """XPU blockwise FP8 w8a16 kernel backed by fp8_gemm_w8a16."""

    apply_input_quant = False

    @classmethod
    def is_supported(cls, compute_capability=None):
        if not current_platform.is_xpu():
            return False, "XPUFp8BlockScaledMM only supported on XPU"
        return True, None

    @classmethod
    def can_implement(cls, config):
        if not (hasattr(torch.ops, "_xpu_C") and hasattr(torch.ops._xpu_C, "fp8_gemm_w8a16")):
            return False, "fp8_gemm_w8a16 op is unavailable"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        params = self._get_layer_params(layer)
        # fp8_gemm_w8a16 expects weight in [in, out]
        replace_parameter(layer, params.WEIGHT, params.weight.data.t())

    def apply_weights(self, layer, x, bias=None, **kwargs):
        params = self._get_layer_params(layer)
        weight_scale = (
            params.weight_scale
            if params.weight_scale_inv is None
            else params.weight_scale_inv
        )
        out = torch.ops._xpu_C.fp8_gemm_w8a16(x, params.weight, weight_scale, bias)
        return out.to(self.config.out_dtype)

    def apply_block_scaled_mm(self, A, B, As, Bs):
        raise NotImplementedError
