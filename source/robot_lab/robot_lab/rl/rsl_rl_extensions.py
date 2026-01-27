# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Iterable


import torch
import torch.nn as nn
import torch.nn.functional as F

from rsl_rl.utils import unpad_trajectories


def _normalize_layer_dims(hidden_dim: int | list[int] | tuple[int, ...], num_layers: int) -> list[int]:
    if isinstance(hidden_dim, (list, tuple)):
        dims = [int(v) for v in hidden_dim]
        if len(dims) != num_layers:
            raise ValueError(
                f"rnn_hidden_dim list length ({len(dims)}) must match rnn_num_layers ({num_layers})."
            )
        return dims
    return [int(hidden_dim)] * num_layers


def _last_hidden_dim(hidden_dim: int | list[int] | tuple[int, ...]) -> int:
    if isinstance(hidden_dim, (list, tuple)):
        return int(hidden_dim[-1])
    return int(hidden_dim)


class _RWKV6LayerNorm(nn.Module):
    def __init__(self, dim: int, bias: bool = False, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)


def _rwkv6_time_shift(x: torch.Tensor, prev: torch.Tensor | None) -> torch.Tensor:
    # x: [B, T, D]
    shifted = torch.zeros_like(x)
    if x.shape[1] > 1:
        shifted[:, 1:] = x[:, :-1]
    if prev is not None:
        shifted[:, 0] = prev
    return shifted


def _rwkv6_wkv_op(
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # r,k,v,w: [B, H, T, K], u: [1, H, K, 1], state: [B, H, K, K]
    batch_size, n_head, seq_len, head_dim = k.shape
    value_dim = v.shape[-1]
    if state is None:
        state = torch.zeros(
            batch_size,
            n_head,
            head_dim,
            value_dim,
            device=k.device,
            dtype=k.dtype,
        )
    outputs = []
    for t in range(seq_len):
        kt = k[:, :, t]
        vt = v[:, :, t]
        kv = kt.unsqueeze(-1) * vt.unsqueeze(-2)
        out = torch.einsum("bhk,bhkv->bhv", r[:, :, t], state + u * kv)
        wt = w[:, :, t].unsqueeze(-1)
        state = state * wt + kv
        outputs.append(out)
    out = torch.stack(outputs, dim=2)
    return out, state


class RWKV6TimeMix(nn.Module):
    def __init__(
        self,
        layer_id: int,
        num_layers: int,
        dim: int,
        head_size: int,
        time_mix_extra_dim: int,
        time_decay_extra_dim: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.head_size = head_size
        if dim % head_size == 0:
            self.n_head = dim // head_size
        else:
            self.n_head = 1
            self.head_size = dim
        ratio_0_to_1 = layer_id / max(num_layers - 1, 1)
        ratio_1_to_almost0 = 1.0 - layer_id / num_layers

        ddd = torch.arange(dim) / dim
        self.x_maa = nn.Parameter(1.0 - ddd.pow(ratio_1_to_almost0))
        self.w_maa = nn.Parameter(1.0 - ddd.pow(ratio_1_to_almost0))
        self.k_maa = nn.Parameter(1.0 - ddd.pow(ratio_1_to_almost0))
        self.v_maa = nn.Parameter(1.0 - (ddd.pow(ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
        self.r_maa = nn.Parameter(1.0 - ddd.pow(0.5 * ratio_1_to_almost0))
        self.g_maa = nn.Parameter(1.0 - ddd.pow(0.5 * ratio_1_to_almost0))

        self.tm_w1 = nn.Parameter(torch.empty(dim, time_mix_extra_dim * 5))
        self.tm_w2 = nn.Parameter(torch.empty(5, time_mix_extra_dim, dim))
        nn.init.zeros_(self.tm_w1)
        nn.init.zeros_(self.tm_w2)

        self.td_w1 = nn.Parameter(torch.empty(dim, time_decay_extra_dim))
        self.td_w2 = nn.Parameter(torch.empty(time_decay_extra_dim, dim))
        nn.init.zeros_(self.td_w1)
        nn.init.zeros_(self.td_w2)

        decay_speed = [
            -6.0 + 5.0 * (i / max(dim - 1, 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            for i in range(dim)
        ]
        decay_speed = torch.tensor(decay_speed, dtype=torch.float32).reshape(self.n_head, self.head_size)
        self.time_decay = nn.Parameter(decay_speed)
        first_speed = [
            ratio_0_to_1 * (1.0 - (i / max(dim - 1, 1))) + ((i + 1) % 3 - 1) * 0.1
            for i in range(dim)
        ]
        first_speed = torch.tensor(first_speed, dtype=torch.float32).reshape(self.n_head, self.head_size)
        self.time_first = nn.Parameter(first_speed)

        self.receptance = nn.Linear(dim, dim, bias=bias)
        self.key = nn.Linear(dim, dim, bias=bias)
        self.value = nn.Linear(dim, dim, bias=bias)
        self.output = nn.Linear(dim, dim, bias=bias)
        self.gate = nn.Linear(dim, dim, bias=bias)
        self.ln_x = nn.GroupNorm(self.n_head, dim, eps=1e-5)

    def forward(
        self,
        x: torch.Tensor,
        prev: torch.Tensor | None,
        state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, T, D]
        shifted = _rwkv6_time_shift(x, prev)
        sx = shifted - x

        x_maa = self.x_maa.view(1, 1, -1)
        xxx = x + sx * x_maa
        mix = torch.tanh(xxx @ self.tm_w1)
        mix = mix.reshape(-1, 5, mix.shape[-1] // 5).transpose(0, 1).contiguous()
        mix = torch.bmm(mix, self.tm_w2).view(5, x.shape[0], x.shape[1], self.dim)
        mw, mk, mv, mr, mg = mix

        w_maa = self.w_maa.view(1, 1, -1)
        k_maa = self.k_maa.view(1, 1, -1)
        v_maa = self.v_maa.view(1, 1, -1)
        r_maa = self.r_maa.view(1, 1, -1)
        g_maa = self.g_maa.view(1, 1, -1)

        wx = x + sx * (w_maa + mw)
        kx = x + sx * (k_maa + mk)
        vx = x + sx * (v_maa + mv)
        rx = x + sx * (r_maa + mr)
        gx = x + sx * (g_maa + mg)

        r = torch.sigmoid(self.receptance(rx))
        k = self.key(kx)
        v = self.value(vx)
        g = F.silu(self.gate(gx))

        w = torch.tanh(wx @ self.td_w1) @ self.td_w2
        w = w.view(x.shape[0], x.shape[1], self.n_head, self.head_size).transpose(1, 2)
        w = self.time_decay.view(1, self.n_head, 1, self.head_size) + w
        w = torch.exp(-torch.exp(w))

        r = r.view(x.shape[0], x.shape[1], self.n_head, self.head_size).transpose(1, 2)
        k = k.view(x.shape[0], x.shape[1], self.n_head, self.head_size).transpose(1, 2)
        v = v.view(x.shape[0], x.shape[1], self.n_head, self.head_size).transpose(1, 2)

        u = self.time_first.view(1, self.n_head, self.head_size, 1).to(r.dtype)
        out, state = _rwkv6_wkv_op(r, k, v, w.to(r.dtype), u, state)
        out = out.transpose(1, 2).contiguous().view(x.shape[0] * x.shape[1], self.dim)
        out = self.ln_x(out).view(x.shape[0], x.shape[1], self.dim)
        out = self.output(out * g)
        new_prev = x[:, -1]
        return out, new_prev, state


class RWKV6ChannelMix(nn.Module):
    def __init__(
        self,
        layer_id: int,
        num_layers: int,
        dim: int,
        ffn_expand: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        ratio_1_to_almost0 = 1.0 - layer_id / num_layers
        ddd = torch.arange(dim) / dim
        self.k_maa = nn.Parameter(1.0 - ddd.pow(ratio_1_to_almost0))
        self.r_maa = nn.Parameter(1.0 - ddd.pow(ratio_1_to_almost0))
        self.key = nn.Linear(dim, dim * ffn_expand, bias=bias)
        self.value = nn.Linear(dim * ffn_expand, dim, bias=bias)
        self.receptance = nn.Linear(dim, dim, bias=bias)

    def forward(self, x: torch.Tensor, prev: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        shifted = _rwkv6_time_shift(x, prev)
        sx = shifted - x
        kx = x + sx * self.k_maa.view(1, 1, -1)
        rx = x + sx * self.r_maa.view(1, 1, -1)
        k = self.key(kx)
        k = F.relu(k).pow(2)
        v = self.value(k)
        r = torch.sigmoid(self.receptance(rx))
        out = r * v
        new_prev = x[:, -1]
        return out, new_prev


class RWKV6Block(nn.Module):
    def __init__(
        self,
        layer_id: int,
        num_layers: int,
        dim: int,
        head_size: int,
        time_mix_extra_dim: int,
        time_decay_extra_dim: int,
        ffn_expand: int,
        bias: bool = False,
        ln_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.ln1 = _RWKV6LayerNorm(dim, bias=bias, eps=ln_eps)
        self.ln2 = _RWKV6LayerNorm(dim, bias=bias, eps=ln_eps)
        self.time_mix = RWKV6TimeMix(
            layer_id,
            num_layers,
            dim,
            head_size,
            time_mix_extra_dim,
            time_decay_extra_dim,
            bias=bias,
        )
        self.channel_mix = RWKV6ChannelMix(
            layer_id,
            num_layers,
            dim,
            ffn_expand,
            bias=bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        prev_t, prev_c, kv_state = state
        y, prev_t, kv_state = self.time_mix(self.ln1(x), prev_t, kv_state)
        x = x + y
        y, prev_c = self.channel_mix(self.ln2(x), prev_c)
        x = x + y
        return x, (prev_t, prev_c, kv_state)


class RWKV6Model(nn.Module):
    def __init__(
        self,
        num_layers: int,
        dim: int,
        head_size: int = 64,
        time_mix_extra_dim: int = 32,
        time_decay_extra_dim: int = 64,
        ffn_expand: int = 3,
        bias: bool = False,
        ln_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.dim = dim
        self.head_size = head_size if dim % head_size == 0 else dim
        self.n_head = dim // self.head_size
        self.blocks = nn.ModuleList(
            [
                RWKV6Block(
                    layer_id=i,
                    num_layers=num_layers,
                    dim=dim,
                    head_size=self.head_size,
                    time_mix_extra_dim=time_mix_extra_dim,
                    time_decay_extra_dim=time_decay_extra_dim,
                    ffn_expand=ffn_expand,
                    bias=bias,
                    ln_eps=ln_eps,
                )
                for i in range(num_layers)
            ]
        )
        self.ln_out = _RWKV6LayerNorm(dim, bias=bias, eps=ln_eps)

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
        prev_t = torch.zeros(self.num_layers, batch_size, self.dim, device=device, dtype=dtype)
        prev_c = torch.zeros(self.num_layers, batch_size, self.dim, device=device, dtype=dtype)
        kv_state = torch.zeros(
            self.num_layers,
            batch_size,
            self.n_head,
            self.head_size,
            self.head_size,
            device=device,
            dtype=dtype,
        )
        return (prev_t, prev_c, kv_state)

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        return_state: bool = True,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None]:
        if state is None:
            state = self.init_state(x.shape[0], x.device, x.dtype)
        prev_t, prev_c, kv_state = state
        if return_state:
            next_prev_t = []
            next_prev_c = []
            next_kv = []
        else:
            next_prev_t = None
            next_prev_c = None
            next_kv = None
        for i, block in enumerate(self.blocks):
            x, layer_state = block(
                x,
                (prev_t[i], prev_c[i], kv_state[i]),
            )
            if return_state:
                next_prev_t.append(layer_state[0])
                next_prev_c.append(layer_state[1])
                next_kv.append(layer_state[2])
        x = self.ln_out(x)
        if return_state:
            next_state = (
                torch.stack(next_prev_t, dim=0),
                torch.stack(next_prev_c, dim=0),
                torch.stack(next_kv, dim=0),
            )
        else:
            next_state = None
        return x, next_state


class ExtendedMemory(nn.Module):
    """Extended memory module for RSL-RL.

    Supports: rnn, gru, lstm, mamba, xlstm, rwkv.
    Hidden state tensors are stored as (num_layers, batch, ...), with extra dims allowed.
    """

    def __init__(
        self,
        input_size: int,
        hidden_dim: int | list[int] | tuple[int, ...] = 256,
        num_layers: int = 1,
        type: str = "lstm",
        transformer_nhead: int | None = None,
        transformer_context_len: int = 32,
        transformer_dropout: float = 0.0,
        rwkv_head_size: int = 32,
        rwkv_time_mix_extra_dim: int = 32,
        rwkv_time_decay_extra_dim: int = 64,
        rwkv_ffn_expand: int = 3,
        rwkv_bias: bool = False,
        rwkv_ln_eps: float = 1e-5,
        rwkv_chunk_size: int = 512,
        xlstm_head_num: int | None = None,
        xlstm_head_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_state = None
        self.rnn_type = type.lower()
        self.layer_dims = _normalize_layer_dims(hidden_dim, num_layers)
        self.hidden_dim = self.layer_dims[-1]
        self.num_layers = num_layers
        self.max_dim = max(self.layer_dims)
        self.uniform_dim = all(d == self.layer_dims[0] for d in self.layer_dims)
        self.input_size = input_size
        self.transformer_nhead = transformer_nhead
        self.transformer_context_len = transformer_context_len
        self.transformer_dropout = transformer_dropout
        self.rwkv_head_size = rwkv_head_size
        self.rwkv_time_mix_extra_dim = rwkv_time_mix_extra_dim
        self.rwkv_time_decay_extra_dim = rwkv_time_decay_extra_dim
        self.rwkv_ffn_expand = rwkv_ffn_expand
        self.rwkv_bias = rwkv_bias
        self.rwkv_ln_eps = rwkv_ln_eps
        self.rwkv_chunk_size = rwkv_chunk_size
        self.xlstm_head_num = xlstm_head_num
        self.xlstm_head_dim = xlstm_head_dim

        self.input_proj = None
        if input_size != self.hidden_dim:
            self.input_proj = nn.Linear(input_size, self.hidden_dim, bias=False)

        if self.rnn_type in {"lstm", "gru", "rnn"}:
            rnn_cls = nn.LSTM
            rnn_kwargs = {}
            if self.rnn_type == "gru":
                rnn_cls = nn.GRU
            elif self.rnn_type == "rnn":
                rnn_cls = nn.RNN
                rnn_kwargs["nonlinearity"] = "tanh"
            if self.uniform_dim:
                self.rnn = rnn_cls(
                    input_size=input_size,
                    hidden_size=self.hidden_dim,
                    num_layers=num_layers,
                    **rnn_kwargs,
                )
                self._mode = "torch_rnn"
            else:
                self._mode = "stacked_rnn"
                self._init_stacked_rnn()
        elif self.rnn_type == "mamba":
            self._mode = "mamba"
            self._init_mamba()
        elif self.rnn_type == "xlstm":
            self._mode = "xlstm"
            self._init_xlstm()
        elif self.rnn_type in {"rwkv", "rwkv_v6"}:
            self._mode = "rwkv"
            self._init_rwkv()
        elif self.rnn_type == "transformer":
            self._mode = "transformer"
            self._init_transformer()
        else:
            raise ValueError(f"Unsupported recurrent type: {type}")

    def _init_mamba(self) -> None:
        if not self.uniform_dim:
            raise ValueError("mamba requires uniform rnn_hidden_dim across layers.")
        try:
            from mamba_ssm import Mamba
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "mamba-ssm is required for rnn_type='mamba'. Install with: pip install mamba-ssm[causal-conv1d]"
            ) from exc

        d_state = 16
        d_conv = 4
        expand = 2
        layers = []
        for i in range(self.num_layers):
            try:
                layer = Mamba(
                    d_model=self.hidden_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    layer_idx=i,
                )
            except TypeError:
                layer = Mamba(d_model=self.hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand)
            layers.append(layer)
        self.rnn_layers = nn.ModuleList(layers)

    def _init_xlstm(self) -> None:
        if not self.uniform_dim:
            raise ValueError("xlstm requires uniform rnn_hidden_dim across layers.")
        try:
            from xlstm.lstm import sLSTM, mLSTM
        except Exception:
            # Avoid importing xlstm.__init__ (which pulls lightning) by loading the module directly.
            import importlib.util
            import os
            import sys

            module_path = None
            for path in sys.path:
                candidate = os.path.join(path, "xlstm", "lstm.py")
                if os.path.isfile(candidate):
                    module_path = candidate
                    break
            if module_path is None:
                raise ImportError(
                    "xlstm is required for rnn_type='xlstm'. Install with: pip install git+https://github.com/myscience/x-lstm"
                )
            import types

            pkg_name = "xlstm"
            if pkg_name not in sys.modules:
                pkg = types.ModuleType(pkg_name)
                pkg.__path__ = [os.path.dirname(module_path)]
                sys.modules[pkg_name] = pkg
            spec = importlib.util.spec_from_file_location("xlstm.lstm", module_path)
            if spec is None or spec.loader is None:
                raise ImportError(
                    "Failed to load xlstm.lstm module directly. Check xlstm installation."
                )
            xlstm_lstm = importlib.util.module_from_spec(spec)
            sys.modules["xlstm.lstm"] = xlstm_lstm
            spec.loader.exec_module(xlstm_lstm)
            sLSTM = getattr(xlstm_lstm, "sLSTM", None)
            mLSTM = getattr(xlstm_lstm, "mLSTM", None)
            if sLSTM is None or mLSTM is None:
                raise ImportError("xlstm.lstm does not define sLSTM/mLSTM.")

        if self.xlstm_head_num is not None:
            head_num = int(self.xlstm_head_num)
        else:
            head_num = 4 if self.hidden_dim % 4 == 0 else 1
        if self.xlstm_head_dim is not None:
            head_dim = int(self.xlstm_head_dim)
        else:
            head_dim = self.hidden_dim // head_num
        if head_num <= 0 or head_dim <= 0:
            raise ValueError("xlstm head_num/head_dim must be positive.")
        if head_num * head_dim != self.hidden_dim:
            raise ValueError(
                f"xlstm head_num*head_dim must equal rnn_hidden_dim ({head_num}*{head_dim} != {self.hidden_dim})."
            )
        self._xlstm_head_num = head_num
        self._xlstm_head_dim = head_dim
        layer_types = []
        layers = []
        for i in range(self.num_layers):
            if i % 2 == 0:
                layers.append(sLSTM(self.hidden_dim, head_dim, head_num))
                layer_types.append("s")
            else:
                layers.append(mLSTM(self.hidden_dim, head_num, head_dim))
                layer_types.append("m")
        self._xlstm_layer_types = layer_types
        self.rnn_layers = nn.ModuleList(layers)
        self._xlstm_state_specs = self._xlstm_build_state_specs()

    def _init_rwkv(self) -> None:
        if not self.uniform_dim:
            raise ValueError("rwkv requires uniform rnn_hidden_dim across layers.")
        self.rnn = RWKV6Model(
            num_layers=self.num_layers,
            dim=self.hidden_dim,
            head_size=self.rwkv_head_size,
            time_mix_extra_dim=self.rwkv_time_mix_extra_dim,
            time_decay_extra_dim=self.rwkv_time_decay_extra_dim,
            ffn_expand=self.rwkv_ffn_expand,
            bias=self.rwkv_bias,
            ln_eps=self.rwkv_ln_eps,
        )

    def _init_transformer(self) -> None:
        if not self.uniform_dim:
            raise ValueError("transformer requires uniform rnn_hidden_dim across layers.")
        if self.transformer_nhead is not None:
            nhead = self.transformer_nhead
        else:
            nhead = 4 if self.hidden_dim % 4 == 0 else 1
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=nhead,
            batch_first=True,
            activation="gelu",
            dropout=self.transformer_dropout,
        )
        self.rnn = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self._transformer_context = int(self.transformer_context_len)
        self._transformer_buffer = None

    def forward(
        self,
        input: torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_state=None,
    ) -> torch.Tensor:
        batch_mode = masks is not None

        if self._mode == "torch_rnn":
            if batch_mode:
                if hidden_state is None:
                    raise ValueError("Hidden states not passed to memory module during policy update")
                out, _ = self.rnn(input, hidden_state)
                out = unpad_trajectories(out, masks)
            else:
                out, self.hidden_state = self.rnn(input.unsqueeze(0), self.hidden_state)
            return out
        if self._mode == "stacked_rnn":
            return self._forward_stacked_rnn(input, masks, hidden_state)

        if self._mode == "mamba":
            return self._forward_mamba(input, masks, hidden_state)
        if self._mode == "xlstm":
            return self._forward_xlstm(input, masks, hidden_state)
        if self._mode == "rwkv":
            return self._forward_rwkv(input, masks, hidden_state)
        if self._mode == "transformer":
            return self._forward_transformer(input, masks, hidden_state)

        raise RuntimeError(f"Unknown mode: {self._mode}")

    def reset(self, dones: torch.Tensor | None = None, hidden_state=None) -> None:
        if dones is None:
            self.hidden_state = hidden_state
            return
        if self.hidden_state is None:
            return
        if isinstance(self.hidden_state, tuple):
            for h in self.hidden_state:
                h[:, dones == 1, ...] = 0.0
        else:
            self.hidden_state[:, dones == 1, ...] = 0.0
        if self._mode == "transformer" and self._transformer_buffer is not None:
            self._transformer_buffer[dones == 1, :, :] = 0.0

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        if self.hidden_state is None:
            return
        if isinstance(self.hidden_state, tuple):
            if dones is None:
                self.hidden_state = tuple(h.detach() for h in self.hidden_state)
            else:
                for h in self.hidden_state:
                    h[:, dones == 1, ...] = h[:, dones == 1, ...].detach()
        else:
            if dones is None:
                self.hidden_state = self.hidden_state.detach()
            else:
                self.hidden_state[:, dones == 1, ...] = self.hidden_state[:, dones == 1, ...].detach()

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_proj is None:
            return x
        return self.input_proj(x)

    def _init_stacked_rnn(self) -> None:
        cells = []
        for i in range(self.num_layers):
            in_dim = self.input_size if i == 0 else self.layer_dims[i - 1]
            out_dim = self.layer_dims[i]
            if self.rnn_type == "lstm":
                cell = nn.LSTMCell(in_dim, out_dim)
            elif self.rnn_type == "gru":
                cell = nn.GRUCell(in_dim, out_dim)
            else:
                cell = nn.RNNCell(in_dim, out_dim, nonlinearity="tanh")
            cells.append(cell)
        self.rnn_cells = nn.ModuleList(cells)

    def _stacked_init_hidden(self, batch_size: int, device: torch.device):
        h_list = [torch.zeros(batch_size, dim, device=device) for dim in self.layer_dims]
        if self.rnn_type == "lstm":
            c_list = [torch.zeros(batch_size, dim, device=device) for dim in self.layer_dims]
            return self._stacked_pack_hidden(h_list, c_list)
        return self._stacked_pack_hidden(h_list, None)

    def _stacked_unpack_hidden(self, hidden_state):
        if hidden_state is None:
            return None
        if self.rnn_type == "lstm":
            h, c = hidden_state
            h_list = [h[i, :, :dim] for i, dim in enumerate(self.layer_dims)]
            c_list = [c[i, :, :dim] for i, dim in enumerate(self.layer_dims)]
            return h_list, c_list
        h = hidden_state
        h_list = [h[i, :, :dim] for i, dim in enumerate(self.layer_dims)]
        return h_list, None

    def _stacked_pack_hidden(self, h_list, c_list):
        batch_size = h_list[0].shape[0]
        h = h_list[0].new_zeros((self.num_layers, batch_size, self.max_dim))
        for i, dim in enumerate(self.layer_dims):
            h[i, :, :dim] = h_list[i]
        if self.rnn_type == "lstm":
            c = c_list[0].new_zeros((self.num_layers, batch_size, self.max_dim))
            for i, dim in enumerate(self.layer_dims):
                c[i, :, :dim] = c_list[i]
            return (h, c)
        return h

    def _forward_stacked_rnn(self, input: torch.Tensor, masks: torch.Tensor | None, hidden_state):
        batch_mode = masks is not None
        if batch_mode:
            if hidden_state is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            h_list, c_list = self._stacked_unpack_hidden(hidden_state)
            outputs = []
            for t in range(input.shape[0]):
                x_t = input[t]
                for layer_idx, cell in enumerate(self.rnn_cells):
                    if self.rnn_type == "lstm":
                        h_t, c_t = cell(x_t, (h_list[layer_idx], c_list[layer_idx]))
                        h_list[layer_idx] = h_t
                        c_list[layer_idx] = c_t
                        x_t = h_t
                    else:
                        h_t = cell(x_t, h_list[layer_idx])
                        h_list[layer_idx] = h_t
                        x_t = h_t
                outputs.append(x_t)
            out = torch.stack(outputs, dim=0)
            out = unpad_trajectories(out, masks)
            return out

        if self.hidden_state is None:
            self.hidden_state = self._stacked_init_hidden(input.shape[0], input.device)
        h_list, c_list = self._stacked_unpack_hidden(self.hidden_state)
        x_t = input
        for layer_idx, cell in enumerate(self.rnn_cells):
            if self.rnn_type == "lstm":
                h_t, c_t = cell(x_t, (h_list[layer_idx], c_list[layer_idx]))
                h_list[layer_idx] = h_t
                c_list[layer_idx] = c_t
                x_t = h_t
            else:
                h_t = cell(x_t, h_list[layer_idx])
                h_list[layer_idx] = h_t
                x_t = h_t
        self.hidden_state = self._stacked_pack_hidden(h_list, c_list)
        return x_t.unsqueeze(0)

    def _forward_mamba(self, input: torch.Tensor, masks: torch.Tensor | None, hidden_state):
        batch_mode = masks is not None
        if batch_mode:
            if hidden_state is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            conv_state, ssm_state = hidden_state
            x = self._project(input)
            outputs = []
            for t in range(x.shape[0]):
                x_t = x[t]
                for layer_idx, layer in enumerate(self.rnn_layers):
                    if hasattr(layer, "step"):
                        y, conv_state[layer_idx], ssm_state[layer_idx] = layer.step(
                            x_t.unsqueeze(1), conv_state[layer_idx], ssm_state[layer_idx]
                        )
                        x_t = y.squeeze(1)
                    else:
                        x_t = layer(x_t.unsqueeze(1)).squeeze(1)
                outputs.append(x_t)
            out = torch.stack(outputs, dim=0)
            out = unpad_trajectories(out, masks)
            return out

        x = self._project(input)
        if self.hidden_state is None:
            conv_state, ssm_state = self._mamba_init_state(x.shape[0], x.device)
        else:
            conv_state, ssm_state = self.hidden_state
        x_t = x
        for layer_idx, layer in enumerate(self.rnn_layers):
            if hasattr(layer, "step"):
                y, conv_state[layer_idx], ssm_state[layer_idx] = layer.step(
                    x_t.unsqueeze(1), conv_state[layer_idx], ssm_state[layer_idx]
                )
                x_t = y.squeeze(1)
            else:
                x_t = layer(x_t.unsqueeze(1)).squeeze(1)
        self.hidden_state = (conv_state, ssm_state)
        return x_t.unsqueeze(0)

    def _mamba_init_state(self, batch_size: int, device: torch.device):
        conv_states = []
        ssm_states = []
        for layer in self.rnn_layers:
            if hasattr(layer, "allocate_inference_cache"):
                conv_state, ssm_state = layer.allocate_inference_cache(batch_size, max_seqlen=1)
            else:
                conv_state = torch.zeros(batch_size, self.hidden_dim, 1, device=device)
                ssm_state = torch.zeros(batch_size, self.hidden_dim, 1, device=device)
            conv_states.append(conv_state.to(device))
            ssm_states.append(ssm_state.to(device))
        conv_state = torch.stack(conv_states, dim=0)
        ssm_state = torch.stack(ssm_states, dim=0)
        return conv_state, ssm_state

    def _forward_xlstm(self, input: torch.Tensor, masks: torch.Tensor | None, hidden_state):
        batch_mode = masks is not None
        if batch_mode:
            if hidden_state is None:
                hidden_state = self._xlstm_init_state(input.shape[1], input.device)
            layer_states = self._xlstm_unpack_state(hidden_state)
            if layer_states is None:
                layer_states = self._xlstm_unpack_state(self._xlstm_init_state(input.shape[1], input.device))
            x = self._project(input)
            outputs = []
            for t in range(x.shape[0]):
                x_t = x[t]
                for layer_idx, layer in enumerate(self.rnn_layers):
                    x_t, layer_states[layer_idx] = layer(x_t, layer_states[layer_idx])
                outputs.append(x_t)
            out = torch.stack(outputs, dim=0)
            out = unpad_trajectories(out, masks)
            return out

        x = self._project(input)
        if self.hidden_state is None:
            hidden_state = self._xlstm_init_state(x.shape[0], x.device)
        layer_states = self._xlstm_unpack_state(hidden_state)
        if layer_states is None:
            layer_states = self._xlstm_unpack_state(self._xlstm_init_state(x.shape[0], x.device))
        x_t = x
        for layer_idx, layer in enumerate(self.rnn_layers):
            x_t, layer_states[layer_idx] = layer(x_t, layer_states[layer_idx])
        self.hidden_state = self._xlstm_pack_state(layer_states)
        return x_t.unsqueeze(0)

    def _xlstm_init_state(self, batch_size: int, device: torch.device):
        layer_states = []
        for layer in self.rnn_layers:
            state = layer.init_hidden(batch_size)
            if not isinstance(state, tuple):
                state = (state,)
            layer_states.append(tuple(s.to(device) for s in state))
        return self._xlstm_pack_state(layer_states)

    def _xlstm_build_state_specs(self):
        specs = []
        max_len = 0
        for layer_type in self._xlstm_layer_types:
            if layer_type == "s":
                shapes = [
                    (self._xlstm_head_num * self._xlstm_head_dim,),
                    (self._xlstm_head_num * self._xlstm_head_dim,),
                    (self._xlstm_head_num * self._xlstm_head_dim,),
                    (self._xlstm_head_num * self._xlstm_head_dim,),
                ]
            else:
                shapes = [
                    (self._xlstm_head_num, self._xlstm_head_dim, self._xlstm_head_dim),
                    (self._xlstm_head_num, self._xlstm_head_dim),
                    (self._xlstm_head_num,),
                ]
            lengths = [int(torch.tensor(shape).prod().item()) for shape in shapes]
            total = sum(lengths)
            max_len = max(max_len, total)
            specs.append({"shapes": shapes, "lengths": lengths, "total": total})
        return {"specs": specs, "max_len": max_len}

    def _xlstm_unpack_state(self, state_tuple):
        if state_tuple is None:
            return None
        if isinstance(state_tuple, tuple):
            # Backward-compat: old stacked tuple format (uniform state components).
            num_components = len(state_tuple)
            layer_states = []
            for layer_idx in range(self.num_layers):
                comp = tuple(state_tuple[c][layer_idx] for c in range(num_components))
                if len(comp) == 1:
                    layer_states.append(comp[0])
                else:
                    layer_states.append(comp)
            return layer_states
        # New packed tensor format: [num_layers, batch, flat_dim]
        specs = self._xlstm_state_specs["specs"]
        layer_states = []
        for layer_idx in range(self.num_layers):
            flat = state_tuple[layer_idx]
            offset = 0
            tensors = []
            for shape, length in zip(specs[layer_idx]["shapes"], specs[layer_idx]["lengths"]):
                chunk = flat[:, offset : offset + length]
                tensors.append(chunk.view(chunk.shape[0], *shape))
                offset += length
            layer_states.append(tuple(tensors))
        return layer_states

    def _xlstm_pack_state(self, layer_states: list):
        if not layer_states:
            return None
        specs = self._xlstm_state_specs["specs"]
        max_len = self._xlstm_state_specs["max_len"]
        batch_size = layer_states[0][0].shape[0]
        packed = layer_states[0][0].new_zeros((self.num_layers, batch_size, max_len))
        for layer_idx, state in enumerate(layer_states):
            flat_chunks = []
            for tensor in state:
                flat_chunks.append(tensor.reshape(batch_size, -1))
            flat = torch.cat(flat_chunks, dim=1)
            total = specs[layer_idx]["total"]
            packed[layer_idx, :, :total] = flat
        return packed

    def _forward_rwkv(self, input: torch.Tensor, masks: torch.Tensor | None, hidden_state):
        batch_mode = masks is not None
        if not batch_mode and input.ndim > 2:
            feature_dim = input.ndim - 1
            candidate_dims = [d for d in range(input.ndim) if d != feature_dim]
            batch_dim = max(candidate_dims, key=lambda d: input.shape[d])
            time_dims = [d for d in range(input.ndim) if d not in (batch_dim, feature_dim)]
            index = [slice(None)] * input.ndim
            for dim in time_dims:
                index[dim] = -1
            reduced = input[tuple(index)]
            if reduced.ndim > 2:
                reduced = reduced.squeeze()
            input = reduced

        x = self._project(input)
        if batch_mode:
            if hidden_state is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            x_seq = x.transpose(0, 1)
            batch_size = x_seq.shape[0]
            chunk_size = int(self.rwkv_chunk_size)
            if chunk_size <= 0 or chunk_size >= batch_size:
                out, _ = self.rnn(x_seq, state=None, return_state=False)
                out = out.transpose(0, 1)
            else:
                outputs = []
                for start in range(0, batch_size, chunk_size):
                    end = min(start + chunk_size, batch_size)
                    out_chunk, _ = self.rnn(x_seq[start:end], state=None, return_state=False)
                    outputs.append(out_chunk)
                out = torch.cat(outputs, dim=0).transpose(0, 1)
            out = unpad_trajectories(out, masks)
            return out

        if self.hidden_state is None:
            state = None
        else:
            state = self.hidden_state
        x_seq = x.unsqueeze(1)
        out, state = self.rnn(x_seq, state=state)
        self.hidden_state = state
        return out[:, -1, :].unsqueeze(0)

    def _forward_transformer(self, input: torch.Tensor, masks: torch.Tensor | None, hidden_state):
        batch_mode = masks is not None
        x = self._project(input)
        if batch_mode:
            # x: [T, B, D] -> [B, T, D]
            x_seq = x.transpose(0, 1)
            seq_len = x_seq.shape[1]
            attn_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x_seq.device, dtype=torch.bool), diagonal=1
            )
            out = self.rnn(x_seq, mask=attn_mask).transpose(0, 1)
            out = unpad_trajectories(out, masks)
            return out

        # inference / rollout collection: cache last context_len tokens
        x_step = x.unsqueeze(1)  # [B, 1, D]
        if self._transformer_buffer is None:
            self._transformer_buffer = x_step
        else:
            self._transformer_buffer = torch.cat([self._transformer_buffer, x_step], dim=1)
            if self._transformer_buffer.shape[1] > self._transformer_context:
                self._transformer_buffer = self._transformer_buffer[:, -self._transformer_context :, :]

        seq_len = self._transformer_buffer.shape[1]
        attn_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1
        )
        out = self.rnn(self._transformer_buffer, mask=attn_mask)
        last = out[:, -1, :]  # [B, D]
        # store a compact hidden state for logging
        self.hidden_state = last.unsqueeze(0).repeat(self.num_layers, 1, 1).contiguous()
        return last.unsqueeze(0)


class ExtendedActorCriticRecurrent(nn.Module):
    is_recurrent: bool = True

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = (256, 256, 256),
        critic_hidden_dims: tuple[int] | list[int] = (256, 256, 256),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        rnn_type: str = "lstm",
        rnn_hidden_dim: int | list[int] | tuple[int, ...] = 256,
        rnn_num_layers: int = 1,
        transformer_nhead: int | None = None,
        transformer_context_len: int = 32,
        transformer_dropout: float = 0.0,
        rwkv_head_size: int = 32,
        rwkv_time_mix_extra_dim: int = 32,
        rwkv_time_decay_extra_dim: int = 64,
        rwkv_ffn_expand: int = 3,
        rwkv_bias: bool = False,
        rwkv_ln_eps: float = 1e-5,
        rwkv_chunk_size: int = 512,
        xlstm_head_num: int | None = None,
        xlstm_head_dim: int | None = None,
        **kwargs,
    ) -> None:
        if "rnn_hidden_size" in kwargs:
            if rnn_hidden_dim == 256:
                rnn_hidden_dim = kwargs.pop("rnn_hidden_size")
        if kwargs:
            print(
                "ActorCriticRecurrent.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys())
            )
        super().__init__()

        self.obs_groups = obs_groups
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCriticRecurrent module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCriticRecurrent module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]

        self.state_dependent_std = state_dependent_std
        rnn_out_dim = _last_hidden_dim(rnn_hidden_dim)

        from rsl_rl.networks import EmpiricalNormalization, MLP

        self.memory_a = ExtendedMemory(
            num_actor_obs,
            rnn_hidden_dim,
            rnn_num_layers,
            rnn_type,
            transformer_nhead=transformer_nhead,
            transformer_context_len=transformer_context_len,
            transformer_dropout=transformer_dropout,
            rwkv_head_size=rwkv_head_size,
            rwkv_time_mix_extra_dim=rwkv_time_mix_extra_dim,
            rwkv_time_decay_extra_dim=rwkv_time_decay_extra_dim,
            rwkv_ffn_expand=rwkv_ffn_expand,
            rwkv_bias=rwkv_bias,
            rwkv_ln_eps=rwkv_ln_eps,
            rwkv_chunk_size=rwkv_chunk_size,
            xlstm_head_num=xlstm_head_num,
            xlstm_head_dim=xlstm_head_dim,
        )
        if self.state_dependent_std:
            self.actor = MLP(rnn_out_dim, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(rnn_out_dim, num_actions, actor_hidden_dims, activation)
        print(f"Actor RNN: {self.memory_a}")
        print(f"Actor MLP: {self.actor}")

        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = nn.Identity()

        self.memory_c = ExtendedMemory(
            num_critic_obs,
            rnn_hidden_dim,
            rnn_num_layers,
            rnn_type,
            transformer_nhead=transformer_nhead,
            transformer_context_len=transformer_context_len,
            transformer_dropout=transformer_dropout,
            rwkv_head_size=rwkv_head_size,
            rwkv_time_mix_extra_dim=rwkv_time_mix_extra_dim,
            rwkv_time_decay_extra_dim=rwkv_time_decay_extra_dim,
            rwkv_ffn_expand=rwkv_ffn_expand,
            rwkv_bias=rwkv_bias,
            rwkv_ln_eps=rwkv_ln_eps,
            rwkv_chunk_size=rwkv_chunk_size,
            xlstm_head_num=xlstm_head_num,
            xlstm_head_dim=xlstm_head_dim,
        )
        self.critic = MLP(rnn_out_dim, 1, critic_hidden_dims, activation)
        print(f"Critic RNN: {self.memory_c}")
        print(f"Critic MLP: {self.critic}")

        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = nn.Identity()

        self.noise_std_type = noise_std_type
        if self.state_dependent_std:
            torch.nn.init.zeros_(self.actor[-2].weight[num_actions:])
            if self.noise_std_type == "scalar":
                torch.nn.init.constant_(self.actor[-2].bias[num_actions:], init_noise_std)
            elif self.noise_std_type == "log":
                torch.nn.init.constant_(
                    self.actor[-2].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                )
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        from torch.distributions import Normal

        Normal.set_default_validate_args(False)
        self.distribution = None

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def forward(self):
        raise NotImplementedError

    def _update_distribution(self, obs: torch.Tensor) -> None:
        if self.state_dependent_std:
            mean_and_std = self.actor(obs)
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(mean_and_std, dim=-2)
            elif self.noise_std_type == "log":
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                std = torch.exp(log_std)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            mean = self.actor(obs)
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        from torch.distributions import Normal

        self.distribution = Normal(mean, std)

    def act(self, obs, masks: torch.Tensor | None = None, hidden_state=None) -> torch.Tensor:
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        out_mem = self.memory_a(obs, masks, hidden_state).squeeze(0)
        self._update_distribution(out_mem)
        return self.distribution.sample()

    def act_inference(self, obs) -> torch.Tensor:
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        out_mem = self.memory_a(obs).squeeze(0)
        if self.state_dependent_std:
            return self.actor(out_mem)[..., 0, :]
        return self.actor(out_mem)

    def evaluate(self, obs, masks: torch.Tensor | None = None, hidden_state=None) -> torch.Tensor:
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        out_mem = self.memory_c(obs, masks, hidden_state).squeeze(0)
        return self.critic(out_mem)

    def get_actor_obs(self, obs) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["policy"]]
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["critic"]]
        return torch.cat(obs_list, dim=-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_hidden_states(self):
        return self.memory_a.hidden_state, self.memory_c.hidden_state

    def update_normalization(self, obs) -> None:
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        super().load_state_dict(state_dict, strict=strict)
        return True


def _gather_hidden_states(
    saved_hidden_states: Iterable[torch.Tensor],
    last_was_done: torch.Tensor,
    first_traj: int,
    last_traj: int,
) -> list[torch.Tensor]:
    batch_states: list[torch.Tensor] = []
    for saved_hidden_state in saved_hidden_states:
        permute_order = (2, 0, 1) + tuple(range(3, saved_hidden_state.ndim))
        permuted = saved_hidden_state.permute(*permute_order)
        selected = permuted[last_was_done][first_traj:last_traj]
        batch_states.append(selected.transpose(1, 0).contiguous())
    return batch_states


def _recurrent_mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8):
    if self.training_type != "rl":
        raise ValueError("This function is only available for reinforcement learning training.")
    from rsl_rl.storage.rollout_storage import split_and_pad_trajectories

    padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
    mini_batch_size = self.num_envs // num_mini_batches
    for _ in range(num_epochs):
        first_traj = 0
        for i in range(num_mini_batches):
            start = i * mini_batch_size
            stop = (i + 1) * mini_batch_size

            dones = self.dones.squeeze(-1)
            last_was_done = torch.zeros_like(dones, dtype=torch.bool)
            last_was_done[1:] = dones[:-1]
            last_was_done[0] = True
            trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
            last_traj = first_traj + trajectories_batch_size

            masks_batch = trajectory_masks[:, first_traj:last_traj]
            obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
            actions_batch = self.actions[:, start:stop]
            old_mu_batch = self.mu[:, start:stop]
            old_sigma_batch = self.sigma[:, start:stop]
            returns_batch = self.returns[:, start:stop]
            advantages_batch = self.advantages[:, start:stop]
            values_batch = self.values[:, start:stop]
            old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]

            last_was_done = last_was_done.permute(1, 0)
            hidden_state_a_batch = _gather_hidden_states(
                self.saved_hidden_state_a, last_was_done, first_traj, last_traj
            )
            hidden_state_c_batch = _gather_hidden_states(
                self.saved_hidden_state_c, last_was_done, first_traj, last_traj
            )
            hidden_state_a_batch = hidden_state_a_batch[0] if len(hidden_state_a_batch) == 1 else hidden_state_a_batch
            hidden_state_c_batch = hidden_state_c_batch[0] if len(hidden_state_c_batch) == 1 else hidden_state_c_batch

            yield (
                obs_batch,
                actions_batch,
                values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                (hidden_state_a_batch, hidden_state_c_batch),
                masks_batch,
            )

            first_traj = last_traj


def register_rsl_rl_extensions() -> None:
    """Patch rsl_rl to support additional recurrent types with extended state shapes."""
    import rsl_rl.modules.actor_critic_recurrent as acr
    import rsl_rl.networks.memory as memory_mod
    import rsl_rl.storage.rollout_storage as rollout_storage
    import rsl_rl.modules as modules
    import rsl_rl.runners.on_policy_runner as on_policy_runner

    if not getattr(memory_mod, "_robot_lab_extended", False):
        memory_mod.Memory = ExtendedMemory
        memory_mod._robot_lab_extended = True
        acr.Memory = ExtendedMemory

    if not getattr(modules, "_robot_lab_extended", False):
        modules.ActorCriticRecurrent = ExtendedActorCriticRecurrent
        on_policy_runner.ActorCriticRecurrent = ExtendedActorCriticRecurrent
        acr.ActorCriticRecurrent = ExtendedActorCriticRecurrent
        modules._robot_lab_extended = True

    if not getattr(rollout_storage, "_robot_lab_extended", False):
        rollout_storage.RolloutStorage.recurrent_mini_batch_generator = _recurrent_mini_batch_generator
        rollout_storage._robot_lab_extended = True
