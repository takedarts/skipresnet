from typing import Callable, List, Tuple, Any
from .modules import Reshape

import torch
import torch.jit
import torch.nn as nn
import torch.nn.functional as F
import collections
import math


@torch.jit.script
def weighted_sum(xs: List[torch.Tensor], ws: List[torch.Tensor]) -> torch.Tensor:
    z = xs[0] * 0
    for x, w in zip(xs, ws):
        z = z + x * w
    return z


class NoneJunction(nn.Module):
    def __init__(
        self,
        index: int,
        settings: List[Tuple[int, int, int]],
        **kwargs
    ) -> None:
        super().__init__()

    def forward(self, y: torch.Tensor, x: List[Any]) -> torch.Tensor:
        return y


class BasicJunction(nn.Module):
    def __init__(
        self,
        index: int,
        settings: List[Tuple[int, int, int]],
        **kwargs
    ) -> None:
        super().__init__()

    def forward(self, y: torch.Tensor, x: List[Any]) -> torch.Tensor:
        return y + x[-1]


class ConcatJunction(nn.Module):
    def __init__(
        self,
        index: int,
        settings: List[Tuple[int, int, int]],
        **kwargs
    ) -> None:
        super().__init__()

    def forward(self, y: torch.Tensor, x: List[Any]) -> torch.Tensor:
        return torch.cat([y, x[-1]], dim=1)


class GateJunction(nn.Module):
    def __init__(
        self,
        inbounds: List[int],
        index: int,
        settings: List[Tuple[int, int, int]],
        gate_normalization: Callable[[int], nn.Module],
        gate_activation: Callable[..., nn.Module],
        gate_reduction: int,
        save_gate_weights: bool = False,
        **kwargs
    ) -> None:
        super().__init__()
        self.index = index
        self.inbounds = inbounds
        self.channels = settings[index][1]
        self.paddings = [settings[index][1] - settings[i][1] for i in inbounds]

        # gate operation
        in_channels = settings[index][1] * (1 + len(self.inbounds))
        out_channels = settings[index][1]
        mid_channels = math.ceil(max(out_channels // gate_reduction, 1) / 8) * 8

        self.op = nn.Sequential(collections.OrderedDict(m for m in [
            ('pool', nn.AdaptiveAvgPool2d((1, 1))),
            ('conv1', nn.Conv2d(
                in_channels, mid_channels, kernel_size=1,
                padding=0, bias=False)),
            ('norm1', gate_normalization(mid_channels)),
            ('act1', gate_activation(inplace=True)),
            ('conv2', nn.Conv2d(
                mid_channels, (1 + len(self.inbounds)) * out_channels,
                kernel_size=1, padding=0, bias=True)),
            ('reshape', Reshape(1 + len(self.inbounds), out_channels)),
        ] if m[1] is not None))

        self.save_gate_weights = save_gate_weights
        self.gate_weights = None

    def forward(self, y: torch.Tensor, x: List[Any]) -> torch.Tensor:
        xs = [
            F.pad(x[i], (0, 0, 0, 0, 0, p)) if p != 0 else x[i]
            for i, p in zip(self.inbounds, self.paddings)]
        w = self.op(torch.cat([y] + xs, dim=1))
        w1, w2 = w.split([1, len(xs)], dim=1)
        w1 = w1.sigmoid()[:, 0, :, None, None]
        w2 = w2.softmax(dim=1)[:, :, :, None, None]

        if self.save_gate_weights:
            self.gate_weights = w2.detach().cpu().numpy()

        y = y * w1
        z = weighted_sum(xs, [w.squeeze(1) for w in w2.split(1, dim=1)])

        return y + z


class SkipJunction(GateJunction):
    def __init__(
        self,
        index: int,
        settings: List[Tuple[int, int, int]],
        gate_normalization: Callable[[int], nn.Module],
        gate_activation: Callable[..., nn.Module],
        skip_connections: int,
        **kwargs
    ) -> None:
        # index of the begining of the section
        for i in range(index, -1, -1):
            start_index = i
            if settings[i][2] != 1:
                break

        # indexes of skip connections
        inbounds = [index]
        for i in range(skip_connections):
            inbound_index = index - (2 ** (i + 1)) + 1
            if inbound_index < start_index:
                break
            inbounds.append(inbound_index)

        # init a gate junction
        super().__init__(
            inbounds, index, settings,
            gate_normalization, gate_activation, **kwargs)


class DenseJunction(GateJunction):
    def __init__(
        self,
        index: int,
        settings: List[Tuple[int, int, int]],
        gate_normalization: Callable[[int], nn.Module],
        gate_activation: Callable[..., nn.Module],
        dense_connections: int,
        **kwargs
    ) -> None:
        # index of the begining of the section
        for i in range(index, -1, -1):
            start_index = i
            if settings[i][2] != 1:
                break

        # indexes of skip connections
        inbounds = [index]
        for i in range(dense_connections):
            inbound_index = index - 1 - i
            if inbound_index < start_index:
                break
            inbounds.append(inbound_index)

        # init a gate junction
        super().__init__(
            inbounds, index, settings,
            gate_normalization, gate_activation, **kwargs)


class DynamicJunction(nn.Module):
    def __init__(
        self,
        index: int,
        settings: List[Tuple[int, int, int]],
        gate_normalization: Callable[[int], nn.Module],
        gate_activation: Callable[..., nn.Module],
        skip_connections: int,
        gate_reduction: int,
        save_gate_weights: bool = False,
        **kwargs
    ) -> None:
        super().__init__()
        # index of the begining of the section
        for i in range(index, -1, -1):
            start_index = i
            if settings[i][2] != 1:
                break

        # indexes of skip connections
        inbounds = [index]
        for i in range(skip_connections):
            inbound_index = index - (2 ** (i + 1)) + 1
            if inbound_index < start_index:
                break
            inbounds.append(inbound_index)

        # parameters
        self.index = index
        self.inbounds = inbounds
        self.channels = settings[index][1]
        self.paddings = [settings[index][1] - settings[i][1] for i in inbounds]

        # gate operation
        in_channels = settings[index][1] * (1 + len(self.inbounds))
        out_channels = settings[index][1]
        mid_channels = math.ceil(max(out_channels // gate_reduction, 1) / 8) * 8

        self.op = nn.Sequential(collections.OrderedDict(m for m in [
            ('pool', nn.AdaptiveAvgPool2d((1, 1))),
            ('conv1', nn.Conv2d(
                in_channels, mid_channels, kernel_size=1,
                padding=0, bias=False)),
            ('norm1', gate_normalization(mid_channels)),
            ('act1', gate_activation(inplace=True)),
            ('conv2', nn.Conv2d(
                mid_channels, (1 + len(self.inbounds)) * out_channels,
                kernel_size=1, padding=0, bias=True)),
            ('reshape', Reshape(1 + len(self.inbounds), out_channels)),
        ] if m[1] is not None))

        self.save_gate_weights = save_gate_weights
        self.gate_weights = None

    def forward(self, y: torch.Tensor, x: List[Any]) -> torch.Tensor:
        xs = [
            F.pad(x[i], (0, 0, 0, 0, 0, p)) if p != 0 else x[i]
            for i, p in zip(self.inbounds, self.paddings)]
        w = self.op(torch.cat([y] + xs, dim=1))
        _, w2 = w.split([1, len(xs)], dim=1)
        w2 = w2.softmax(dim=1)[:, :, :, None, None]

        if self.save_gate_weights:
            self.gate_weights = w2.detach().cpu().numpy()

        z = weighted_sum(xs, [w.squeeze(1) for w in w2.split(1, dim=1)])

        return y + z


class StaticJunction(nn.Module):
    def __init__(
        self,
        index: int,
        settings: List[Tuple[int, int, int]],
        skip_connections: int,
        save_gate_weights: bool = False,
        **kwargs
    ) -> None:
        super().__init__()
        # index of the begining of the section
        for i in range(index, -1, -1):
            start_index = i
            if settings[i][2] != 1:
                break

        # indexes of skip connections
        inbounds = [index]
        for i in range(skip_connections):
            inbound_index = index - (2 ** (i + 1)) + 1
            if inbound_index < start_index:
                break
            inbounds.append(inbound_index)

        # parameters
        self.index = index
        self.inbounds = inbounds
        self.channels = settings[index][1]
        self.paddings = [settings[index][1] - settings[i][1] for i in inbounds]

        # gate operation
        out_channels = settings[index][1]
        self.logits = nn.Parameter(torch.randn([len(self.inbounds), out_channels]))

        self.save_gate_weights = save_gate_weights
        self.gate_weights = None

    def forward(self, y: torch.Tensor, x: List[Any]) -> torch.Tensor:
        xs = [
            F.pad(x[i], (0, 0, 0, 0, 0, p)) if p != 0 else x[i]
            for i, p in zip(self.inbounds, self.paddings)]
        w2 = self.logits.softmax(dim=0)[None, :, :, None, None]

        if self.save_gate_weights:
            self.gate_weights = w2.detach().cpu().numpy()

        z = weighted_sum(xs, [w.squeeze(1) for w in w2.split(1, dim=1)])

        return y + z


class MeanJunction(nn.Module):
    def __init__(
        self,
        index: int,
        settings: List[Tuple[int, int, int]],
        skip_connections: int,
        **kwargs
    ) -> None:
        super().__init__()
        # index of the begining of the section
        for i in range(index, -1, -1):
            start_index = i
            if settings[i][2] != 1:
                break

        # indexes of skip connections
        inbounds = [index]
        for i in range(skip_connections):
            inbound_index = index - (2 ** (i + 1)) + 1
            if inbound_index < start_index:
                break
            inbounds.append(inbound_index)

        # parameters
        self.index = index
        self.inbounds = inbounds
        self.channels = settings[index][1]
        self.paddings = [settings[index][1] - settings[i][1] for i in inbounds]

    def forward(self, y: torch.Tensor, x: List[Any]) -> torch.Tensor:
        xs = [
            F.pad(x[i], (0, 0, 0, 0, 0, p)) if p != 0 else x[i]
            for i, p in zip(self.inbounds, self.paddings)]

        if len(self.inbounds) == 1:
            return y + xs[0]
        else:
            return y + torch.stack(xs, dim=1).mean(dim=1)


class SumJunction(nn.Module):
    def __init__(
        self,
        index: int,
        settings: List[Tuple[int, int, int]],
        skip_connections: int,
        **kwargs
    ) -> None:
        super().__init__()
        # index of the begining of the section
        for i in range(index, -1, -1):
            start_index = i
            if settings[i][2] != 1:
                break

        # indexes of skip connections
        inbounds = [index]
        for i in range(skip_connections):
            inbound_index = index - (2 ** (i + 1)) + 1
            if inbound_index < start_index:
                break
            inbounds.append(inbound_index)

        # parameters
        self.index = index
        self.inbounds = inbounds
        self.channels = settings[index][1]
        self.paddings = [settings[index][1] - settings[i][1] for i in inbounds]

    def forward(self, y: torch.Tensor, x: List[Any]) -> torch.Tensor:
        xs = [
            F.pad(x[i], (0, 0, 0, 0, 0, p)) if p != 0 else x[i]
            for i, p in zip(self.inbounds, self.paddings)]

        if len(self.inbounds) == 1:
            return y + xs[0]
        else:
            return y + torch.stack(xs, dim=1).sum(dim=1)
