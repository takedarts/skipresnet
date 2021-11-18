import torch
import torch.nn as nn
import torch.autograd as autograd


class AdjustedStackFunction(autograd.Function):

    @staticmethod
    def forward(ctx, *xs):
        ctx.channels = [x.shape[1] for x in xs]
        base = xs[0]

        y = torch.zeros(
            base.shape[0], len(xs), base.shape[1], *base.shape[2:],
            device=base.device, dtype=base.dtype)

        for i, x in enumerate(xs):
            if x.shape[1] < y.shape[2]:
                y[:, i, :x.shape[1]] = x
            elif x.shape[1] > y.shape[2]:
                y[:, i] = x[:, :y.shape[2]]
            else:
                y[:, i] = x

        return y

    @staticmethod
    def backward(ctx, g):
        channels = ctx.channels[0]
        g_xs = []

        for i, c in enumerate(ctx.channels):
            g_x = g[:, i, :min(c, channels)]

            if c > channels:
                g_x = nn.functional.pad(g_x, (0, 0, 0, 0, 0, c - channels))

            g_xs.append(g_x)

        return tuple(g_xs)


def adjusted_stack(xs, channels=None):
    xs = list(xs)

    if channels is not None:
        if xs[0].shape[1] > channels:
            xs[0] = xs[0][:, :channels]
        elif xs[0].shape[1] < channels:
            xs[0] = nn.functional.pad(xs[0], (0, 0, 0, 0, 0, channels - xs[0].shape[1]))

    return AdjustedStackFunction.apply(*xs)


def adjusted_concat(xs, channels=None):
    y = adjusted_stack(xs, channels=channels)

    return y.reshape(y.shape[0], -1, *y.shape[3:])
