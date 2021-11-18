import torch
import torch.autograd as autograd


class SignalAugmentFunction(autograd.Function):

    @staticmethod
    def forward(ctx, x, std, dims=1):
        if std != 0:
            size = list(x.shape[:dims]) + [1] * (len(x.shape) - dims)
            noise = torch.randn(size, device=x.device, requires_grad=False) * std + 1.0

            return x * noise
        else:
            return x

    @staticmethod
    def backward(ctx, grad):
        return grad, None, None


signal_augment = SignalAugmentFunction.apply
