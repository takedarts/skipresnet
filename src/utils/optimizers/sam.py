'''
Sharpness-Aware Minimization for Efficiently Improving Generalization.
'''
import torch


class SAM(torch.optim.SGD):
    def __init__(
        self,
        params,
        lr: float,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        rho: float = 0.05,
    ) -> None:
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        super().__init__(
            params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)

        self.defaults['rho'] = rho

        for param_group in self.param_groups:
            param_group.setdefault('rho', rho)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        super().step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        if closure is None:
            raise NotImplementedError(
                "SAM doesn't work like the other optimizers,"
                + " you should first call `first_step` and the `second_step`;"
                + " see the documentation for more info.")

        # first step
        closure()
        self.first_step(zero_grad=False)

        # second step
        closure()
        self.second_step(zero_grad=False)

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None]), p=2)
        return norm
