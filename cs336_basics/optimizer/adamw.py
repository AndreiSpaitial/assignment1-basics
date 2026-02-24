from collections.abc import Callable

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-1,
        weight_decay: float = 0.,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-5,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "eps": eps,
        }
        super().__init__(params, defaults)
        self.lr_scheduler = None
        self.gradient_clipping = None

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta_1, beta_2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            if self.gradient_clipping:
                self.gradient_clipping(group["params"])
            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad.data
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(g))
                v = state.get("v", torch.zeros_like(g))

                m = beta_1*m + (1-beta_1)*g
                v = beta_2*v + (1-beta_2)*g**2

                lr_adjusted = lr
                if self.lr_scheduler is not None:
                    lr_adjusted = self.lr_scheduler(t)
                lr_t = lr_adjusted*(1-beta_2**t)**0.5/(1-beta_1**t)

                p.data -= lr_t*m/(v**0.5+eps)
                p.data -= lr*weight_decay*p.data

                state["t"] = t+1
                state["m"] = m
                state["v"] = v

        return loss
