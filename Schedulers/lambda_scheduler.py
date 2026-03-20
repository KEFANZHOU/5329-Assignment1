from torch.optim.lr_scheduler import LRScheduler


class LambdaLR(LRScheduler):
    """Multiplies the learning rate of each param group by the output of a
    user-supplied function of the current step:

        lr_t = base_lr * lr_lambda(t)

    Args:
        optimizer:  wrapped optimizer
        lr_lambda:  a function that takes the current step (int) and returns
                    a multiplicative factor (float)
    """

    def __init__(self, optimizer, lr_lambda, last_epoch=-1):
        self.lr_lambda = lr_lambda
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        t = self.last_epoch
        factor = self.lr_lambda(t)
        return [base_lr * factor for base_lr in self.base_lrs]


class _WarmupInvSqrtLambda:
    """Picklable callable implementing warmup + inverse-sqrt decay."""

    def __init__(self, peak_lr: float, warmup_steps: int):
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps

    def __call__(self, step: int) -> float:
        step = max(1, step)
        if step <= self.warmup_steps:
            return self.peak_lr * step / self.warmup_steps
        return self.peak_lr * (self.warmup_steps ** 0.5) / (step ** 0.5)


def make_warmup_lambda(peak_lr: float, warmup_steps: int):
    """Return a warmup + inverse-sqrt decay schedule.

    The returned callable emits the effective learning rate directly. It is
    intended to be paired with optimizers whose base learning rate is 1.0.
    """
    if peak_lr <= 0.0:
        raise ValueError(f"peak_lr must be positive, got {peak_lr}")
    if warmup_steps <= 0:
        raise ValueError(f"warmup_steps must be positive, got {warmup_steps}")

    return _WarmupInvSqrtLambda(peak_lr, warmup_steps)
