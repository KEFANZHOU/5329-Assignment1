from Schedulers.cosine_scheduler import CosineAnnealingLR
from Schedulers.lambda_scheduler import LambdaLR
from Schedulers.step_scheduler import StepLR


# ── Scheduler factories ──────────────────────────────────────────────────────

def cosine_scheduler(optimizer, args):
    """Cosine annealing over the full training run."""
    return CosineAnnealingLR(
        optimizer,
        T_max=args.num_steps,
    )


def step_scheduler(optimizer, args):
    """Step decay: multiply LR by gamma every lr_step_size steps."""
    return StepLR(
        optimizer,
        step_size=getattr(args, "lr_step_size", 10000),
        gamma=getattr(args, "lr_gamma", 0.5),
    )


def lambda_scheduler(optimizer, args):
    """LambdaLR with a constant factor of 1.0 — learning rate stays fixed."""
    return LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

class NoOpScheduler:
    """No operation scheduler."""
    def __init__(self, optimizer):
        self.optimizer = optimizer
    def step(self): pass
    def get_last_lr(self):
        return [g['lr'] for g in self.optimizer.param_groups]
    def state_dict(self): return {}
    def load_state_dict(self, d): pass

def none_scheduler(optimizer, args):
    return NoOpScheduler(optimizer)

# ── Registry ─────────────────────────────────────────────────────────────────

schedulers = {
    "cosine":  cosine_scheduler,
    "step":    step_scheduler,
    "lambda":  lambda_scheduler,
    "none":    none_scheduler,
}
