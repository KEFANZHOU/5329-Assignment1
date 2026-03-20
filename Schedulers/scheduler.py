from Schedulers.cosine_scheduler import CosineAnnealingLR
from Schedulers.lambda_scheduler import LambdaLR, make_warmup_lambda
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
    """Linear warmup to args.learning_rate, then inverse-sqrt decay."""
    return LambdaLR(
        optimizer,
        lr_lambda=make_warmup_lambda(
            peak_lr=getattr(args, "learning_rate", 1e-3),
            warmup_steps=getattr(args, "warmup_steps", 4000),
        ),
    )

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
