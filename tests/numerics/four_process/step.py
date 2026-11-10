"""Values returned by one four-process update."""
import torch


class Step:
    def __init__(self, forward, losses, gradients, update_order):
        self.forward = forward
        self.losses = losses
        self.gradients = gradients
        self.update_order = tuple(update_order)


def maybe_nan(loss, fault):
    if fault == "nan":
        return loss * float("nan")
    return loss


def apply_gradient_fault(modules, fault):
    if fault == "gradient":
        next(modules["generator"].parameters()).grad.add_(1)


def grads(module, prefix):
    out = {}
    for name, param in module.named_parameters():
        if param.grad is None:
            continue
        out[f"{prefix}.{name}"] = param.grad.detach().cpu().clone()
    return out


def finite(loss, label):
    if not torch.isfinite(loss).all():
        raise ValueError(f"nonfinite loss ({label})")
    return loss
