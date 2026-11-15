"""MossFormerGAN plateau. Five stalls restore the best pair and halve the generator rate."""

from __future__ import annotations

import copy
from typing import Any

from src.experiments.official.mossformergan.reference import mossformer_validation_decision
from src.experiments.official.mossformergan.state import MossProgress


def apply_plateau(
    progress: MossProgress,
    summary: dict[str, Any],
    *,
    generator: Any,
    discriminator: Any,
    optim_g: Any,
    optim_d: Any,
) -> str:
    generator_loss = (summary.get("losses") or {}).get("generator") or {}
    value = generator_loss.get("value")
    if generator_loss.get("status") != "ok" or value is None:
        progress.plateau_history.append({"action": "absent", "logical_step": progress.logical_step})
        return "absent"
    action_loss = float(value)
    progress.best_val, progress.stalls, action = mossformer_validation_decision(
        progress.best_val, progress.stalls, action_loss
    )
    if action == "improve":
        progress.best_snapshot = _capture(generator, discriminator, optim_g, optim_d)
    elif action == "halve":
        _restore_and_halve(progress.best_snapshot, generator, discriminator, optim_g, optim_d)
    elif action == "stop":
        progress.stop_training = True
        progress.halt_reason = "plateau_stop"
    progress.plateau_history.append(
        {
            "action": action,
            "best": progress.best_val,
            "logical_step": progress.logical_step,
            "loss": action_loss,
            "lr_d": float(optim_d.param_groups[0]["lr"]),
            "lr_g": float(optim_g.param_groups[0]["lr"]),
            "stalls": progress.stalls,
        }
    )
    return action


def _capture(generator: Any, discriminator: Any, optim_g: Any, optim_d: Any) -> dict[str, Any]:
    return {
        "discriminator": {
            key: value.detach().cpu().clone() for key, value in discriminator.unwrapped().state_dict().items()
        },
        "generator": {
            key: value.detach().cpu().clone() for key, value in generator.unwrapped().state_dict().items()
        },
        "optim_d": copy.deepcopy(optim_d.state_dict()),
        "optim_g": copy.deepcopy(optim_g.state_dict()),
    }


def _restore_and_halve(snapshot: dict[str, Any] | None, generator: Any, discriminator: Any, optim_g: Any, optim_d: Any) -> None:
    if snapshot is not None:
        generator.unwrapped().load_state_dict(snapshot["generator"], strict=True)
        discriminator.unwrapped().load_state_dict(snapshot["discriminator"], strict=True)
        optim_g.load_state_dict(snapshot["optim_g"])
        optim_d.load_state_dict(snapshot["optim_d"])
    optim_g.param_groups[0]["lr"] *= 0.5
