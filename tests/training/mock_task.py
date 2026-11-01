"""Small linear generator and discriminator for the real Trainer.

This is not a second training loop. ``Trainer.fit`` calls ``train_update``.
The waveforms come from ``RecipeDataset``; the task only reduces each crop to
four deterministic features so the gradient has a closed form.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from src.training.core.distributed import ProcessInfo, unwrap, wrap_module
from src.training.core.rng import seed_model_init
from src.training.core.trainer import accumulation_scope
from src.training.errors import TrainingError
from src.training.se.optim import step_scheduler


def _features(wave: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [wave.mean(dim=-1), wave[:, 0], wave[:, -1], wave.square().mean(dim=-1).clamp_min(0).sqrt()],
        dim=-1,
    )


def _clone(module: torch.nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in unwrap(module).parameters()]


class LinearPairTask:
    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        self.config = config
        self.device = device
        self.events: list[str] = []
        self.case_ids: list[list[str]] = []
        optimizer = config["optimizer"]
        betas = (float(optimizer["betas"][0]), float(optimizer["betas"][1]))
        with seed_model_init(int(config["seed"])):
            self.generator = torch.nn.Linear(4, 4).to(device)
            self.discriminator = torch.nn.Linear(4, 1).to(device)
        self.optim_g = torch.optim.AdamW(
            self.generator.parameters(),
            lr=float(optimizer["lr"]),
            betas=betas,
            weight_decay=float(optimizer["weight_decay"]),
        )
        self.optim_d = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=float(optimizer["lr"]),
            betas=betas,
            weight_decay=float(optimizer["weight_decay"]),
        )
        gamma = float(config["scheduler"]["gamma"])
        self.sched_g = torch.optim.lr_scheduler.ExponentialLR(self.optim_g, gamma=gamma)
        self.sched_d = torch.optim.lr_scheduler.ExponentialLR(self.optim_d, gamma=gamma)

    def distribute(self, process: ProcessInfo) -> None:
        self.generator = wrap_module(self.generator, process)
        self.discriminator = wrap_module(self.discriminator, process)

    def train_mode(self) -> None:
        self.generator.train()
        self.discriminator.train()

    def eval_mode(self) -> None:
        unwrap(self.generator).eval()
        unwrap(self.discriminator).eval()

    def learning_rates(self) -> dict[str, float]:
        return {
            "generator": float(self.optim_g.param_groups[0]["lr"]),
            "discriminator": float(self.optim_d.param_groups[0]["lr"]),
        }

    def step_schedulers(self, validation: dict[str, Any] | None = None) -> None:
        step_scheduler(self.sched_g, validation)
        step_scheduler(self.sched_d, validation)

    def generator_state_dict(self) -> dict[str, torch.Tensor]:
        return unwrap(self.generator).state_dict()

    def state_dict(self) -> dict[str, Any]:
        return {
            "generator": self.generator_state_dict(),
            "discriminator": unwrap(self.discriminator).state_dict(),
            "optim_g": self.optim_g.state_dict(),
            "optim_d": self.optim_d.state_dict(),
            "sched_g": self.sched_g.state_dict(),
            "sched_d": self.sched_d.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if isinstance(self.generator, DistributedDataParallel):
            raise TrainingError("load a checkpoint before wrapping DistributedDataParallel")
        self.generator.load_state_dict(state["generator"])
        self.discriminator.load_state_dict(state["discriminator"])
        self.optim_g.load_state_dict(state["optim_g"])
        self.optim_d.load_state_dict(state["optim_d"])
        self.sched_g.load_state_dict(state["sched_g"])
        self.sched_d.load_state_dict(state["sched_d"])

    def train_update(self, batches: list[dict[str, Any]], accumulation: int) -> dict[str, float]:
        self.train_mode()
        self.case_ids.append([case_id for batch in batches for case_id in batch["case_id"]])
        self.optim_g.zero_grad(set_to_none=True)
        generator_before = _clone(self.generator)
        self.events.append("disc")
        self.optim_d.zero_grad(set_to_none=True)
        disc_total = 0.0
        for index, batch in enumerate(batches):
            last = index == len(batches) - 1
            clean = _features(batch["clean"].to(self.device))
            noisy = _features(batch["noisy"].to(self.device))
            with torch.no_grad():
                fake = unwrap(self.generator)(noisy)
            with accumulation_scope([self.discriminator], last=last):
                ones = torch.ones(clean.shape[0], 1, device=self.device)
                zeros = torch.zeros(clean.shape[0], 1, device=self.device)
                loss = F.mse_loss(self.discriminator(clean), ones) + F.mse_loss(self.discriminator(fake), zeros)
                (loss / accumulation).backward()
            disc_total += float(loss.detach())
        for parameter in unwrap(self.generator).parameters():
            if parameter.grad is not None:
                raise TrainingError("discriminator backward wrote a generator gradient")
        self.optim_d.step()
        if any(not torch.equal(before, after) for before, after in zip(generator_before, _clone(self.generator))):
            raise TrainingError("discriminator step changed generator weights")

        discriminator_before = _clone(self.discriminator)
        self.events.append("gen")
        self.optim_g.zero_grad(set_to_none=True)
        for parameter in unwrap(self.discriminator).parameters():
            parameter.requires_grad_(False)
        gen_total = 0.0
        try:
            for index, batch in enumerate(batches):
                last = index == len(batches) - 1
                clean = _features(batch["clean"].to(self.device))
                noisy = _features(batch["noisy"].to(self.device))
                with accumulation_scope([self.generator], last=last):
                    loss = F.mse_loss(self.generator(noisy), clean)
                    (loss / accumulation).backward()
                gen_total += float(loss.detach())
        finally:
            for parameter in unwrap(self.discriminator).parameters():
                parameter.requires_grad_(True)
        for module in (unwrap(self.generator), unwrap(self.discriminator)):
            for parameter in module.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise TrainingError("non-finite gradient")
        self.optim_g.step()
        if any(not torch.equal(before, after) for before, after in zip(discriminator_before, _clone(self.discriminator))):
            raise TrainingError("generator step changed discriminator weights")
        count = len(batches)
        return {"generator": gen_total / count, "discriminator": disc_total / count}
