"""MossFormerGAN SE 16 kHz training step.

The generator is ``SyncANet`` from ``src.se``. Preprocessing matches the
released solver: energy normalization of the noisy waveform, Hamming window
with ``periodic=False``, ``center=True``, and magnitude compression 0.3.
The loss is ``0.1 * RI + 0.9 * magnitude + 0.2 * time + 0.05 * GAN``.
The time-domain reconstruction uses the released pairing of the clean real
part with the predicted imaginary part.

The discriminator reproduces the released NISQA CNN and Conformer scorer.
Targets are the same: real pairs go to 1 and fakes go to wideband PESQ
normalized by ``(score - 1) / 3.5``. A missing PESQ score or a non-finite
generator loss skips that ``train_update`` group. The target is not replaced
with zero. The returned metrics stay finite and ``update_applied`` is 0.

Update order follows the accumulation branch of the released solver, which is
what the published yaml enables: backward generator, backward discriminator,
then step generator and discriminator. Discriminator learning rate is twice
the generator learning rate. AdamW weight decay is the PyTorch default 0.01
because the released constructor does not pass ``weight_decay``. The YAML
value is not the optimizer value.

The released solver has no ``lr_scheduler`` object. After each validation it
keeps the learning rate, or, on the fifth validation that is not strictly
better, reloads the last improving checkpoint and halves only the generator
learning rate. The tenth such validation stops training. A non-improving
loss replaces the stored best, matching that solver.

Validation uses that same generator objective (RI, magnitude, the released
time term, and the GAN term). It does not minimize time-domain L1 alone.
``model.eval()`` is applied to the generator only. The discriminator stays in
train mode, and neither optimizer is stepped. Official validation does not
add a missing loss and still divides by the batch count. This task reports 0
for that batch so the trainer mean keeps the same denominator.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from src.se import build_model
from src.training.core.distributed import ProcessInfo, unwrap, wrap_module
from src.training.core.rng import seed_model_init
from src.training.core.trainer import accumulation_scope
from src.training.errors import TrainingError
from src.training.se.mossformer_discriminator import Discriminator
from src.training.se.losses import metric_targets
from src.training.se.optim import build_optimizer, build_scheduler, clip_gradients, step_scheduler
from src.training.se.stft import centered_stft

_RI = 0.1
_MAG = 0.9
_TIME = 0.2
_GAN = 0.05
# torch.optim.AdamW default. The released MossFormerGAN constructor omits weight_decay.
OFFICIAL_ADAMW_WEIGHT_DECAY = 0.01

logger = logging.getLogger(__name__)


def mossformer_validation_decision(best: float, stalls: int, loss: float) -> tuple[float, int, str]:
    """Return the next best, stall count, and action for one validation.

    Actions are ``improve``, ``hold``, ``halve``, and ``stop``. This follows
    ``Solver.train`` in ClearerVoice ``train/speech_enhancement/solver.py``:
    a loss that is not strictly lower replaces the stored best and increments
    the stall count. The fifth stall halves the generator learning rate. The
    tenth stall stops. The stall count is not cleared by the halving.
    """
    if loss >= best:
        best = float(loss)
        stalls += 1
        if stalls == 5:
            return best, stalls, "halve"
        if stalls >= 10:
            return best, stalls, "stop"
        return best, stalls, "hold"
    return float(loss), 0, "improve"


class MossFormerGANTask:
    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        self.config = config
        self.device = device
        model = config["model"]
        if config["loss"] != "official":
            raise TrainingError("MossFormerGAN training uses loss official")
        architecture = dict(model.get("architecture") or {})
        self.n_fft = int(architecture.get("n_fft", 400))
        self.win = int(architecture.get("win_len", self.n_fft))
        self.hop = int(architecture.get("hop", 100))
        self.grad_clip = config.get("grad_clip", 10.0)
        self.sample_rate = int(model["sampling_rate"])
        build_config = {key: value for key, value in architecture.items() if key in {"n_fft", "win_len", "hop", "num_channel"}}
        variant = model.get("variant") or "se_16k"
        # ClearerVoice train.py builds two network_wrapper objects. Each wrapper
        # builds two MossFormerGAN_SE_16K modules, and each of those builds
        # SyncANet then Discriminator when mode is train. The kept weights are
        # the first generator (G1) and the fourth discriminator (D4).
        # train.py moves the kept generator to the device before the second wrapper.
        with seed_model_init(config["seed"]):
            generator = _mossformer_generator(build_config, variant)
            _discard_initialized(Discriminator(ndf=16))
            _discard_initialized(_mossformer_generator(build_config, variant))
            _discard_initialized(Discriminator(ndf=16))
            self.generator = generator.to(device)
            _discard_initialized(_mossformer_generator(build_config, variant))
            _discard_initialized(Discriminator(ndf=16))
            _discard_initialized(_mossformer_generator(build_config, variant))
            self.discriminator = Discriminator(ndf=16).to(device)
        self.validation_requires_all_ranks = False
        self._cpu_syncbn_unwrapped = False
        self.discriminator_ddp_broadcast_buffers = False
        self.discriminator_ddp_find_unused_parameters = False
        self.last_skip: dict[str, Any] | None = None
        self.skip_log: list[dict[str, Any]] = []
        generator_config = dict(config)
        generator_config["optimizer"] = dict(config["optimizer"])
        generator_config["optimizer"]["weight_decay"] = OFFICIAL_ADAMW_WEIGHT_DECAY
        self.optim_g = build_optimizer(self.generator.parameters(), generator_config)
        self.optim_d = build_optimizer(self.discriminator.parameters(), generator_config, lr=float(config["optimizer"]["lr"]) * 2)
        self.sched_g = build_scheduler(self.optim_g, config)
        self.sched_d = build_scheduler(self.optim_d, config)
        self._stalls = 0
        self._best_val = float("inf")
        self._best_snapshot: dict[str, Any] | None = None
        self.stop_training = False

    def distribute(self, process: ProcessInfo) -> None:
        # world_size 1 keeps ordinary BatchNorm, matching Solver when distributed is false.
        # Above 1, convert the discriminator only, before DDP. The generator has no
        # live BatchNorm, so convert_sync_batchnorm is not applied to it.
        # torch 2.6 SyncBatchNorm.forward rejects CPU before its gloo all_gather,
        # and DistributedDataParallel rejects a CPU SyncBatchNorm
        # ("SyncBatchNorm layers only work with GPU modules"). CPU keeps the
        # converted module unwrapped. CUDA wraps it.
        self.discriminator_ddp_broadcast_buffers = False
        self.discriminator_ddp_find_unused_parameters = False
        self._cpu_syncbn_unwrapped = False
        if process.world_size > 1:
            self.discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.discriminator)
        self.validation_requires_all_ranks = any(
            isinstance(module, torch.nn.SyncBatchNorm) for module in self.discriminator.modules()
        )
        self.generator = wrap_module(self.generator, process)
        if process.world_size > 1 and process.device.type != "cuda":
            self._cpu_syncbn_unwrapped = True
            return
        # Three discriminator forwards share one autograd graph. DDP's default
        # broadcast_buffers=True would mutate BatchNorm state saved for backward.
        # Official DDP leaves broadcast_buffers at its default True. This wrapper
        # keeps False, and train_update broadcasts the buffers once per micro-batch.
        # find_unused_parameters stays at wrap_module's default False.
        self.discriminator = wrap_module(
            self.discriminator,
            process,
            broadcast_buffers=self.discriminator_ddp_broadcast_buffers,
            find_unused_parameters=self.discriminator_ddp_find_unused_parameters,
        )

    def train_mode(self) -> None:
        self.generator.train()
        self.discriminator.train()

    def eval_mode(self) -> None:
        unwrap(self.generator).eval()
        # Official validation calls generator.eval() only. The discriminator
        # stays in train mode, so its BatchNorm updates running statistics.
        unwrap(self.discriminator).train()

    def learning_rates(self) -> dict[str, float]:
        return {
            "generator": float(self.optim_g.param_groups[0]["lr"]),
            "discriminator": float(self.optim_d.param_groups[0]["lr"]),
        }

    def step_schedulers(self, validation: dict[str, Any] | None = None) -> None:
        step_scheduler(self.sched_g, validation)
        step_scheduler(self.sched_d, validation)
        if not validation:
            return
        value = (validation.get("losses") or {}).get("generator", {}).get("value")
        if value is None:
            return
        self._best_val, self._stalls, action = mossformer_validation_decision(
            self._best_val, self._stalls, float(value)
        )
        if action == "improve":
            self._best_snapshot = self._capture_training_state()
            return
        if action == "halve":
            self._restore_best_and_halve_generator()
            return
        if action == "stop":
            self.stop_training = True

    def _capture_training_state(self) -> dict[str, Any]:
        return {
            "generator": {key: value.detach().cpu().clone() for key, value in unwrap(self.generator).state_dict().items()},
            "discriminator": {
                key: value.detach().cpu().clone() for key, value in unwrap(self.discriminator).state_dict().items()
            },
            "optim_g": copy.deepcopy(self.optim_g.state_dict()),
            "optim_d": copy.deepcopy(self.optim_d.state_dict()),
        }

    def _restore_best_and_halve_generator(self) -> None:
        snapshot = self._best_snapshot
        if snapshot is not None:
            unwrap(self.generator).load_state_dict(snapshot["generator"])
            unwrap(self.discriminator).load_state_dict(snapshot["discriminator"])
            self.optim_g.load_state_dict(snapshot["optim_g"])
            self.optim_d.load_state_dict(snapshot["optim_d"])
        self.optim_g.param_groups[0]["lr"] *= 0.5

    def generator_state_dict(self) -> dict[str, torch.Tensor]:
        return unwrap(self.generator).state_dict()

    def export_for_inference(self) -> dict[str, Any]:
        return {"model": self.generator_state_dict()}

    def state_dict(self) -> dict[str, Any]:
        return {
            "loss": "official",
            "generator": self.generator_state_dict(),
            "discriminator": unwrap(self.discriminator).state_dict(),
            "optim_g": self.optim_g.state_dict(),
            "optim_d": self.optim_d.state_dict(),
            "sched_g": None if self.sched_g is None else self.sched_g.state_dict(),
            "sched_d": None if self.sched_d is None else self.sched_d.state_dict(),
            "plateau": {
                "best_val_loss": self._best_val,
                "val_no_impv": self._stalls,
                "stop": self.stop_training,
                "best_snapshot": self._best_snapshot,
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("loss") != "official":
            raise TrainingError("checkpoint loss does not match MossFormerGAN")
        if isinstance(self.generator, DistributedDataParallel):
            raise TrainingError("load a checkpoint before wrapping DistributedDataParallel")
        self.generator.load_state_dict(state["generator"], strict=True)
        self.discriminator.load_state_dict(state["discriminator"], strict=True)
        self.optim_g.load_state_dict(state["optim_g"])
        self.optim_d.load_state_dict(state["optim_d"])
        plateau = state.get("plateau") or {}
        self._best_val = float(plateau.get("best_val_loss", float("inf")))
        self._stalls = int(plateau.get("val_no_impv", 0))
        self.stop_training = bool(plateau.get("stop", False))
        self._best_snapshot = plateau.get("best_snapshot")

    def train_update(self, batches: list[dict[str, Any]], accumulation: int) -> dict[str, float]:
        """One optimizer update, or a skipped group when PESQ or the generator loss fails.

        Official ``loss_mossformergan_se_16k`` returns ``None, None`` when the
        generator loss is NaN or PESQ is missing. ``Solver`` then does not
        backward and does not ``optimizer.step``. Ranks must take that branch
        together: a local skip flag is reduced with MAX before any backward.
        If any rank skips, every rank skips backward, clipping, and both
        optimizer steps for this whole group. The group is not retried.

        Forwards that already ran stay in ``no_sync`` so a skipped rank does
        not leave DDP waiting for backward. On a kept group the last
        micro-batch averages generator and discriminator gradients with SUM
        and ``world_size`` after each backward and before that backward's
        clip. Earlier micro-batches are still clipped before any cross-rank
        average. That is the same clip-versus-average order as before
        (P10P-D-001 is unchanged).

        ``update_applied`` is 1 when both optimizers stepped and 0 when the
        group was skipped. Skipped generator and discriminator values are 0.
        The trainer must not increment ``global_step`` when ``update_applied``
        is 0. This method cannot change the trainer.
        """
        self.train_mode()
        self.optim_g.zero_grad(set_to_none=True)
        self.optim_d.zero_grad(set_to_none=True)
        prepared: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        for index, batch in enumerate(batches):
            is_last = index == len(batches) - 1
            self._broadcast_discriminator_buffers()
            clean = batch["clean"].to(self.device)
            noisy = batch["noisy"].to(self.device)
            # no_sync on every forward. A later skip must not leave the DDP
            # reducer waiting for backward. The last micro-batch averages below.
            with accumulation_scope([self.generator], last=False), accumulation_scope([self.discriminator], last=False):
                terms = self._generator_terms(self.generator, self.discriminator, clean, noisy)
            generator_loss = terms["generator"]
            reason = None
            targets = None
            if not torch.isfinite(generator_loss).all():
                reason = "non-finite generator loss"
            else:
                targets = metric_targets(terms["teacher"], terms["audio"], self.sample_rate)
                if targets is None:
                    reason = "pesq"
            if reason is not None:
                events.append(_skip_event(reason, batch, index))
            if self._agree_skip(reason is not None):
                if reason is None:
                    events.append(_skip_event("peer", batch, index))
                return self._commit_skip(events, len(batches))
            assert targets is not None
            with accumulation_scope([self.discriminator], last=False):
                ones = torch.ones(clean.shape[0], device=self.device)
                fake_detached = self.discriminator(terms["label_mag"], terms["pred_mag"].detach())
                real_score = self.discriminator(terms["label_mag"], terms["label_mag"])
                discriminator_loss = F.mse_loss(real_score.flatten(), ones) + F.mse_loss(
                    fake_detached.flatten(), targets.to(self.device)
                )
            disc_reason = None
            if not torch.isfinite(discriminator_loss).all():
                disc_reason = "non-finite discriminator loss"
                events.append(_skip_event(disc_reason, batch, index))
            if self._agree_skip(disc_reason is not None):
                if disc_reason is None:
                    events.append(_skip_event("peer", batch, index))
                return self._commit_skip(events, len(batches))
            prepared.append(
                {
                    "is_last": is_last,
                    "generator_loss": generator_loss,
                    "discriminator_loss": discriminator_loss,
                }
            )
        gen_total = 0.0
        disc_total = 0.0
        for item in prepared:
            generator_loss = item["generator_loss"]
            discriminator_loss = item["discriminator_loss"]
            (generator_loss / accumulation).backward()
            if item["is_last"]:
                self._average_grads(self.generator)
            clip_gradients(unwrap(self.generator), self.grad_clip)
            (discriminator_loss / accumulation).backward()
            if item["is_last"] and self._discriminator_needs_cross_rank():
                self._average_named_grads(unwrap(self.discriminator))
            clip_gradients(unwrap(self.discriminator), self.grad_clip)
            gen_total += float(generator_loss.detach())
            disc_total += float(discriminator_loss.detach())
        self.optim_g.step()
        self.optim_d.step()
        count = len(batches)
        return {"generator": gen_total / count, "discriminator": disc_total / count, "update_applied": 1.0}

    def _discriminator_needs_cross_rank(self) -> bool:
        return isinstance(self.discriminator, DistributedDataParallel) or self._cpu_syncbn_unwrapped

    def _broadcast_discriminator_buffers(self) -> None:
        if not self._discriminator_needs_cross_rank():
            return
        if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() < 2:
            return
        with torch.no_grad():
            for buffer in unwrap(self.discriminator).buffers():
                dist.broadcast(buffer, src=0)

    def _agree_skip(self, local_skip: bool) -> bool:
        """True when this rank or any other rank skips the whole group."""
        if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() < 2:
            return local_skip
        flag = torch.tensor([1 if local_skip else 0], dtype=torch.int32)
        if self.device.type == "cuda":
            flag = flag.to(self.device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return int(flag.item()) == 1

    def _average_grads(self, module: torch.nn.Module) -> None:
        if not isinstance(module, DistributedDataParallel):
            return
        self._average_named_grads(unwrap(module))

    def _average_named_grads(self, module: torch.nn.Module) -> None:
        if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() < 2:
            return
        world = dist.get_world_size()
        for parameter in module.parameters():
            if parameter.grad is None:
                continue
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(world)

    def _commit_skip(self, events: list[dict[str, Any]], group_size: int) -> dict[str, float]:
        primary = events[0] if events else {"reason": "peer", "case_ids": None, "micro_batch_index": None}
        record = {
            "reason": primary["reason"],
            "case_ids": primary["case_ids"],
            "micro_batch_index": primary["micro_batch_index"],
            "count": group_size,
            "events": list(events),
        }
        self.last_skip = record
        self.skip_log.append(record)
        for event in events:
            logger.info(
                "mossformer skip reason=%s case_ids=%s micro_batch_index=%s count=%s",
                event["reason"],
                event["case_ids"],
                event["micro_batch_index"],
                group_size,
            )
        return {"generator": 0.0, "discriminator": 0.0, "update_applied": 0.0}

    def validation_report(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Official validation generator loss for one already-cropped batch.

        The 2 second crop is applied by the dataset rule before this method
        runs. This method does not step either optimizer. PESQ itself is not
        the plateau metric.

        Official ``Solver._run_one_epoch_mossformergan_se_16k`` adds the loss
        only when it is not None, then returns ``total_loss / (i+1)`` for both
        train and validation (the division is outside ``state=='train'``).
        A missing loss is not written as ``+= 0`` in that source. The batch
        still stays in the denominator. This report uses value 0 and status
        ok so the trainer average keeps that batch. A missing PESQ score does
        not raise. A non-finite generator loss still raises here, which is
        narrower than the official ``None`` return.
        """
        clean = batch["clean"].to(self.device)
        noisy = batch["noisy"].to(self.device)
        generator = unwrap(self.generator)
        discriminator = unwrap(self.discriminator)
        generator.eval()
        discriminator.train()
        with torch.no_grad():
            terms = self._generator_terms(generator, discriminator, clean, noisy)
            _finite(terms["generator"], "mossformer validation generator")
            targets = metric_targets(terms["teacher"], terms["audio"], self.sample_rate)
        included = targets is not None
        if included:
            value = float(terms["generator"].detach())
            reason = "official generator loss; pesq gate included this batch"
        else:
            value = 0.0
            reason = (
                "pesq failed; official validation skips the add and divides by "
                "the batch count, so this report contributes 0"
            )
        losses = {
            "generator": _metric(value, reason=reason),
            "ri": _component(terms["ri"], included),
            "magnitude": _component(terms["magnitude"], included),
            "time": _component(terms["time"], included),
            "gan": _component(terms["gan"], included),
            "discriminator": _metric(
                status="na",
                reason="validation does not step the discriminator",
            ),
        }
        return {
            "losses": losses,
            "pesq": {
                "status": "na",
                "value": None,
                "reason": "pesq gates inclusion and is not the plateau metric",
                "count": 0,
            },
            "stoi": {"status": "na", "value": None, "reason": "not computed", "count": 0},
            "pesq_gate": "included" if included else "counted_as_zero",
        }

    def participating_validation_forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Validation forward every rank must enter when SyncBatchNorm is active.

        ``validation_report`` itself has no rank-0-only collective. The
        collectives are inside the discriminator's SyncBatchNorm forwards.
        Calling this method, or ``validation_report``, on rank 0 only deadlocks
        when ``validation_requires_all_ranks`` is True. The current trainer
        validates on rank 0 only; that call pattern is unsafe for this task
        once ``distribute`` has converted the discriminator.
        """
        return self.validation_report(batch)

    def _generator_terms(self, module, discriminator, clean: torch.Tensor, noisy: torch.Tensor) -> dict[str, torch.Tensor]:
        """Released ``loss_mossformergan_se_16k`` generator terms.

        The time term reconstructs from the clean real part and the predicted
        imaginary part. The GAN term asks the discriminator to score the fake
        pair as 1. Discriminator targets that use PESQ are not part of this sum.
        """
        scale = torch.sqrt(noisy.shape[-1] / (noisy.square().sum(dim=-1) + 1e-6)).unsqueeze(-1)
        normalized = noisy * scale
        real, imag, pred_mag, label_real, _label_imag, label_mag = self._forward(
            module, normalized, clean * scale
        )
        ones = torch.ones(clean.shape[0], device=clean.device)
        gan = F.mse_loss(discriminator(label_mag, pred_mag).flatten(), ones)
        loss_mag = F.mse_loss(pred_mag, label_mag)
        loss_ri = F.mse_loss(real, label_real) + F.mse_loss(imag, _label_imag)
        audio = _istft(_uncompress(label_real, imag).squeeze(1), self.n_fft, self.hop, self.win, clean.shape[-1])
        teacher = clean * scale
        length = min(audio.shape[-1], teacher.shape[-1])
        time_loss = (audio[..., :length] - teacher[..., :length]).abs().mean()
        generator_loss = _RI * loss_ri + _MAG * loss_mag + _TIME * time_loss + _GAN * gan
        return {
            "generator": generator_loss,
            "ri": loss_ri,
            "magnitude": loss_mag,
            "time": time_loss,
            "gan": gan,
            "audio": audio[..., :length],
            "teacher": teacher[..., :length],
            "label_mag": label_mag,
            "pred_mag": pred_mag,
            "predicted_real": real,
            "predicted_imag": imag,
            "label_real": label_real,
        }

    def _forward(self, module, noisy: torch.Tensor, clean: torch.Tensor):
        noisy_spec = _compress(_stft(noisy, self.n_fft, self.hop, self.win)).permute(0, 1, 3, 2)
        real, imag = module(noisy_spec)
        real = real.permute(0, 1, 3, 2)
        imag = imag.permute(0, 1, 3, 2)
        pred_mag = torch.sqrt(real ** 2 + imag ** 2)
        label_spec = _compress(_stft(clean, self.n_fft, self.hop, self.win))
        label_real = label_spec[:, 0, :, :].unsqueeze(1)
        label_imag = label_spec[:, 1, :, :].unsqueeze(1)
        label_mag = torch.sqrt(label_real ** 2 + label_imag ** 2)
        return real, imag, pred_mag, label_real, label_imag, label_mag


def _mossformer_generator(build_config: dict[str, Any], variant: str) -> torch.nn.Module:
    return build_model("mossformergan", build_config or None, variant=variant)


def _discard_initialized(module: torch.nn.Module) -> None:
    """Drop a module after its constructor has run ``reset_parameters``."""
    del module


def _skip_event(reason: str, batch: dict[str, Any], index: int) -> dict[str, Any]:
    case_ids = None
    if "case_id" in batch:
        raw = batch["case_id"]
        if isinstance(raw, (list, tuple)):
            case_ids = [str(item) for item in raw]
        else:
            case_ids = [str(raw)]
    return {"reason": reason, "case_ids": case_ids, "micro_batch_index": index}


def _stft(waveform: torch.Tensor, n_fft: int, hop: int, win: int) -> torch.Tensor:
    # The released solver constructs the window on CPU before moving it.
    window = torch.hamming_window(win, periodic=False, dtype=waveform.dtype).to(waveform.device)
    # Same frames as center=True. The pad is an index reflection so CUDA
    # deterministic mode does not need reflection_pad1d_backward.
    return torch.view_as_real(centered_stft(waveform, n_fft, hop, window))


def _istft(spectrum: torch.Tensor, n_fft: int, hop: int, win: int, length: int) -> torch.Tensor:
    window = torch.hamming_window(win, periodic=False, dtype=spectrum.dtype).to(spectrum.device)
    return torch.istft(torch.view_as_complex(spectrum), n_fft, hop, win, window=window, center=False)


def _compress(spectrum: torch.Tensor) -> torch.Tensor:
    value = torch.complex(spectrum[..., 0], spectrum[..., 1])
    magnitude = value.abs().pow(0.3)
    return torch.stack((magnitude * torch.cos(value.angle()), magnitude * torch.sin(value.angle())), 1)


def _uncompress(real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
    value = torch.complex(real, imag)
    magnitude = value.abs().pow(1.0 / 0.3)
    phase = value.angle()
    return torch.stack((magnitude * torch.cos(phase), magnitude * torch.sin(phase)), -1)


def _finite(loss: torch.Tensor, name: str) -> None:
    if not torch.isfinite(loss).all():
        raise TrainingError(f"non-finite {name} loss")


def _component(value: torch.Tensor, included: bool) -> dict[str, Any]:
    if not included:
        return _metric(status="na", reason="batch excluded by the pesq gate")
    return _metric(float(value.detach()))


def _metric(value: float | None = None, *, status: str = "ok", reason: str | None = None) -> dict[str, Any]:
    if status == "ok" and (value is None or value != value or value in (float("inf"), float("-inf"))):
        return {"status": "na", "value": None, "reason": "non-finite loss"}
    return {"status": status, "value": value, "reason": reason}
