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
loss skips that micro-batch. Later successful micro-batches still fill the
accumulation window. The target is not replaced with zero. Returned loss
values stay finite. ``update_applied`` is 1 only when an optimizer step ran.

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
add a missing loss and still divides by the batch count. A failed batch is
reported as missing, not as a loss of 0. The reduced value divides the sum
of included losses by every batch, which is that quotient.
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
from src.experiments.calc.distributed import ProcessInfo, unwrap, wrap_module
from src.experiments.calc.rng import seed_model_init
from src.experiments.calc.scope import accumulation_scope
from src.experiments.calc.errors import TrainingError
from src.experiments.official.mossformergan.discriminator import Discriminator
from src.experiments.calc.losses import metric_targets
from src.experiments.calc.optim import build_optimizer, build_scheduler, clip_gradients, step_scheduler
from src.experiments.calc.stft import centered_stft

_RI = 0.1
_MAG = 0.9
_TIME = 0.2
_GAN = 0.05
# torch.optim.AdamW default. The released MossFormerGAN constructor omits weight_decay.
OFFICIAL_ADAMW_WEIGHT_DECAY = 0.01

logger = logging.getLogger(__name__)


def _empty_counters() -> dict[str, int]:
    return {
        "micro_batches_processed": 0,
        "micro_batches_succeeded": 0,
        "micro_batches_failed": 0,
        "optimizer_updates": 0,
        "official_step": 0,
    }


def average_new_grad(previous: torch.Tensor | None, local: torch.Tensor, peer_locals: list[torch.Tensor]) -> torch.Tensor:
    """Average this micro-batch's gradient, then add it to the running sum.

    ``local`` is this rank's newly computed gradient. Peers contribute the
    same micro-batch only. Earlier micro-batches already stored in
    ``previous`` are not averaged again.
    """
    total = local.detach().clone()
    for peer in peer_locals:
        total = total + peer.to(device=total.device, dtype=total.dtype)
    averaged = total / float(1 + len(peer_locals))
    if previous is None:
        return averaged
    return previous.detach() + averaged


def local_validation_totals(reports: list[dict[str, Any]]) -> dict[str, float]:
    """Official per-rank validation totals.

    Alignment forwards that exist only to keep SyncBatchNorm in step are not
    part of the denominator. PESQ failures stay in the denominator and do not
    add a numeric loss. Their contribution is absence, not a stored zero.
    """
    loss_sum = 0.0
    batches = 0
    failed = 0
    included = 0
    alignment = 0
    for report in reports:
        if report.get("alignment_pad"):
            alignment += 1
            continue
        batches += 1
        if report.get("generator_loss_included") and report["losses"]["generator"]["status"] == "ok":
            loss_sum += float(report["losses"]["generator"]["value"])
            included += 1
        else:
            failed += 1
    mean = 0.0 if batches == 0 else loss_sum / batches
    return {
        "sum": loss_sum,
        "batches": float(batches),
        "failed": float(failed),
        "included": float(included),
        "alignment": float(alignment),
        "mean": mean,
    }


def official_validation_mean(rank_means: list[float]) -> float:
    """Mean of per-rank means, matching ``Solver._reduce_tensor``."""
    if not rank_means:
        raise TrainingError("validation produced no batches")
    return float(sum(rank_means) / len(rank_means))


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
    # Plateau follows the data epoch, not ``global_step % validation_interval``.
    validate_on_data_epoch_end = True

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
        self.validation_order = "rank_mod_fixed_v1"
        self._cpu_syncbn_unwrapped = False
        # Official DDP uses find_unused_parameters=True and the default
        # broadcast_buffers=True. A measured 3-forward, 2-backward step
        # completes with True, so the discriminator uses that default.
        # DDP broadcasts buffers itself; a second manual broadcast is not added.
        self.generator_ddp_broadcast_buffers = True
        self.generator_ddp_find_unused_parameters = True
        self.discriminator_ddp_broadcast_buffers = True
        self.discriminator_ddp_find_unused_parameters = True
        self.last_skip: dict[str, Any] | None = None
        self.skip_log: list[dict[str, Any]] = []
        self._accu_count = 0
        self.counters = _empty_counters()
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
        # world_size 1 keeps ordinary BatchNorm. Official multi-GPU training
        # is CUDA, and only that path converts to SyncBatchNorm. CPU world 2
        # keeps BatchNorm so forward does not require a GPU tensor.
        # The live generator has no BatchNorm, so convert_sync_batchnorm does
        # not add SyncBatchNorm modules there. The discriminator does.
        self.generator_ddp_broadcast_buffers = True
        self.generator_ddp_find_unused_parameters = True
        self.discriminator_ddp_broadcast_buffers = True
        self.discriminator_ddp_find_unused_parameters = True
        self._cpu_syncbn_unwrapped = False
        cuda_sync = process.world_size > 1 and process.device.type == "cuda"
        if cuda_sync:
            self.generator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.generator)
            self.discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.discriminator)
        self.validation_requires_all_ranks = cuda_sync and any(
            isinstance(module, torch.nn.SyncBatchNorm) for module in self.discriminator.modules()
        )
        self.generator = wrap_module(
            self.generator,
            process,
            broadcast_buffers=self.generator_ddp_broadcast_buffers,
            find_unused_parameters=self.generator_ddp_find_unused_parameters,
        )
        if process.world_size > 1:
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
            "accumulation": {
                "count": int(getattr(self, "_accu_count", 0)),
                "counters": dict(getattr(self, "counters", _empty_counters())),
                "generator_grads": self._grad_state(self.generator) if getattr(self, "_accu_count", 0) else {},
                "discriminator_grads": self._grad_state(self.discriminator) if getattr(self, "_accu_count", 0) else {},
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
        accumulation_state = state.get("accumulation")
        if isinstance(accumulation_state, dict):
            self._accu_count = int(accumulation_state.get("count") or 0)
            stored = accumulation_state.get("counters") or {}
            self.counters = _empty_counters()
            for key in self.counters:
                self.counters[key] = int(stored.get(key) or 0)
            MossFormerGANTask._restore_grads(self, self.generator, accumulation_state.get("generator_grads") or {})
            MossFormerGANTask._restore_grads(self, self.discriminator, accumulation_state.get("discriminator_grads") or {})
        else:
            self._accu_count = 0
            self.counters = _empty_counters()

    def start_epoch(self, *, resume_mid_epoch: bool = False) -> None:
        """Drop a partial accumulation window at an epoch boundary.

        Official ``_run_one_epoch`` sets ``accu_count`` to 0 and zeroes both
        optimizers at the start of every epoch. A mid-epoch resume keeps the
        window that was stored with the checkpoint.
        """
        if resume_mid_epoch:
            return
        self.optim_g.zero_grad(set_to_none=True)
        self.optim_d.zero_grad(set_to_none=True)
        self._accu_count = 0

    def train_update(self, batches: list[dict[str, Any]], accumulation: int) -> dict[str, float]:
        """Continue the official accumulation window across this micro-batch group.

        Official ``loss_mossformergan_se_16k`` returns ``None, None`` when the
        generator loss is NaN or PESQ is missing. That micro-batch does not
        backward and does not increment ``accu_count``. ``self.step`` still
        increments. The next successful micro-batch keeps filling the same
        window. One GPU follows that rule.

        Two or more ranks take the skip together. A local flag is reduced
        with MAX before backward. If any rank skips a micro-batch, every rank
        skips backward for that micro-batch. The window index stays aligned.
        This is not the official per-rank skip, which would leave the other
        rank inside a DDP collective.

        Forwards stay in ``no_sync``. After each successful backward, only the
        new gradient is averaged across ranks, added to the running sum, and
        then clipped. On PyTorch 2.6, re-averaging an accumulated gradient that
        is already identical on every rank matches that increment average.
        The released loop does not use ``no_sync``. Its discriminator is
        forwarded three times and then backward runs twice. The second
        backward does not all-reduce, so discriminator weights diverge across
        ranks. This method still averages the discriminator-loss increment.
        That is the safe stand-in, not a copy of the divergent step.
        World size 1 uses the released order: scale by the accumulation count,
        backward, clip that module, and do this again for the other loss.
        Discriminator DDP uses ``broadcast_buffers=True``. That broadcast is
        the DDP forward itself, so this method does not broadcast again.

        ``update_applied`` is 1 when this call executed an optimizer step.
        The trainer's ``global_step`` counts those steps. ``official_step``
        counts every micro-batch, including failures. A call can step once
        and still leave a partial window for the next call.
        """
        if accumulation < 1:
            raise TrainingError("MossFormerGAN accumulation must be positive")
        self.train_mode()
        successes: list[tuple[float, float]] = []
        stepped = False
        for index, batch in enumerate(batches):
            clean = batch["clean"].to(self.device)
            noisy = batch["noisy"].to(self.device)
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
            self.counters["micro_batches_processed"] += 1
            self.counters["official_step"] += 1
            if self._agree_skip(reason is not None):
                if reason is None:
                    reason = "peer"
                self._note_skip(_skip_event(reason, batch, index))
                self.counters["micro_batches_failed"] += 1
                continue
            assert targets is not None
            with accumulation_scope([self.discriminator], last=False):
                ones = torch.ones(clean.shape[0], device=self.device)
                fake_detached = self.discriminator(terms["label_mag"], terms["pred_mag"].detach())
                real_score = self.discriminator(terms["label_mag"], terms["label_mag"])
                discriminator_loss = F.mse_loss(real_score.flatten(), ones) + F.mse_loss(
                    fake_detached.flatten(), targets.to(self.device)
                )
            disc_reason = None if torch.isfinite(discriminator_loss).all() else "non-finite discriminator loss"
            if self._agree_skip(disc_reason is not None):
                self._note_skip(_skip_event(disc_reason or "peer", batch, index))
                self.counters["micro_batches_failed"] += 1
                continue
            self.counters["micro_batches_succeeded"] += 1
            self._accu_count += 1
            self._backward_average_and_clip(generator_loss / accumulation, unwrap(self.generator))
            self._backward_average_and_clip(discriminator_loss / accumulation, unwrap(self.discriminator))
            successes.append((float(generator_loss.detach()), float(discriminator_loss.detach())))
            if self._accu_count == accumulation:
                self.optim_g.step()
                self.optim_d.step()
                self.optim_g.zero_grad(set_to_none=True)
                self.optim_d.zero_grad(set_to_none=True)
                self._accu_count = 0
                self.counters["optimizer_updates"] += 1
                stepped = True
        if successes:
            generator = sum(item[0] for item in successes) / len(successes)
            discriminator = sum(item[1] for item in successes) / len(successes)
        else:
            generator = 0.0
            discriminator = 0.0
        return {
            "generator": generator,
            "discriminator": discriminator,
            "update_applied": 1.0 if stepped else 0.0,
        }

    def _discriminator_needs_cross_rank(self) -> bool:
        return isinstance(self.discriminator, DistributedDataParallel) or self._cpu_syncbn_unwrapped

    def _cross_rank(self) -> bool:
        return self._discriminator_needs_cross_rank() or isinstance(self.generator, DistributedDataParallel)

    def _agree_skip(self, local_skip: bool) -> bool:
        """True when this rank or any other rank skips this micro-batch."""
        if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() < 2:
            return local_skip
        flag = torch.tensor([1 if local_skip else 0], dtype=torch.int32)
        if self.device.type == "cuda":
            flag = flag.to(self.device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return int(flag.item()) == 1

    def _backward_average_and_clip(self, loss: torch.Tensor, clip_module: torch.nn.Module) -> None:
        """Backward, average only the new gradient, then clip the running sum.

        World size 1 clips whatever ``backward`` just added to ``.grad``.
        A larger world averages the new increment before that clip. Clipping
        the sum once at the end of the window is a different update once a
        micro-batch norm exceeds the threshold.
        """
        modules = (unwrap(self.generator), unwrap(self.discriminator))
        before: dict[int, torch.Tensor | None] = {}
        for module in modules:
            for parameter in module.parameters():
                if not parameter.requires_grad:
                    continue
                before[id(parameter)] = None if parameter.grad is None else parameter.grad.detach().clone()
        loss.backward()
        if self._cross_rank() and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            world = dist.get_world_size()
            for module in modules:
                for parameter in module.parameters():
                    if parameter.grad is None:
                        continue
                    old = before.get(id(parameter))
                    raw = parameter.grad if old is None else parameter.grad - old
                    delta = raw.detach().contiguous()
                    dist.all_reduce(delta, op=dist.ReduceOp.SUM)
                    delta.div_(world)
                    parameter.grad = delta if old is None else old + delta
        clip_gradients(clip_module, self.grad_clip)

    def _note_skip(self, event: dict[str, Any]) -> None:
        record = {
            "reason": event["reason"],
            "case_ids": event["case_ids"],
            "micro_batch_index": event["micro_batch_index"],
            "count": 1,
            "events": [event],
        }
        self.last_skip = record
        self.skip_log.append(record)
        logger.info(
            "mossformer skip reason=%s case_ids=%s micro_batch_index=%s count=%s",
            event["reason"],
            event["case_ids"],
            event["micro_batch_index"],
            1,
        )

    def _grad_state(self, module: torch.nn.Module) -> dict[str, torch.Tensor]:
        grads = {}
        for name, parameter in unwrap(module).named_parameters():
            if parameter.grad is not None:
                grads[name] = parameter.grad.detach().cpu().clone()
        return grads

    def _restore_grads(self, module: torch.nn.Module, grads: dict[str, Any]) -> None:
        named = dict(unwrap(module).named_parameters())
        for parameter in named.values():
            parameter.grad = None
        for name, value in grads.items():
            parameter = named[name]
            parameter.grad = value.detach().to(device=parameter.device, dtype=parameter.dtype)

    def validation_report(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Official validation generator loss for one already-cropped batch.

        The 2 second crop is applied by the dataset rule before this method
        runs. This method does not step either optimizer. PESQ itself is not
        the plateau metric.

        Official ``Solver._run_one_epoch_mossformergan_se_16k`` adds the loss
        only when it is not None, then returns ``total_loss / (i+1)``. A
        missing loss is not written as zero. The batch stays in the
        denominator. This report keeps that batch out of the numeric values
        and marks it for the denominator. ``reduce_validation`` divides the
        included sum by every non-padding batch. A non-finite generator loss
        is the same exclusion. It does not raise, so the next rank can still
        enter the following forward.
        """
        alignment = bool(batch.get("validation_alignment_pad"))
        clean = batch["clean"].to(self.device)
        noisy = batch["noisy"].to(self.device)
        generator = unwrap(self.generator)
        discriminator = unwrap(self.discriminator)
        generator.eval()
        discriminator.train()
        included = False
        reason = "pesq failed; batch stays in the official denominator and is not a loss of 0"
        terms: dict[str, torch.Tensor] | None = None
        with torch.no_grad():
            terms = self._generator_terms(generator, discriminator, clean, noisy)
            finite = bool(torch.isfinite(terms["generator"]).all())
            targets = metric_targets(terms["teacher"], terms["audio"], self.sample_rate) if finite else None
        if alignment:
            included = False
            reason = "alignment forward keeps SyncBatchNorm in step and is outside the denominator"
        elif not finite:
            included = False
            reason = "non-finite generator loss; official validation leaves it out of the sum and keeps the batch in the denominator"
        elif targets is None:
            included = False
            reason = "pesq failed; batch stays in the official denominator and is not a loss of 0"
        else:
            included = True
            reason = "official generator loss; pesq gate included this batch"
        assert terms is not None
        if included:
            generator_metric = _metric(float(terms["generator"].detach()), reason=reason)
        else:
            generator_metric = _metric(status="na", reason=reason)
        losses = {
            "generator": generator_metric,
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
            "pesq_gate": "included" if included else "denominator_only",
            "generator_loss_included": included and not alignment,
            "alignment_pad": alignment,
        }

    def reduce_validation(self, reports: list[dict[str, Any]], *, collective: bool | None = None) -> dict[str, Any]:
        """Official validation loss: the mean of the per-rank means.

        Each rank's mean divides by its own batch count, including PESQ
        failures and official sampler padding, and excluding alignment pads.
        Ranks are then averaged with equal weight. Equal batch counts make
        this the same number as one global quotient. Unequal counts keep the
        official unweighted mean of means.
        """
        local = local_validation_totals(reports)
        rows = [local]
        if collective is None:
            collective = bool(self.validation_requires_all_ranks)
        if (
            collective
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        ):
            payload = torch.tensor(
                [local["sum"], local["batches"], local["failed"], local["included"], local["alignment"], local["mean"]],
                dtype=torch.float64,
            )
            if self.device.type == "cuda":
                payload = payload.to(self.device)
            gathered = [torch.empty_like(payload) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, payload)
            names = ("sum", "batches", "failed", "included", "alignment", "mean")
            rows = [dict(zip(names, [float(value) for value in item.tolist()], strict=True)) for item in gathered]
        active = [row for row in rows if row["batches"] > 0]
        if not active:
            raise TrainingError("validation produced no batches")
        official = official_validation_mean([row["mean"] for row in active])
        failed = int(sum(row["failed"] for row in rows))
        included = int(sum(row["included"] for row in rows))
        batches = int(sum(row["batches"] for row in rows))
        alignment = int(sum(row["alignment"] for row in rows))
        per_rank_batches = [int(row["batches"]) for row in rows]
        return {
            "losses": {
                "generator": {
                    "status": "ok",
                    "value": official,
                    "reason": (
                        "mean of per-rank means; failed PESQ batches stay in each rank's "
                        "denominator and are not stored as loss 0"
                    ),
                    "count": batches,
                }
            },
            "pesq": {
                "status": "na",
                "value": None,
                "reason": "pesq gates inclusion and is not the plateau metric",
                "count": 0,
                "failed_batches": failed,
                "included_batches": included,
            },
            "stoi": {"status": "na", "value": None, "reason": "not computed", "count": 0},
            "validation_coverage": {
                "batches": batches,
                "included_batches": included,
                "pesq_failed_batches": failed,
                "alignment_pad_forwards": alignment,
                "per_rank_batches": per_rank_batches,
                "denominator": "per-rank batch count, then unweighted mean across ranks",
            },
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


def _component(value: torch.Tensor, included: bool) -> dict[str, Any]:
    if not included:
        return _metric(status="na", reason="batch excluded by the pesq gate")
    return _metric(float(value.detach()))


def forward_spectra(view, module, noisy, clean):
    """Spectra for one MossFormerGAN forward. ``view`` supplies STFT sizes."""

    return MossFormerGANTask._forward(view, module, noisy, clean)


def generator_terms(view, module, discriminator, clean, noisy):
    """Official generator terms. ``view`` supplies STFT sizes and ``_forward``."""

    return MossFormerGANTask._generator_terms(view, module, discriminator, clean, noisy)


def validation_report(view, batch):
    """Official validation report for one cropped batch."""

    return MossFormerGANTask.validation_report(view, batch)


def reduce_validation(view, reports, *, collective=None):
    """Mean of per-rank validation means."""

    return MossFormerGANTask.reduce_validation(view, reports, collective=collective)


def _metric(value: float | None = None, *, status: str = "ok", reason: str | None = None) -> dict[str, Any]:
    if status == "ok" and (value is None or value != value or value in (float("inf"), float("-inf"))):
        return {"status": "na", "value": None, "reason": "non-finite loss"}
    return {"status": status, "value": value, "reason": reason}
