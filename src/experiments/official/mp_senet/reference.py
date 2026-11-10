"""MP-SENet generator and metric-discriminator updates.

The generator module is the inference implementation from ``src.se``. The
discriminator and the losses stay in this package. ``enhance()`` is not called
during training. ``loss: reconstruction`` is a time L1 objective for foundation
tests. It is not the official MP-SENet objective.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from src.se import build_model
from src.experiments.calc.distributed import ProcessInfo, unwrap, wrap_module
from src.experiments.calc.rng import seed_model_init
from src.experiments.calc.scope import accumulation_scope
from src.experiments.calc.errors import TrainingError
from src.experiments.official.mp_senet.discriminator import MetricDiscriminator
from src.experiments.calc.losses import discriminator_loss, generator_losses, metric_targets
from src.experiments.calc.stft import mag_pha_istft, mag_pha_stft


def validation_model(config: dict[str, Any], device: torch.device) -> "MPSENetTask":
    """Reference MP-SENet modules used to score validation on a weight copy."""

    return MPSENetTask(config, device)


class MPSENetTask:
    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        self.config = config
        self.device = device
        self.loss_name = config["loss"]
        model = config["model"]
        self.spec = model
        if self.loss_name not in ("official", "reconstruction"):
            raise TrainingError(f"unknown loss {self.loss_name!r}")
        if int(model["dense_channel"]) % 4 != 0:
            raise TrainingError("model.dense_channel must be divisible by the 4 attention heads")
        with seed_model_init(config["seed"]):
            self.generator = build_model("mp_senet", model).to(device)
            self.discriminator = MetricDiscriminator().to(device) if self.loss_name == "official" else None
        optimizer = config["optimizer"]
        betas = (float(optimizer["betas"][0]), float(optimizer["betas"][1]))
        self.optim_g = torch.optim.AdamW(
            self.generator.parameters(),
            lr=float(optimizer["lr"]),
            betas=betas,
            weight_decay=float(optimizer["weight_decay"]),
        )
        self.sched_g = torch.optim.lr_scheduler.ExponentialLR(
            self.optim_g, gamma=float(config["scheduler"]["gamma"])
        )
        self.optim_d = None
        self.sched_d = None
        if self.discriminator is not None:
            self.optim_d = torch.optim.AdamW(
                self.discriminator.parameters(),
                lr=float(optimizer["lr"]),
                betas=betas,
                weight_decay=float(optimizer["weight_decay"]),
            )
            self.sched_d = torch.optim.lr_scheduler.ExponentialLR(
                self.optim_d, gamma=float(config["scheduler"]["gamma"])
            )

    def distribute(self, process: ProcessInfo) -> None:
        self.generator = wrap_module(self.generator, process)
        if self.discriminator is not None:
            self.discriminator = wrap_module(self.discriminator, process)

    def train_mode(self) -> None:
        self.generator.train()
        if self.discriminator is not None:
            self.discriminator.train()

    def eval_mode(self) -> None:
        unwrap(self.generator).eval()
        if self.discriminator is not None:
            unwrap(self.discriminator).eval()

    def learning_rates(self) -> dict[str, float]:
        rates = {"generator": float(self.optim_g.param_groups[0]["lr"])}
        if self.optim_d is not None:
            rates["discriminator"] = float(self.optim_d.param_groups[0]["lr"])
        return rates

    def step_schedulers(self, validation: dict[str, Any] | None = None) -> None:
        del validation
        self.sched_g.step()
        if self.sched_d is not None:
            self.sched_d.step()

    def export_for_inference(self) -> dict[str, Any]:
        return {"generator": self.generator_state_dict()}

    def generator_state_dict(self) -> dict[str, torch.Tensor]:
        return unwrap(self.generator).state_dict()

    def state_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss_name,
            "generator": unwrap(self.generator).state_dict(),
            "discriminator": None
            if self.discriminator is None
            else unwrap(self.discriminator).state_dict(),
            "optim_g": self.optim_g.state_dict(),
            "optim_d": None if self.optim_d is None else self.optim_d.state_dict(),
            "sched_g": self.sched_g.state_dict(),
            "sched_d": None if self.sched_d is None else self.sched_d.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["loss"] != self.loss_name:
            raise TrainingError("checkpoint loss does not match the task")
        if isinstance(self.generator, DistributedDataParallel):
            raise TrainingError("load a checkpoint before wrapping DistributedDataParallel")
        self.generator.load_state_dict(state["generator"], strict=True)
        self.optim_g.load_state_dict(state["optim_g"])
        self.sched_g.load_state_dict(state["sched_g"])
        if self.discriminator is None:
            if state["discriminator"] is not None:
                raise TrainingError("checkpoint has a discriminator but this run does not")
            return
        self.discriminator.load_state_dict(state["discriminator"], strict=True)
        self.optim_d.load_state_dict(state["optim_d"])
        self.sched_d.load_state_dict(state["sched_d"])

    def train_update(self, batches: list[dict[str, Any]], accumulation: int) -> dict[str, float]:
        self.train_mode()
        if self.loss_name == "reconstruction":
            return self._reconstruction(batches, accumulation)
        return self._official(batches, accumulation)

    def validation_metrics(self, batch: dict[str, Any]) -> dict[str, float]:
        clean, noisy = self._waveforms(batch)
        with torch.no_grad():
            parts = self._generate(clean, noisy, generator=unwrap(self.generator))
        metrics = {
            "time": float(F.l1_loss(clean, parts["audio"]).detach()),
            "magnitude": float(F.mse_loss(parts["clean_mag"], parts["mag"]).detach()),
        }
        if self.loss_name == "official":
            targets = metric_targets(clean, parts["audio"], int(self.spec["sampling_rate"]))
            if targets is not None:
                metrics["pesq_normalized"] = float(targets.mean())
        return metrics

    def validation_report(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Losses and reference metrics for one validation batch.

        The caller has already switched to eval mode and disabled gradients.
        The generator receives the full waveform. Official ``center=True``
        iSTFT may come back at the largest hop multiple that does not exceed
        that length. This method does not trim the model input.
        ``align_validation_length`` remains available for the old truncation
        length. Training updates still reject an iSTFT length mismatch.

        Magnitude, phase, complex, and consistency spectrograms are the
        full-length STFT pair used by official validation. Time-domain L1
        uses only the overlapping prefix and records both lengths.

        ``loss == "official"`` puts PESQ failures in the mean as -1, matching
        ``eval_pesq``. The research path still drops ``na`` and errors.
        A missing PESQ library is ``unavailable`` on both paths, not -1 and
        not 0.
        """
        clean, noisy = self._waveforms(batch)
        with torch.no_grad():
            parts = self._generate(
                clean,
                noisy,
                generator=unwrap(self.generator),
                allow_official_istft_length=True,
            )
        audio = parts["audio"]
        overlap = min(int(clean.shape[-1]), int(audio.shape[-1]))
        if overlap <= 0:
            time_metric = _metric(status="na", reason="iSTFT audio and reference have no overlap")
            time_value = None
        else:
            time_value = float(F.l1_loss(clean[..., :overlap], audio[..., :overlap]).detach())
            time_metric = _metric(time_value)
        magnitude = float(F.mse_loss(parts["clean_mag"], parts["mag"]).detach())
        losses: dict[str, dict[str, Any]] = {
            "time": time_metric,
            "magnitude": _metric(magnitude),
        }
        generator_value = time_value
        if (
            self.loss_name == "official"
            and self.discriminator is not None
            and parts.get("mag_hat") is not None
            and overlap > 0
        ):
            try:
                _require_discriminator_resolution(parts["clean_mag"].shape[-2], parts["clean_mag"].shape[-1])
                ones = torch.ones(clean.shape[0], device=self.device)
                raw = unwrap(self.discriminator)
                metric = raw(parts["clean_mag"], parts["mag_hat"])
                detailed = generator_losses(
                    parts["clean_mag"],
                    parts["mag"],
                    parts["clean_phase"],
                    parts["phase"],
                    parts["clean_complex"],
                    parts["complex"],
                    parts["recomputed_complex"],
                    clean[..., :overlap],
                    audio[..., :overlap],
                    metric,
                    ones,
                )
                generator_value = float(detailed["total"].detach())
                for name in ("phase", "complex", "consistency", "metric"):
                    losses[name] = _metric(float(detailed[name].detach()))
                real = raw(parts["clean_mag"], parts["clean_mag"])
                fake = raw(parts["clean_mag"], parts["mag_hat"])
                targets = metric_targets(clean, audio, int(self.spec["sampling_rate"]))
                losses["discriminator"] = _metric(float(discriminator_loss(real, fake, ones, targets).detach()))
            except (TrainingError, RuntimeError, ValueError) as exc:
                losses["discriminator"] = _metric(status="na", reason=str(exc))
        elif self.loss_name == "official" and self.discriminator is not None:
            losses["discriminator"] = _metric(
                status="na",
                reason="iSTFT audio is too short to form the consistency spectrogram",
            )
        else:
            losses["discriminator"] = _metric(
                status="na",
                reason="reconstruction loss has no discriminator",
            )
        if generator_value is None:
            losses["generator"] = _metric(status="na", reason="iSTFT audio and reference have no overlap")
        else:
            losses["generator"] = _metric(generator_value)
        rate = int(self.spec["sampling_rate"])
        pesq = _validation_pesq(clean, audio, rate, official=self.loss_name == "official")
        stoi_scores = [
            _reference_score(row_clean, row_audio, rate, kind="stoi")
            for row_clean, row_audio in _rows(clean, audio)
        ]
        return {
            "losses": losses,
            "pesq": pesq,
            "stoi": _combine_scores(stoi_scores),
            "audio_lengths": _audio_length_rows(clean, audio),
            "pesq_failures": pesq.get("pesq_failures"),
            "pesq_scored": pesq.get("pesq_scored"),
            "pesq_policy": pesq.get("pesq_policy"),
        }

    def _reconstruction(self, batches: list[dict[str, Any]], accumulation: int) -> dict[str, float]:
        self.optim_g.zero_grad(set_to_none=True)
        total = 0.0
        for index, batch in enumerate(batches):
            with accumulation_scope([self.generator], last=index == len(batches) - 1):
                clean, noisy = self._waveforms(batch)
                parts = self._generate(clean, noisy, generator=self.generator)
                loss = F.l1_loss(clean, parts["audio"]) / accumulation
                loss.backward()
            total += float(loss.detach()) * accumulation
        self.optim_g.step()
        mean = total / len(batches)
        return {"generator": mean, "time": mean}

    def _official(self, batches: list[dict[str, Any]], accumulation: int) -> dict[str, float]:
        if self.discriminator is None or self.optim_d is None:
            raise TrainingError("official loss requires a discriminator")
        raw_discriminator = unwrap(self.discriminator)
        # One generator graph per micro-batch, matching official train.py.
        # The discriminator sees a detached magnitude. A second forward would
        # be a different CUDA evaluation and would feed the generator loss a
        # waveform the discriminator did not score.
        self.optim_d.zero_grad(set_to_none=True)
        self.optim_g.zero_grad(set_to_none=True)
        discriminator_total = 0.0
        stored: list[tuple[torch.Tensor, dict[str, torch.Tensor]]] = []
        for index, batch in enumerate(batches):
            clean, noisy = self._waveforms(batch)
            last = index == len(batches) - 1
            # The generator graphs are retained until after the discriminator step.
            # Reduce explicitly after all delayed backward calls: multiple DDP
            # forwards before backward cannot safely select the last reduction.
            with accumulation_scope([self.generator], last=False):
                parts = self._generate(clean, noisy, generator=self.generator)
            stored.append((clean, parts))
            ones = torch.ones(clean.shape[0], device=self.device)
            _require_discriminator_resolution(parts["clean_mag"].shape[-2], parts["clean_mag"].shape[-1])
            with accumulation_scope([self.discriminator], last=last):
                real = self.discriminator(parts["clean_mag"], parts["clean_mag"])
                fake = self.discriminator(parts["clean_mag"], parts["mag_hat"].detach())
                targets = metric_targets(clean, parts["audio"], int(self.spec["sampling_rate"]))
                loss = discriminator_loss(real, fake, ones, targets) / accumulation
                loss.backward()
            discriminator_total += float(loss.detach()) * accumulation
        self.optim_d.step()

        frozen = [parameter for parameter in raw_discriminator.parameters() if parameter.requires_grad]
        for parameter in frozen:
            parameter.requires_grad_(False)
        component_totals = {
            name: 0.0 for name in ("total", "magnitude", "phase", "complex", "consistency", "metric", "time")
        }
        try:
            for clean, parts in stored:
                ones = torch.ones(clean.shape[0], device=self.device)
                # The unwrapped discriminator avoids a second DDP all-reduce.
                # Its parameters are frozen, so the metric gradient reaches the generator only.
                metric = raw_discriminator(parts["clean_mag"], parts["mag_hat"])
                losses = generator_losses(
                    parts["clean_mag"],
                    parts["mag"],
                    parts["clean_phase"],
                    parts["phase"],
                    parts["clean_complex"],
                    parts["complex"],
                    parts["recomputed_complex"],
                    clean,
                    parts["audio"],
                    metric,
                    ones,
                )
                loss = losses["total"] / accumulation
                loss.backward()
                for name, value in losses.items():
                    component_totals[name] += float(value.detach())
        finally:
            for parameter in frozen:
                parameter.requires_grad_(True)
        if isinstance(self.generator, DistributedDataParallel):
            import torch.distributed as dist
            for parameter in unwrap(self.generator).parameters():
                if parameter.grad is not None:
                    dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                    parameter.grad.div_(dist.get_world_size())
        self.optim_g.step()
        count = len(batches)
        return {
            "generator": component_totals["total"] / count,
            "discriminator": discriminator_total / count,
            "magnitude": component_totals["magnitude"] / count,
            "phase": component_totals["phase"] / count,
            "complex": component_totals["complex"] / count,
            "consistency": component_totals["consistency"] / count,
            "metric": component_totals["metric"] / count,
            "time": component_totals["time"] / count,
        }

    def _waveforms(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        clean = batch["clean"].to(self.device)
        noisy = batch["noisy"].to(self.device)
        if clean.ndim != 2 or clean.shape != noisy.shape:
            raise TrainingError(f"expected matching [batch, time] waveforms, got {tuple(clean.shape)}")
        return clean, noisy

    def _generate(
        self,
        clean: torch.Tensor,
        noisy: torch.Tensor,
        *,
        generator: torch.nn.Module,
        allow_official_istft_length: bool = False,
    ) -> dict[str, torch.Tensor]:
        spec = self.spec
        hop = int(spec["hop_size"])
        clean_mag, clean_phase, clean_complex = mag_pha_stft(
            clean, spec["n_fft"], hop, spec["win_size"], spec["compress_factor"]
        )
        noisy_mag, noisy_phase, _noisy_complex = mag_pha_stft(
            noisy, spec["n_fft"], hop, spec["win_size"], spec["compress_factor"]
        )
        magnitude, phase, complex_spectrum = generator(noisy_mag, noisy_phase)
        audio = mag_pha_istft(magnitude, phase, spec["n_fft"], hop, spec["win_size"], spec["compress_factor"])
        waveform_length = int(clean.shape[-1])
        audio_length = int(audio.shape[-1])
        if audio_length != waveform_length:
            official_length = official_center_istft_length(waveform_length, hop)
            if not (allow_official_istft_length and audio_length == official_length):
                raise TrainingError(
                    f"iSTFT length {audio_length} does not match the waveform length {waveform_length}"
                )
        try:
            magnitude_hat, phase_hat, recomputed = mag_pha_stft(
                audio, spec["n_fft"], hop, spec["win_size"], spec["compress_factor"]
            )
        except (RuntimeError, TrainingError):
            if audio_length == waveform_length or not allow_official_istft_length:
                raise
            magnitude_hat = None
        else:
            magnitude_hat = (magnitude_hat, phase_hat, recomputed)
        result = {
            "clean_mag": clean_mag,
            "clean_phase": clean_phase,
            "clean_complex": clean_complex,
            "mag": magnitude,
            "phase": phase,
            "complex": complex_spectrum,
            "audio": audio,
        }
        if magnitude_hat is not None:
            hat, phase_hat, recomputed = magnitude_hat
            result["mag_hat"] = hat
            result["phase_hat"] = phase_hat
            result["recomputed_complex"] = recomputed
        return result


def official_center_istft_length(length: int, hop: int) -> int:
    """Samples restored by official ``center=True`` iSTFT for this input length.

    With ``center=True``, the frame count is ``1 + length // hop`` and iSTFT
    without an explicit length returns ``(frames - 1) * hop``. That is the
    largest multiple of ``hop`` that does not exceed ``length``.
    """
    if hop < 1:
        raise TrainingError("hop must be positive")
    if length < 0:
        raise TrainingError("length must be non-negative")
    return (int(length) // int(hop)) * int(hop)


def align_validation_length(length: int, hop: int, n_fft: int) -> int:
    """Old validation truncation: largest hop multiple at most ``length``.

    ``validation_report`` does not call this to cut the generator input.
    The update path still requires the waveform and iSTFT lengths to match.
    Lengths whose hop multiple is at most ``n_fft // 2`` are rejected here
    because the old path refused to build that shorter STFT.
    """
    if hop < 1:
        raise TrainingError("hop must be positive")
    if length % hop == 0:
        return length
    aligned = (length // hop) * hop
    if aligned <= n_fft // 2:
        raise TrainingError(f"utterance length {length} is too short to align to hop {hop}")
    return aligned


def _conv_out(size: int) -> int:
    return (size + 2 - 4) // 2 + 1


def _require_discriminator_resolution(n_freq: int, n_time: int) -> None:
    """InstanceNorm rejects a 1x1 map while the discriminator is training.

    Four stride-2 convolutions shrink the spectrogram before that norm. The
    official 400-point STFT and 2-second segments stay large enough. A smaller
    setup is rejected instead of switching the norm to evaluation statistics.
    """
    freq, time = int(n_freq), int(n_time)
    for _ in range(4):
        freq, time = _conv_out(freq), _conv_out(time)
        if freq < 1 or time < 1 or freq * time <= 1:
            raise TrainingError(
                "official loss needs a larger spectrogram so the metric discriminator "
                "InstanceNorm sees more than one spatial location. Increase segment_size or n_fft."
            )


def _metric(value: float | None = None, *, status: str = "ok", reason: str | None = None) -> dict[str, Any]:
    if status == "ok" and (value is None or value != value or value in (float("inf"), float("-inf"))):
        return {"status": "na", "value": None, "reason": "non-finite loss"}
    return {"status": status, "value": value, "reason": reason}


def _rows(clean: torch.Tensor, audio: torch.Tensor):
    return zip(clean.detach().cpu(), audio.detach().cpu(), strict=True)


def _reference_score(reference: torch.Tensor, estimate: torch.Tensor, sample_rate: int, *, kind: str) -> dict[str, Any]:
    if kind == "pesq":
        from src.evaluation.metrics.pesq import score_pesq

        result = score_pesq(reference, estimate, sample_rate)
    else:
        from src.evaluation.metrics.stoi import score_stoi

        result = score_stoi(reference, estimate, sample_rate, extended=False)
    status = result["status"]
    if status == "ok":
        return _metric(float(result["value"]))
    if status == "unavailable":
        return _metric(status="unavailable", reason=str(result["reason"]))
    return _metric(status="na", reason=str(result.get("reason") or status))


def _combine_scores(scores: list[dict[str, Any]]) -> dict[str, Any]:
    if any(item["status"] == "unavailable" for item in scores):
        reason = next(item["reason"] for item in scores if item["status"] == "unavailable")
        return {"status": "unavailable", "value": None, "reason": reason, "count": 0}
    values = [float(item["value"]) for item in scores if item["status"] == "ok" and item["value"] is not None]
    if values:
        return {"status": "ok", "value": sum(values) / len(values), "reason": None, "count": len(values)}
    reason = next((item["reason"] for item in scores if item["reason"]), "no score")
    return {"status": "na", "value": None, "reason": reason, "count": 0}


OFFICIAL_PESQ_POLICY = "official_include_negative_one"
RESEARCH_PESQ_POLICY = "research_exclude_na"


def _audio_length_rows(clean: torch.Tensor, audio: torch.Tensor) -> list[dict[str, int]]:
    model_input = int(clean.shape[-1])
    produced = int(audio.shape[-1])
    overlap = min(model_input, produced)
    return [
        {
            "model_input": model_input,
            "clean": model_input,
            "audio": produced,
            "pesq_reference": model_input,
            "pesq_estimate": produced,
            "time_overlap": overlap,
        }
        for _index in range(int(clean.shape[0]))
    ]


def _pesq_unavailable(reason: str, policy: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "value": None,
        "reason": reason,
        "count": 0,
        "pesq_failures": None,
        "pesq_scored": 0,
        "pesq_policy": policy,
    }


def _waveform_numpy(waveform: torch.Tensor) -> Any:
    return waveform.detach().to(dtype=torch.float32).contiguous().numpy()


def official_eval_pesq(clean_utt: Any, esti_utt: Any, sample_rate: int) -> float:
    """One utterance, matching official ``eval_pesq``.

    An exception, a non-finite value, or a negative library error code becomes
    -1. The caller keeps that -1 in the mean. ImportError is not converted:
    a missing library stays unavailable.
    """
    from pesq import PesqError, pesq

    try:
        score = pesq(int(sample_rate), clean_utt, esti_utt, on_error=PesqError.RETURN_VALUES)
    except Exception:
        return -1.0
    try:
        score = float(score)
    except (TypeError, ValueError):
        return -1.0
    if not math.isfinite(score) or score < 0.0:
        return -1.0
    return score


def combine_official_pesq_values(values: list[float]) -> dict[str, Any]:
    """``np.mean`` of official per-utterance scores, including -1."""
    if not values:
        return {
            "status": "na",
            "value": None,
            "reason": "no score",
            "count": 0,
            "pesq_failures": 0,
            "pesq_scored": 0,
            "pesq_policy": OFFICIAL_PESQ_POLICY,
        }
    failures = sum(1 for value in values if not math.isfinite(float(value)) or float(value) < 0.0)
    mean = float(sum(float(value) for value in values)) / float(len(values))
    return {
        "status": "ok",
        "value": mean,
        "reason": None,
        "count": len(values),
        "pesq_failures": failures,
        "pesq_scored": len(values) - failures,
        "pesq_policy": OFFICIAL_PESQ_POLICY,
    }


def _annotate_research_pesq(scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Research mean: drop ``na`` and errors. Do not rename this into the official mean."""
    summary = dict(_combine_scores(scores))
    summary["pesq_policy"] = RESEARCH_PESQ_POLICY
    if summary["status"] == "unavailable":
        summary["pesq_failures"] = None
        summary["pesq_scored"] = 0
        return summary
    summary["pesq_scored"] = sum(1 for item in scores if item["status"] == "ok" and item["value"] is not None)
    summary["pesq_failures"] = sum(1 for item in scores if item["status"] != "ok")
    return summary


def _validation_pesq(
    clean: torch.Tensor,
    audio: torch.Tensor,
    sample_rate: int,
    *,
    official: bool,
) -> dict[str, Any]:
    pairs = list(_rows(clean, audio))
    if not official:
        scores = [_reference_score(reference, estimate, sample_rate, kind="pesq") for reference, estimate in pairs]
        return _annotate_research_pesq(scores)
    from src.evaluation.metrics.pesq import library_status

    missing = library_status()
    if missing:
        return _pesq_unavailable(str(missing), OFFICIAL_PESQ_POLICY)
    values: list[float] = []
    try:
        for reference, estimate in pairs:
            values.append(
                official_eval_pesq(_waveform_numpy(reference), _waveform_numpy(estimate), sample_rate)
            )
    except ImportError as exc:
        return _pesq_unavailable(str(exc), OFFICIAL_PESQ_POLICY)
    return combine_official_pesq_values(values)
