"""DeepFilterNet3 loss, schedule, and a short speech run versus the official modules.

Tolerances, fixed before reading measurements:

- multi-resolution spectral loss and local SNR: atol 1e-5
- AdamW update under the shipped per-step schedule: weight atol 1e-4
  when the gradient cosine of large tensors is >= 0.999

The 16 kHz smoke waveforms are converted with
``torchaudio.functional.resample`` (sinc, default rolloff and lowpass) to
48 kHz before either model sees them. Official training hears native 48 kHz.
``libdf`` is not installed, so the official ``DfNet`` class is constructed
only when its Python module imports. The ERB/STFT front-end inside ``libdf``
is not executed.

steps_per_epoch is 8 and epochs is 120, matching the smoke set at batch 1
and the shipped ``max_epochs``. Only the first 20 schedule values are applied.

    CUDA_VISIBLE_DEVICES=<free> CUBLAS_WORKSPACE_CONFIG=:4096:8 \\
        .venvs/lab/bin/python tests/training/reference/compare_deepfilternet.py
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import torch
import torchaudio.functional as AF

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.se import build_model, load_enhancer  # noqa: E402
from src.training.se.deepfilternet import DeepFilterNetTask, deepfilter_cosine_schedules  # noqa: E402
from tests.training.reference.common import (  # noqa: E402
    environment,
    gap,
    grad_report,
    load_cached,
    optim_report,
    reference_scores,
    save_report,
    weight_report,
)

LOSS_ATOL = 1e-5
GRAD_ATOL = 1e-4
CONFIG_INI = ROOT / "data" / "checkpoints" / "deepfilternet3" / "DeepFilterNet3" / "config.ini"
STEPS_PER_EPOCH = 8
EPOCHS = 120


def _install_official():
    package = ROOT / "src" / "se" / "official" / "deepfilternet" / "DeepFilterNet" / "df"
    df = types.ModuleType("df")
    df.__path__ = [str(package)]
    df.__package__ = "df"
    sys.modules["df"] = df
    libdf = types.ModuleType("libdf")
    libdf.unit_norm_init = lambda *args, **kwargs: None
    libdf.DF = type("DF", (), {})
    sys.modules["libdf"] = libdf
    loguru = types.ModuleType("loguru")

    class _Level:
        no = 30

    class _Logger:
        def level(self, name):
            return _Level()

        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    loguru.logger = _Logger()
    sys.modules["loguru"] = loguru
    sys.path.insert(0, str(package.parent))
    from df.config import config
    from df.loss import LocalSnrLoss, MultiResSpecLoss
    from df.lr import cosine_scheduler
    from df.modules import local_snr

    config.load(str(CONFIG_INI))
    return MultiResSpecLoss, LocalSnrLoss, local_snr, cosine_scheduler


def _schedule_config() -> dict:
    return {
        "optimizer": {"name": "adamw", "lr": 1e-3, "betas": [0.9, 0.999], "weight_decay": 1e-12, "amsgrad": True},
        "scheduler": {
            "name": "deepfilter_cosine",
            "steps_per_epoch": STEPS_PER_EPOCH,
            "epochs": EPOCHS,
            "lr_min": 1e-6,
            "lr_warmup": 1e-4,
            "warmup_epochs": 3,
            "weight_decay_end": 0.01,
        },
    }


def _task_config() -> dict:
    schedule = _schedule_config()
    return {
        "seed": 7,
        "loss": "official",
        "grad_clip": 1.0,
        "optimizer": schedule["optimizer"],
        "scheduler": schedule["scheduler"],
        "model": {"name": "deepfilternet3", "sampling_rate": 48000, "architecture": {}},
    }


def _resample(waveform: torch.Tensor) -> torch.Tensor:
    return AF.resample(waveform.cpu(), 16000, 48000).to(dtype=torch.float32)


def _official_total(mr, local_loss, enhanced_td, clean_td, predicted, target) -> torch.Tensor:
    spectral = mr(enhanced_td, clean_td)
    snr = local_loss(predicted, target)
    return spectral + snr


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MultiResSpecLoss, LocalSnrLoss, local_snr, cosine_scheduler = _install_official()
    ours_lr, ours_wd = deepfilter_cosine_schedules(_schedule_config())
    official_lr = cosine_scheduler(
        1e-3, 1e-6, epochs=EPOCHS, niter_per_ep=STEPS_PER_EPOCH, warmup_epochs=3, start_warmup_value=1e-4
    )
    official_wd = cosine_scheduler(1e-12, 0.01, epochs=EPOCHS, niter_per_ep=STEPS_PER_EPOCH)
    report = {
        "environment": environment(device),
        "official_commit": "d375b2d8309e0935d165700c91da9de862a99c31",
        "config_sha256": "415eb925d44990d938fb739f514aa3662c1ec0ea836cff044fa1291b82cb4290",
        "resample": "torchaudio.functional.resample 16000 -> 48000, default sinc lowpass",
        "steps_per_epoch": STEPS_PER_EPOCH,
        "epochs": EPOCHS,
        "grad_clip": 1.0,
        "optimizer": _schedule_config()["optimizer"],
        "tolerances": {"loss_atol": LOSS_ATOL, "grad_atol": GRAD_ATOL, "large_cosine_min": 0.999},
        "schedule_gap": {
            "lr_max_abs": max(abs(float(left) - float(right)) for left, right in zip(official_lr[:20], ours_lr[:20], strict=False)),
            "wd_max_abs": max(abs(float(left) - float(right)) for left, right in zip(official_wd[:20], ours_wd[:20], strict=False)),
            "lr0": float(official_lr[0]),
            "lr_end_warmup": float(official_lr[3 * STEPS_PER_EPOCH - 1]),
            "wd0": float(official_wd[0]),
        },
        "libdf_frontend": "NOT TESTED. libdf is not installed; ERB and the official analysis STFT were not executed.",
    }
    mr = MultiResSpecLoss([256, 512, 1024, 2048], 0.3, 500.0, 500.0).to(device)
    local = LocalSnrLoss(1e-3).to(device)
    torch.manual_seed(5)
    wave_a = torch.randn(1, 4800, device=device)
    wave_b = torch.randn(1, 4800, device=device)
    from src.training.se.deepfilternet import _multi_resolution_spectral

    report["spectral_identity"] = abs(float(mr(wave_a, wave_b).detach()) - float(_multi_resolution_spectral(wave_a, wave_b).detach()))
    save_report("deepfilternet3.json", report)
    if report["spectral_identity"] > 1e-4 or report["schedule_gap"]["lr_max_abs"] > 1e-8:
        raise SystemExit(f"loss or schedule identity failed: {report['spectral_identity']} {report['schedule_gap']}")
    task = DeepFilterNetTask(_task_config(), device)
    other = build_model("deepfilternet3", None).to(device)
    other.load_state_dict(task.generator.state_dict(), strict=True)
    report["parameters"] = sum(parameter.numel() for parameter in other.parameters())
    report["official_dfnet"] = _try_official_dfnet(other)
    train = load_cached("df_train")
    valid = load_cached("valid")
    report["train_case_ids"] = train["case_id"]
    report["valid_case_id"] = valid["case_id"]
    optim_other = torch.optim.AdamW(other.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-12, amsgrad=True)
    history = []
    for step in range(20):
        index = step % train["clean"].shape[0]
        case_id = train["case_id"][index]
        clean = _resample(train["clean"][index : index + 1]).to(device)
        noisy = _resample(train["noisy"][index : index + 1]).to(device)
        for group in task.optim_g.param_groups:
            group["lr"] = float(ours_lr[step])
            group["weight_decay"] = float(ours_wd[step])
        # train_update applies the same schedule from its own counter.
        metrics = task.train_update([{"clean": clean, "noisy": noisy}], 1)
        manual = _manual_step(other, optim_other, task, mr, local, local_snr, clean, noisy, float(ours_lr[step]), float(ours_wd[step]))
        record = {
            "step": step,
            "case_id": case_id,
            "lr": float(ours_lr[step]),
            "weight_decay": float(ours_wd[step]),
            "task_loss": metrics["generator"],
            "manual_loss": manual["loss"],
            "loss_gap": abs(metrics["generator"] - manual["loss"]),
            "spectral_gap": manual["spectral_gap"],
            "local_snr_gap": manual["local_gap"],
            "grad": manual["grad"],
            "weight": weight_report(other, task.generator),
            "optim": optim_report(optim_other, task.optim_g),
        }
        history.append({key: record[key] for key in ("step", "case_id", "lr", "task_loss", "manual_loss", "loss_gap", "spectral_gap", "local_snr_gap")})
        history[-1]["grad_max_abs"] = record["grad"]["max_abs"]
        history[-1]["large_cosine"] = record["grad"]["large_cosine_min"]
        history[-1]["weight_max_abs"] = record["weight"]["max_abs"]
        report["history"] = history
        report["last_step"] = record
        save_report("deepfilternet3.json", report)
        print(f"df step {step} loss {metrics['generator']:.6f} gap {record['loss_gap']:.6e}", flush=True)
    clean = _resample(valid["clean"][:, :8000]).to(device)
    noisy = _resample(valid["noisy"][:, :8000]).to(device)
    task.eval_mode()
    other.eval()
    with torch.no_grad():
        task_wave = _enhanced_wave(task, clean, noisy)
        other_wave = _manual_wave(other, task, clean, noisy)
    report["probe_case_id"] = valid["case_id"]
    report["probe"] = {
        "crop_16k_samples": 8000,
        "wave": gap(other_wave, task_wave),
        "task": reference_scores(clean[0].cpu(), task_wave[0].cpu(), 48000),
        "manual": reference_scores(clean[0].cpu(), other_wave[0].cpu(), 48000),
    }
    exported = ROOT / "data" / "results" / "training" / "official-compare-20261101" / "deepfilternet3_generator.pt"
    torch.save(task.generator_state_dict(), exported)
    enhancer = load_enhancer("deepfilternet3", exported, device=str(device))
    with torch.no_grad():
        reloaded = enhancer.enhance(valid["noisy"][0, :8000].cpu(), 16000)
    report["reload_16k"] = {
        "finite": bool(torch.isfinite(reloaded).all()),
        "samples": int(reloaded.numel()),
    }
    path = save_report("deepfilternet3.json", report)
    print(path)


def _try_official_dfnet(independent: torch.nn.Module) -> dict:
    try:
        from df.deepfilternet3 import DfNet

        erb = independent.erb_fb.detach().clone()
        inverse = independent.mask.erb_inv_fb.detach().clone()
        official = DfNet(erb, inverse)
        official.load_state_dict(independent.state_dict(), strict=True)
        return {"status": "constructed", "parameters": sum(parameter.numel() for parameter in official.parameters())}
    except Exception as exc:
        return {"status": "NOT TESTED", "reason": f"{type(exc).__name__}: {exc}"}


def _manual_step(module, optim, task, mr, local_loss, local_snr, clean, noisy, learning_rate, decay) -> dict:
    module.train()
    for group in optim.param_groups:
        group["lr"] = learning_rate
        if float(group["weight_decay"]) > 0.0:
            group["weight_decay"] = decay
    optim.zero_grad(set_to_none=True)
    loss, spectral_gap, local_gap = _manual_loss(module, task, mr, local_loss, local_snr, clean, noisy)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
    grad = grad_report(module, task.generator, atol=GRAD_ATOL)
    optim.step()
    return {"loss": float(loss.detach()), "spectral_gap": spectral_gap, "local_gap": local_gap, "grad": grad}


def _manual_loss(module, task, mr, local_loss, local_snr, clean, noisy):
    from src.training.se.deepfilternet import (
        _local_snr_db,
        _multi_resolution_spectral,
        _vorbis_istft,
    )
    from src.se.models.deepfilternet3.dsp import vorbis_window

    task.stream.reset()
    clean_spec, _clean_erb, _clean_feat, clean_complex = task._analyze_features(clean)
    task.stream.reset()
    noisy_spec, noisy_erb, noisy_feat, noisy_complex = task._analyze_features(noisy)
    enhanced, _mask, lsnr, _coefs = module(noisy_spec, noisy_erb, noisy_feat)
    enhanced_complex = torch.view_as_complex(enhanced.squeeze(1))
    clean_c = torch.view_as_complex(clean_spec.squeeze(1))
    noisy_c = noisy_complex
    frames = min(enhanced_complex.shape[-2], clean_c.shape[-2], noisy_c.shape[-2])
    bins = min(enhanced_complex.shape[-1], clean_c.shape[-1], noisy_c.shape[-1])
    enhanced_complex = enhanced_complex[..., :frames, :bins]
    clean_c = clean_c[..., :frames, :bins]
    noisy_c = noisy_c[..., :frames, :bins]
    params = module.params
    window = vorbis_window(params.fft_size).to(device=enhanced_complex.device, dtype=enhanced_complex.real.dtype)
    enhanced_wave = _vorbis_istft(enhanced_complex, params.fft_size, params.hop_size, window)
    clean_wave = _vorbis_istft(clean_c, params.fft_size, params.hop_size, window)
    spectral_official = mr(enhanced_wave, clean_wave)
    spectral_ours = _multi_resolution_spectral(enhanced_wave, clean_wave)
    target = _local_snr_db(
        clean_c,
        noisy_c - clean_c,
        sample_rate=params.sr,
        n_fft=params.fft_size,
        hop=params.hop_size,
        low=params.lsnr_min - 1,
        high=params.lsnr_max + 1,
    )
    predicted = lsnr.reshape(target.shape[0], -1)
    # Official local_snr expects [B, C, T, F] complex. Compare the target, then the MSE.
    window_frames = _window_frames(params)
    official_target = local_snr(
        clean_c.unsqueeze(1),
        (noisy_c - clean_c).unsqueeze(1),
        window_size=window_frames,
        db=True,
        window_size_ns=window_frames * 2,
    )[0].squeeze(1).clamp(params.lsnr_min - 1, params.lsnr_max + 1)
    local_ours = torch.nn.functional.mse_loss(predicted, target) * 1e-3
    local_official = local_loss(predicted.unsqueeze(-1), official_target)
    loss = spectral_official + local_official
    return (
        loss,
        abs(float(spectral_official.detach()) - float(spectral_ours.detach())),
        float((target.detach() - official_target.detach()).abs().max()),
    )


def _window_frames(params) -> int:
    fft_ms = params.fft_size / params.sr * 1000
    hop_ms = params.hop_size / params.sr * 1000
    windows = 1 + (20 - fft_ms) / hop_ms
    return max(int(round(windows)), 1)


def _enhanced_wave(task, clean, noisy):
    return _manual_wave(task.generator, task, clean, noisy)


def _manual_wave(module, task, clean, noisy):
    from src.training.se.deepfilternet import _vorbis_istft
    from src.se.models.deepfilternet3.dsp import vorbis_window

    task.stream.reset()
    noisy_spec, noisy_erb, noisy_feat, _noisy_complex = task._analyze_features(noisy)
    enhanced, _mask, _lsnr, _coefs = module(noisy_spec, noisy_erb, noisy_feat)
    enhanced_complex = torch.view_as_complex(enhanced.squeeze(1))
    params = module.params
    window = vorbis_window(params.fft_size).to(device=enhanced_complex.device, dtype=enhanced_complex.real.dtype)
    return _vorbis_istft(enhanced_complex, params.fft_size, params.hop_size, window)


if __name__ == "__main__":
    main()
