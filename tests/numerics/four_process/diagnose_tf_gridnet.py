"""Split the TF-GridNet official/independent GPU loss gap by operation.

The script does not overwrite an existing results directory. It writes one new
report and exits. Launch only after nvidia-smi shows a free GPU.
"""
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.experiments.calc.spectral import multi_resolution_l1, _rect_magnitude_l1
from tests.numerics.reference.compare_tf_gridnet import (
    ARCHITECTURE,
    WINDOWS,
    _official_loss_class,
    _official_value,
    build_model,
    reference_v1,
)


def gap(left, right):
    left = left.detach().float().cpu().reshape(-1)
    right = right.detach().float().cpu().reshape(-1)
    delta = (left - right).abs()
    return {
        "max_abs": float(delta.max()) if delta.numel() else None,
        "mean_abs": float(delta.mean()) if delta.numel() else None,
    }


def flags():
    return {
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "matmul_precision": torch.get_float32_matmul_precision(),
    }


def build(device):
    torch.manual_seed(3)
    official = reference_v1()(input_dim=0, n_srcs=1, use_builtin_complex=True, **ARCHITECTURE).to(device)
    independent = build_model("tf_gridnet", ARCHITECTURE, variant="ms-snsd").to(device)
    independent.load_state_dict(official.state_dict(), strict=True)
    criterion = _official_loss_class()(
        window_sz=list(WINDOWS), hop_sz=None, eps=1e-8, time_domain_weight=0.5,
        normalize_variance=False, reduction="sum",
    ).to(device)
    generator = torch.Generator().manual_seed(4)
    clean = torch.randn(1, 2048, generator=generator)
    noisy = clean * 0.5 + 0.05
    return official, independent, criterion, clean.to(device), noisy.to(device)


def components(criterion, wave):
    target = wave.detach()
    estimate = wave.detach()
    # Recompute the published branches on this waveform. scaling uses estimate==target here
    # only when the caller passes identical tensors; the training path passes distinct ones.
    return criterion, target, estimate


def loss_parts(criterion, clean, estimate):
    """Return official branch values and the independent branch values for one pair."""
    official = {}
    # Call the module, then reproduce branches with the same tensors.
    official["total"] = float(_official_value(criterion, clean, estimate).detach())
    estimate_f = estimate.float()
    target_f = clean.float()
    scale = torch.sum(estimate_f * target_f, -1, keepdim=True) / (torch.sum(estimate_f**2, -1, keepdim=True) + 1e-8)
    time = torch.sum((estimate_f * scale - target_f).abs(), dim=-1)
    official["time"] = float((time * 0.5).mean())
    spectral = []
    for encoder in criterion.stft_encoders:
        target_mag = criterion.get_magnitude(encoder(target_f)[0])
        estimate_mag = criterion.get_magnitude(encoder(estimate_f * scale)[0])
        spectral.append(float(torch.sum((estimate_mag - target_mag).abs(), dim=(1, 2)).mean()))
    official["spectral_windows"] = spectral
    independent = float(multi_resolution_l1(clean, estimate, WINDOWS, normalize_variance=False).detach())
    ind_windows = []
    for window in WINDOWS:
        ind_windows.append(float(_rect_magnitude_l1(target_f, estimate_f * scale, window, window // 2).mean()))
    return {"official_total": official["total"], "official_time": official["time"], "official_windows": spectral, "independent_total": independent, "independent_windows": ind_windows}


def stft_paths(wave):
    """Compare torch.stft(center=True) with the explicit reflect pad used by the independent loss."""
    from src.experiments.calc.stft import centered_stft
    window = 256
    hop = 128
    ones = torch.ones(window, device=wave.device, dtype=wave.dtype)
    manual = centered_stft(wave, window, hop, ones).abs()
    native = torch.stft(wave, window, hop_length=hop, win_length=window, window=None, center=True, return_complex=True).abs()
    native_ones = torch.stft(wave, window, hop_length=hop, win_length=window, window=ones, center=True, return_complex=True).abs()
    return {
        "manual_vs_center_none": gap(manual, native),
        "manual_vs_center_ones": gap(manual, native_ones),
        "center_none_vs_center_ones": gap(native, native_ones),
    }


def main():
    device_name = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    out = Path(sys.argv[2])
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    device = torch.device(device_name)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    official, independent, criterion, clean, noisy = build(device)
    official.train(); independent.train()
    lengths = torch.tensor([noisy.shape[-1]], device=device)
    with torch.no_grad():
        official_wave = official(noisy, lengths)[0][0]
        independent_wave = independent(noisy)
    report = {
        "device": device_name,
        "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "flags": flags(),
        "cublas": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "forward": gap(official_wave, independent_wave),
        "same_official_wave": loss_parts(criterion, clean, official_wave),
        "same_independent_wave": loss_parts(criterion, clean, independent_wave),
        "stft_256": stft_paths(official_wave),
    }
    # One controlled substitution: independent formula on a center=True magnitude path.
    report["dtype"] = str(official_wave.dtype)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
