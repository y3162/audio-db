"""Where a repeated Parakeet TDT loss starts to differ.

This is a real-model GPU test. It is not part of the CPU suite's work: hidden
CUDA is NOT_RUN, and a missing NeMo install is a skip. The research loss is
unchanged. Omega stays at the checkpoint value.

Run with the integrated environment and one free GPU:

    CUDA_VISIBLE_DEVICES=3 scripts/setup/with_env.sh \\
      .venvs/se/bin/python -m unittest tests.research.test_tdt_determinism -v
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_SPEECH = _ROOT / "data/corpora/VCTK/wav48/p225/p225_001.wav"
_MODEL = "nvidia/parakeet-tdt-0.6b-v2"


def _reason() -> str | None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return "NOT_RUN: CUDA_VISIBLE_DEVICES is unset"
    if visible.strip() in {"", "-1"}:
        return "NOT_RUN: CUDA is hidden"
    if not _SPEECH.is_file():
        return "smoke speech wav is absent"
    try:
        import nemo  # noqa: F401
        import numba  # noqa: F401
    except Exception:
        return "NeMo or Numba is not installed in this environment"
    return None


def _load_wave(seconds: float = 1.0):
    import soundfile as sf
    import torch
    import torchaudio

    audio, sample_rate = sf.read(_SPEECH, dtype="float32", always_2d=False)
    wave = torch.as_tensor(audio, dtype=torch.float32)
    if wave.ndim == 2:
        wave = wave.mean(dim=1)
    if int(sample_rate) != 16000:
        wave = torchaudio.functional.resample(wave, int(sample_rate), 16000)
    count = max(1, int(seconds * 16000))
    return wave[:count].contiguous()


def _finite_stats(left, right) -> dict:
    import torch

    same_shape = tuple(left.shape) == tuple(right.shape) and left.dtype == right.dtype
    bitwise = bool(same_shape and torch.equal(left.detach().cpu(), right.detach().cpu()))
    left_f = left.detach().float().cpu()
    right_f = right.detach().float().cpu()
    nonfinite = bool((~torch.isfinite(left_f)).any().item() or (~torch.isfinite(right_f)).any().item())
    if left_f.shape != right_f.shape or left_f.numel() == 0 or right_f.numel() == 0:
        return {
            "bitwise": bitwise,
            "max_abs": None,
            "mean_abs": None,
            "max_rel": None,
            "nonfinite": nonfinite,
            "shape": [tuple(left.shape), tuple(right.shape)],
        }
    diff = (left_f - right_f).abs()
    denom = right_f.abs().clamp_min(1e-8)
    return {
        "bitwise": bitwise,
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "max_rel": float((diff / denom).max().item()),
        "nonfinite": nonfinite,
        "sha256": _digest(left),
        "other_sha256": _digest(right),
    }


def _digest(value) -> str:
    cpu = value.detach().contiguous().cpu()
    payload = str(tuple(cpu.shape)).encode() + str(cpu.dtype).encode() + cpu.numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _apply_mode(mode: str) -> None:
    from src.experiments.calc.rng import set_determinism

    if mode == "training":
        set_determinism(True, cuda=True)
        return
    if mode == "default":
        set_determinism(False, cuda=True)
        return
    raise ValueError(mode)


def _seed(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_model(device: str):
    from src.asr.models.parakeet_tdt import ParakeetTDT

    model = ParakeetTDT(_MODEL)
    model.enable_tdt_loss()
    model.to(device)
    model.eval()
    return model


def _tokens(model, wave) -> list[int]:
    import torch

    lengths = torch.tensor([wave.shape[-1]], device=wave.device)
    recognition = model.recognize(wave.view(1, -1), lengths)[0]
    return [int(token.token_id) for token in recognition.tokens]


def _stages(model, wave, token_ids: list[int]) -> dict:
    """One loss path. Features are the tensor ``encode`` actually consumes."""

    import torch

    waveform = wave.detach().clone()
    if waveform.ndim == 1:
        waveform = waveform.view(1, -1)
    waveform = waveform.requires_grad_(True)
    lengths = torch.tensor([waveform.shape[-1]], device=waveform.device, dtype=torch.long)
    captured: dict[str, torch.Tensor] = {}
    original = model._filterbank.forward

    def wrapped(featurizer, audio, audio_lengths):
        features, feature_lengths = original(featurizer, audio, audio_lengths)
        captured["features"] = features.detach()
        captured["feature_lengths"] = feature_lengths.detach()
        return features, feature_lengths

    model._filterbank.forward = wrapped
    try:
        encoded = model.encode(waveform, lengths)
    finally:
        model._filterbank.forward = original
    if not token_ids:
        zero = waveform.sum() * 0
        zero.backward()
        return {
            "waveform": waveform.detach(),
            "features": captured.get("features", waveform.new_zeros(0)),
            "encoder": encoded.encoded.detach(),
            "decoder": waveform.new_zeros(0),
            "logits": waveform.new_zeros(0),
            "tdt": zero.detach(),
            "grad": waveform.grad.detach(),
            "empty_targets": True,
        }
    targets = torch.tensor([token_ids], dtype=torch.int64, device=waveform.device)
    target_lengths = torch.tensor([len(token_ids)], dtype=torch.int64, device=waveform.device)
    model._set_decoder_backward_enabled(True)
    decoder_outputs, _, _ = model.model.decoder(targets=targets, target_length=target_lengths)
    logits = model.model.joint.joint(encoded.encoded.transpose(1, 2), decoder_outputs.transpose(1, 2))
    act_lens = encoded.encoded_length.detach().to(dtype=torch.int64)
    nll = model._tdt_loss(
        acts=logits,
        labels=targets,
        act_lens=act_lens,
        label_lens=target_lengths,
    )
    loss = nll.mean() if nll.ndim else nll
    loss.backward()
    torch.cuda.synchronize()
    return {
        "waveform": waveform.detach(),
        "features": captured["features"],
        "encoder": encoded.encoded.detach(),
        "decoder": decoder_outputs.detach(),
        "logits": logits.detach(),
        "tdt": loss.detach(),
        "grad": waveform.grad.detach(),
        "empty_targets": False,
        "act_lens": act_lens.detach(),
        "nll": nll.detach(),
    }


def _compare_stages(left: dict, right: dict) -> dict:
    keys = ("waveform", "features", "encoder", "decoder", "logits", "tdt", "grad")
    return {key: _finite_stats(left[key], right[key]) for key in keys}


def _numba_repeat(model, logits, token_ids: list[int], act_lens, repeats: int) -> dict:
    import torch

    device = logits.device
    targets = torch.tensor([token_ids], dtype=torch.int64, device=device)
    target_lengths = torch.tensor([len(token_ids)], dtype=torch.int64, device=device)
    act_lens = act_lens.detach().to(device=device, dtype=torch.int64)
    omega = []
    zero = []
    for _ in range(repeats):
        acts = logits.detach().clone()
        first = model._tdt_loss(acts=acts, labels=targets, act_lens=act_lens, label_lens=target_lengths)
        second = model._deterministic_tdt_loss(
            acts=acts.detach().clone(),
            labels=targets,
            act_lens=act_lens,
            label_lens=target_lengths,
        )
        omega.append(first.detach())
        zero.append(second.detach())
        torch.cuda.synchronize()
    pairs = {
        "omega_checkpoint": [_finite_stats(omega[0], item) for item in omega[1:]],
        "omega_zero": [_finite_stats(zero[0], item) for item in zero[1:]],
    }
    return {
        "omega": model.omega,
        "first_omega": float(omega[0].mean() if omega[0].ndim else omega[0]),
        "first_omega_zero": float(zero[0].mean() if zero[0].ndim else zero[0]),
        "pairs": pairs,
    }


def _save_stages(path: Path, stages: dict) -> None:
    import torch

    cpu = {key: value.detach().cpu() for key, value in stages.items() if key != "empty_targets"}
    cpu["empty_targets"] = torch.tensor([1 if stages["empty_targets"] else 0])
    torch.save(cpu, path)


def _load_stages(path: Path, device: str) -> dict:
    import torch

    cpu = torch.load(path, map_location="cpu", weights_only=False)
    empty = bool(int(cpu.pop("empty_targets")[0]))
    stages = {key: value.to(device) for key, value in cpu.items()}
    stages["empty_targets"] = empty
    return stages


def _child() -> int:
    import torch

    mode = os.environ["AUDIO_DB_TDT_MODE"]
    seed = int(os.environ["AUDIO_DB_TDT_SEED"])
    destination = Path(os.environ["AUDIO_DB_TDT_OUT"])
    tokens = json.loads(os.environ["AUDIO_DB_TDT_TOKENS"])
    _apply_mode(mode)
    _seed(seed)
    device = "cuda:0"
    model = _load_model(device)
    wave = _load_wave().to(device)
    stages = _stages(model, wave, [int(token) for token in tokens])
    _save_stages(destination, stages)
    print(json.dumps({"tdt": float(stages["tdt"]), "bitwise_file": str(destination)}), flush=True)
    return 0


class TDTDeterminismTest(unittest.TestCase):
    def test_stages_repeat_under_training_and_default_cuda(self) -> None:
        reason = _reason()
        if reason:
            self.skipTest(reason)
        import torch

        from src.experiments.calc.rng import capture_rng, restore_rng

        device = "cuda:0"
        report: dict[str, object] = {"cuda_initialized_before_mode": torch.cuda.is_initialized()}
        for mode in ("training", "default"):
            _apply_mode(mode)
            _seed(1234)
            model = _load_model(device)
            wave = _load_wave().to(device)
            lengths = torch.tensor([wave.shape[-1]], device=device)
            first_ids = _tokens(model, wave)
            second_ids = _tokens(model, wave)
            _seed(1234)
            before = capture_rng(cuda=True)
            left = _stages(model, wave, first_ids)
            restore_rng(before, cuda=True)
            restored = _stages(model, wave, first_ids)
            again = _stages(model, wave, first_ids)
            numba = (
                _numba_repeat(model, left["logits"], first_ids, left["act_lens"], repeats=4)
                if first_ids
                else {"empty": True}
            )
            with tempfile.TemporaryDirectory(prefix="tdt-det-") as directory:
                reference = Path(directory) / "reference.pt"
                child_out = Path(directory) / "child.pt"
                _save_stages(reference, left)
                env = os.environ.copy()
                env.update(
                    {
                        "AUDIO_DB_TDT_CHILD": "1",
                        "AUDIO_DB_TDT_MODE": mode,
                        "AUDIO_DB_TDT_SEED": "1234",
                        "AUDIO_DB_TDT_OUT": str(child_out),
                        "AUDIO_DB_TDT_TOKENS": json.dumps(first_ids),
                        "HF_HUB_OFFLINE": "1",
                        "PYTHONPATH": str(_ROOT),
                    }
                )
                completed = subprocess.run(
                    [sys.executable, "-m", "tests.research.test_tdt_determinism"],
                    cwd=_ROOT,
                    env=env,
                    text=True,
                    capture_output=True,
                    timeout=900,
                    check=False,
                )
                child = None
                if completed.returncode == 0 and child_out.is_file():
                    child = _compare_stages(left, _load_stages(child_out, device))
            report[mode] = {
                "omega": model.omega,
                "sigma": model.sigma,
                "token_ids": first_ids,
                "recognition_repeat": first_ids == second_ids,
                "same_process": _compare_stages(left, again),
                "rng_restored": _compare_stages(left, restored),
                "numba_fixed_logits": numba,
                "other_process": child,
                "other_process_returncode": completed.returncode,
                "other_process_stderr_tail": completed.stderr[-1500:],
                "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
                "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            }
            del model
            torch.cuda.empty_cache()
        print("TDT_DETERMINISM " + json.dumps(report, default=str), flush=True)
        training = report["training"]
        self.assertIsInstance(training, dict)
        self.assertEqual(training["other_process_returncode"], 0, training["other_process_stderr_tail"])
        self.assertFalse(training["same_process"]["tdt"]["nonfinite"])
        self.assertGreater(len(training["token_ids"]), 0)


if __name__ == "__main__":
    if os.environ.get("AUDIO_DB_TDT_CHILD") == "1":
        raise SystemExit(_child())
    unittest.main()
