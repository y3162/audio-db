"""Real Parakeet and SEMamba++ checks for the unified environment.

This is a real-model GPU test. Run it with ``.venvs/se`` and one free GPU.
A missing package is a skip. Hidden CUDA is NOT_RUN. A wrong
gradient or a failed launch is a failure.

The one-step launches use the checked-in research YAML (official width,
reference scan) and write only under a temporary directory.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_SPEECH = _ROOT / "data/corpora/VCTK/wav48/p225/p225_001.wav"
_NOISE = _ROOT / "data/corpora/DNS2020/datasets/noise/---1_cCGK4M.wav"
_CONFIGS = _ROOT / "configs/research/asr_guided_se/semambapp/v1"
_PRIORITY = ("all_off", "features", "error", "tdt", "all_on")
_REMAINING = ("features_error", "features_tdt", "error_tdt")
_NEMO_MARKERS = ("preprocessor", "joint.", "decoder.prediction", "fastconformer")


def _reexec_numba_cuda() -> None:
    """Start again so Numba's loader sees this venv's CUDA 12 NVVM."""

    if os.environ.get("AUDIO_DB_NUMBA_CUDA") == "ready":
        return
    if Path(sys.prefix).name != "se":
        return
    if os.environ.get("CUDA_HOME"):
        os.environ["AUDIO_DB_NUMBA_CUDA"] = "ready"
        return
    site = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    home = site / "nvidia" / "cuda_nvcc"
    if not (home / "nvvm" / "lib64" / "libnvvm.so").is_file():
        return
    env = os.environ.copy()
    env["CUDA_HOME"] = str(home)
    env["PATH"] = str(home / "bin") + os.pathsep + env.get("PATH", "")
    libs = [
        str(home / "nvvm" / "lib64"),
        str(home / "lib64"),
        str(site / "nvidia" / "cuda_runtime" / "lib"),
    ]
    previous = env.get("LD_LIBRARY_PATH")
    env["LD_LIBRARY_PATH"] = os.pathsep.join(libs) + (os.pathsep + previous if previous else "")
    env["AUDIO_DB_NUMBA_CUDA"] = "ready"
    os.execve(sys.executable, [sys.executable, *sys.argv], env)


def _stack_reason() -> str | None:
    try:
        import nemo  # noqa: F401
        import numba  # noqa: F401
        from nnAudio.features.cqt import CQT2010v2  # noqa: F401
    except Exception:
        return "NeMo, Numba, or nnAudio is not installed in this environment"
    return None


def _gpu_reason() -> str | None:
    missing = _stack_reason()
    if missing:
        return missing
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") in {"", "-1"}:
        return "NOT_RUN: set CUDA_VISIBLE_DEVICES to one free GPU"
    if not _SPEECH.is_file() or not _NOISE.is_file():
        return "smoke speech or noise wav is absent"
    return None


_reexec_numba_cuda()
_STACK_SKIP = _stack_reason()
_GPU_SKIP = _gpu_reason()


def _load_16k(path: Path, seconds: float):
    import soundfile as sf
    import torch
    import torchaudio

    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    wave = torch.as_tensor(audio, dtype=torch.float32)
    if wave.ndim == 2:
        wave = wave.mean(dim=1)
    if int(sample_rate) != 16000:
        wave = torchaudio.functional.resample(wave, int(sample_rate), 16000)
    count = max(1, int(seconds * 16000))
    return wave[:count].contiguous()


def _mix(speech, noise, scale: float = 0.3):
    import torch

    clip = noise[: speech.shape[0]]
    if clip.shape[0] < speech.shape[0]:
        clip = torch.nn.functional.pad(clip, (0, speech.shape[0] - clip.shape[0]))
    speech_rms = torch.sqrt(torch.mean(speech.square()) + 1e-8)
    noise_rms = torch.sqrt(torch.mean(clip.square()) + 1e-8)
    mixed = speech + float(scale) * clip * (speech_rms / noise_rms)
    return mixed.clamp(-1.0, 1.0).contiguous()


def _narrow(**overrides):
    values = dict(
        seed=1234,
        hid_feature=8,
        num_tfmamba=1,
        d_state=4,
        d_conv=2,
        expand=2,
        norm_epsilon=1e-5,
        input_channel=2,
        output_channel=1,
        n_fft=400,
        scan_backend="reference",
        hop=100,
        win=400,
        sampling_rate=16000,
    )
    values.update(overrides)
    return values


def _loaded_library(name: str) -> str:
    for line in Path("/proc/self/maps").read_text(encoding="utf-8", errors="replace").splitlines():
        if name in line:
            return line.rsplit(" ", 1)[-1]
    return ""


def _emit(name: str, payload: dict) -> None:
    print(name + " " + json.dumps(payload, sort_keys=True, default=str), flush=True)


@unittest.skipIf(_STACK_SKIP is not None, _STACK_SKIP or "")
class UnifiedEnvironmentTest(unittest.TestCase):
    def test_pinned_imports(self) -> None:
        import nemo
        import numba
        import numpy
        import torch
        from nnAudio.features.cqt import CQT2010v2

        self.assertEqual(nemo.__version__, "2.2.1")
        self.assertEqual(numba.__version__, "0.61.0")
        self.assertEqual(numpy.__version__, "2.1.3")
        self.assertTrue(torch.__version__.startswith("2.6.0"))
        self.assertTrue(callable(CQT2010v2))


@unittest.skipIf(_GPU_SKIP is not None, _GPU_SKIP or "")
class RealParakeetGPUTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch

        from src.research.asr_guided_se.asr import FrozenParakeet

        if not torch.cuda.is_available():
            raise unittest.SkipTest("NOT_RUN: CUDA is not available")
        cls.device = torch.device("cuda")
        cls.speech = _load_16k(_SPEECH, 2.0).to(cls.device)
        cls.noisy = _mix(cls.speech, _load_16k(_NOISE, 2.0).to(cls.device))
        try:
            cls.asr = FrozenParakeet(tdt_loss=True).to_device(cls.device)
        except RuntimeError as exc:
            if "Refusing to download" in str(exc):
                raise unittest.SkipTest(str(exc)) from exc
            raise
        cls.lengths = torch.tensor([cls.speech.shape[0]], device=cls.device)

    def test_encoder_layer_15_is_finite_and_detached(self) -> None:
        import torch

        asr = self.asr
        self.assertEqual(asr.samples_per_frame, 1280)
        self.assertGreater(asr.encoder_layers, 15)
        self.assertEqual(asr.inner.omega, 0.1)
        hidden, encoded_length = asr.encode_layer(self.noisy.view(1, -1), self.lengths, 15)
        hidden = hidden.detach()
        self.assertEqual(hidden.ndim, 3)
        self.assertEqual(int(hidden.shape[0]), 1)
        self.assertEqual(int(hidden.shape[1]), int(asr.encoder_dim))
        self.assertEqual(int(hidden.shape[2]), int(encoded_length[0]))
        self.assertTrue(torch.isfinite(hidden).all())
        self.assertIsNone(hidden.grad_fn)
        for parameter in asr.parameters():
            self.assertFalse(parameter.requires_grad)
            self.assertIsNone(parameter.grad)

    def test_tdt_gradient_reaches_the_waveform_and_not_asr(self) -> None:
        import torch

        from numba.cuda.cuda_paths import get_cuda_paths

        asr = self.asr
        recognition = asr.recognize(self.speech.view(1, -1), self.lengths)[0]
        ids = [int(token.token_id) for token in recognition.tokens]
        self.assertTrue(ids)
        before = {name: parameter.detach().clone() for name, parameter in asr.inner.model.named_parameters()}
        wave = self.speech.detach().clone().requires_grad_(True)
        deterministic = []
        sampled = []
        for _ in range(2):
            deterministic.append(
                float(
                    asr.loss_from_ids(wave.view(1, -1), self.lengths, [ids], deterministic_tdt=True).detach()
                )
            )
            sampled.append(
                float(
                    asr.loss_from_ids(wave.view(1, -1), self.lengths, [ids], deterministic_tdt=False).detach()
                )
            )
        loss = asr.loss_from_ids(wave.view(1, -1), self.lengths, [ids], deterministic_tdt=False)
        self.assertTrue(torch.isfinite(loss).all())
        loss.sum().backward()
        self.assertIsNotNone(wave.grad)
        self.assertTrue(torch.isfinite(wave.grad).all())
        self.assertGreater(float(wave.grad.abs().sum()), 0.0)
        for name, parameter in asr.inner.model.named_parameters():
            self.assertFalse(parameter.requires_grad, name)
            self.assertIsNone(parameter.grad, name)
            self.assertTrue(torch.equal(before[name], parameter), name)
        # The CUDA 12 wheel names the library libnvvm.so, without the version
        # suffix Numba's finder requires. Loading then uses LD_LIBRARY_PATH.
        nvvm = _loaded_library("libnvvm.so")
        self.assertIn("cuda_nvcc", nvvm)
        self.assertNotIn("/usr/local/cuda/", nvvm)
        located = get_cuda_paths()["nvvm"]
        _emit(
            "ASR_GUIDED_TDT",
            {
                "text": recognition.text,
                "tokens": len(ids),
                "omega": asr.inner.omega,
                "deterministic_losses": deterministic,
                "sampled_losses": sampled,
                "deterministic_equal": deterministic[0] == deterministic[1],
                "sampled_equal": sampled[0] == sampled[1],
                "waveform_grad_abs_sum": float(wave.grad.abs().sum()),
                "nvvm": nvvm,
                "nvvm_located_by": located.by,
                "nvvm_located_info": located.info,
                "numba": asr.inner.software_fingerprint().get("numba"),
            },
        )

    def test_error_mask_uses_real_clean_and_noisy_tokens(self) -> None:
        from src.research.asr_guided_se.error_mask import crop_error_mask

        asr = self.asr
        clean = asr.recognize(self.speech.view(1, -1), self.lengths)[0]
        # The mild mix used above can decode to the same text. A louder mix is
        # what makes the real alignment mark an error region.
        loud = _mix(self.speech, _load_16k(_NOISE, 2.0).to(self.speech.device), scale=4.0)
        noisy = asr.recognize(loud.view(1, -1), self.lengths)[0]
        self.assertTrue(clean.text)
        self.assertNotEqual(clean.text, noisy.text)
        window = int(self.speech.shape[0])
        crop_start = 1600
        content = window - crop_start
        mask = crop_error_mask(
            clean.tokens,
            noisy.tokens,
            crop_start=crop_start,
            content_length=content,
            window=window,
            hop=int(asr.samples_per_frame),
            full_length=window,
        )
        self.assertEqual(len(mask), window)
        self.assertGreater(sum(bool(flag) for flag in mask), 0)
        self.assertTrue(all(index < content for index, flag in enumerate(mask) if flag))
        self.assertFalse(any(mask[content:]))
        _emit(
            "ASR_GUIDED_MASK",
            {
                "clean_text": clean.text,
                "noisy_text": noisy.text,
                "clean_tokens": len(clean.tokens),
                "noisy_tokens": len(noisy.tokens),
                "mask_true": sum(bool(flag) for flag in mask),
                "mask_len": len(mask),
            },
        )


@unittest.skipIf(_GPU_SKIP is not None, _GPU_SKIP or "")
class RealGeneratorGPUTest(unittest.TestCase):
    def test_reference_and_fused_scans_backward(self) -> None:
        import torch

        from src.models.semambapp.models import SEMambaPPGenerator
        from nnAudio.features.cqt import CQT2010v2

        device = torch.device("cuda")
        wave = torch.randn(1, 8000, device=device)
        cqt = CQT2010v2(sr=16000, hop_length=256, n_bins=84, bins_per_octave=12, verbose=False).to(device)
        spectrum = cqt(wave)
        self.assertTrue(torch.isfinite(spectrum).all())
        for backend in ("reference", "fused"):
            generator = SEMambaPPGenerator(**_narrow(scan_backend=backend)).to(device)
            output = generator(wave, wave)
            loss = output["audio"].square().mean()
            loss.backward()
            grads = [parameter.grad for parameter in generator.parameters() if parameter.requires_grad]
            self.assertTrue(grads)
            self.assertTrue(all(item is not None and torch.isfinite(item).all() for item in grads))
            self.assertGreater(sum(float(item.abs().sum()) for item in grads if item is not None), 0.0)

    def test_tdt_updates_the_generator_and_leaves_asr_fixed(self) -> None:
        import torch

        from src.research.asr_guided_se.asr import FrozenParakeet
        from src.research.asr_guided_se.semambapp.v1.config import GuidanceV1
        from src.research.asr_guided_se.semambapp.v1.model import GuidedSEMambaPPGenerator

        device = torch.device("cuda")
        speech = _load_16k(_SPEECH, 2.0).to(device)
        noisy = _mix(speech, _load_16k(_NOISE, 2.0).to(device))
        try:
            asr = FrozenParakeet(tdt_loss=True).to_device(device)
        except RuntimeError as exc:
            if "Refusing to download" in str(exc):
                self.skipTest(str(exc))
            raise
        lengths = torch.tensor([speech.shape[0]], device=device)
        ids = [int(token.token_id) for token in asr.recognize(speech.view(1, -1), lengths)[0].tokens]
        self.assertTrue(ids)
        guidance = GuidanceV1(True, 15, False, 0.0, True)
        generator = GuidedSEMambaPPGenerator(guidance=guidance, asr=asr, **_narrow()).to(device)
        generator.train()
        before_asr = {name: parameter.detach().clone() for name, parameter in asr.inner.model.named_parameters()}
        clean = speech.view(1, -1)
        output = generator(clean, noisy.view(1, -1))
        audio = output["audio"]
        out_lengths = torch.tensor([audio.shape[-1]], device=device)
        loss = asr.loss_from_ids(audio, out_lengths, [ids], deterministic_tdt=False)
        loss.sum().backward()
        fuse = generator.generator.asr_fuse.weight
        self.assertIsNotNone(fuse.grad)
        self.assertTrue(torch.isfinite(fuse.grad).all())
        self.assertGreater(float(fuse.grad.abs().sum()), 0.0)
        body = [
            parameter
            for name, parameter in generator.generator.named_parameters()
            if name.startswith("TSMamba") and parameter.grad is not None
        ]
        self.assertTrue(body)
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in body))
        trainable = [parameter for parameter in generator.parameters() if parameter.requires_grad]
        before = [parameter.detach().clone() for parameter in trainable]
        torch.optim.AdamW(trainable, lr=1e-4).step()
        self.assertTrue(any(not torch.equal(old, parameter) for old, parameter in zip(before, trainable)))
        self.assertTrue(all(torch.isfinite(parameter).all() for parameter in trainable))
        for name, parameter in asr.inner.model.named_parameters():
            self.assertFalse(parameter.requires_grad, name)
            self.assertIsNone(parameter.grad, name)
            self.assertTrue(torch.equal(before_asr[name], parameter), name)
        _emit(
            "ASR_GUIDED_GENERATOR",
            {
                "tdt": float(loss.detach()),
                "fuse_grad_abs_sum": float(fuse.grad.abs().sum()),
                "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024 * 1024),
            },
        )


def _launch(name: str, directory: Path, *, preflight: bool) -> dict:
    source = _CONFIGS / f"{name}.yaml"
    run = directory / "run"
    preflight_line = "" if preflight else "    preflight: false\n"
    (directory / "experiment.yaml").write_text(
        "extends: "
        + str(source)
        + "\nruntime:\n  kwargs:\n    device: cuda\n"
        + preflight_line
        + "output:\n  directory: "
        + str(run)
        + "\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("AUDIO_DB_GPU_MEMORY_INTERVAL", "2")
    env["AUDIO_DB_RESOURCE_PROFILE"] = "1"
    env["PYTHONPATH"] = str(_ROOT)
    started = time.perf_counter()
    completed = subprocess.run(
        [
            str(_ROOT / "scripts/setup/with_env.sh"),
            sys.executable,
            "-m",
            "src.experiments.launcher",
            "--config",
            str(directory / "experiment.yaml"),
            "--nproc",
            "1",
        ],
        cwd=_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=1500,
        check=False,
    )
    elapsed = time.perf_counter() - started
    log = directory / "launcher.log"
    log.write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
    return {"returncode": completed.returncode, "elapsed_seconds": elapsed, "run": run, "log": log}


def _step_report(name: str, launched: dict) -> dict:
    import torch

    run = Path(launched["run"])
    report = {
        "name": name,
        "returncode": launched["returncode"],
        "elapsed_seconds": round(launched["elapsed_seconds"], 3),
        "log": str(launched["log"]),
    }
    if launched["returncode"] != 0:
        text = launched["log"].read_text(encoding="utf-8", errors="replace")
        report["error_tail"] = text[-2000:]
        return report
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)
    metrics = [
        json.loads(line)
        for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    keys = []
    for state_name, blob in checkpoint.get("states", {}).items():
        if isinstance(blob, dict):
            keys.extend(f"{state_name}.{key}" for key in blob)
    forbidden = [key for key in keys if any(marker in key for marker in _NEMO_MARKERS)]
    profile = {}
    profile_path = run / "resource_profile.json"
    if profile_path.is_file():
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
    peaks = []
    summary_path = run / "gpu_memory.json"
    if summary_path.is_file():
        peaks = json.loads(summary_path.read_text(encoding="utf-8")).get("peaks", [])
    probe = None
    meta_path = run / "metadata.json"
    if meta_path.is_file():
        launches = json.loads(meta_path.read_text(encoding="utf-8")).get("launches") or []
        if launches:
            probe = launches[-1].get("batch_probe")
    report.update(
        {
            "status": status.get("status"),
            "logical_step": checkpoint.get("logical_step"),
            "format": checkpoint.get("format"),
            "optimizer_updates": checkpoint.get("optimizer_updates"),
            "loss": None if not metrics else metrics[-1].get("loss"),
            "metric_keys": sorted(metrics[-1]) if metrics else [],
            "nemo_keys": forbidden,
            "peak_allocated_mib": None
            if profile.get("peak_allocated_bytes") is None
            else profile["peak_allocated_bytes"] / (1024 * 1024),
            "gpu_peaks_mib": [row.get("memory_used_mib") for row in peaks],
            "preflight_status": None if not isinstance(probe, dict) else probe.get("status", probe.get("finite")),
            "preflight_loss": None if not isinstance(probe, dict) else probe.get("loss"),
        }
    )
    return report


@unittest.skipIf(_GPU_SKIP is not None, _GPU_SKIP or "")
class ResearchTrainerOneStepTest(unittest.TestCase):
    def test_official_width_one_logical_step(self) -> None:
        reports = []
        failed = False
        for name in _PRIORITY:
            with tempfile.TemporaryDirectory(prefix=f"asr-guided-{name}-") as directory:
                launched = _launch(name, Path(directory), preflight=True)
                report = _step_report(name, launched)
            reports.append(report)
            _emit("ASR_GUIDED_STEP", report)
            if report.get("status") != "COMPLETED" or report.get("logical_step") != 1 or report.get("nemo_keys"):
                failed = True
                break
        if not failed:
            for name in _REMAINING:
                with tempfile.TemporaryDirectory(prefix=f"asr-guided-{name}-") as directory:
                    launched = _launch(name, Path(directory), preflight=True)
                    report = _step_report(name, launched)
                reports.append(report)
                _emit("ASR_GUIDED_STEP", report)
                if report.get("status") != "COMPLETED" or report.get("logical_step") != 1 or report.get("nemo_keys"):
                    failed = True
        preflight = None
        if any(item.get("name") == "tdt" and item.get("status") == "COMPLETED" for item in reports):
            with tempfile.TemporaryDirectory(prefix="asr-guided-tdt-noprobe-") as directory:
                launched = _launch("tdt", Path(directory), preflight=False)
                preflight = _step_report("tdt_preflight_off", launched)
            _emit("ASR_GUIDED_PREFLIGHT", preflight)
            probed = next(item for item in reports if item["name"] == "tdt")
            if preflight.get("loss") is not None and probed.get("loss") is not None:
                preflight["loss_abs_diff_vs_preflight_on"] = abs(float(preflight["loss"]) - float(probed["loss"]))
                _emit("ASR_GUIDED_PREFLIGHT", preflight)
        self.assertFalse(failed, json.dumps(reports, default=str)[-4000:])
        self.assertEqual([item["name"] for item in reports], list(_PRIORITY + _REMAINING))
        for report in reports:
            self.assertEqual(report["format"], "audio-db-experiment-v1")
            self.assertEqual(report["nemo_keys"], [])
            self.assertTrue(report["optimizer_updates"])
            self.assertTrue(all(int(value) >= 1 for value in report["optimizer_updates"].values()))
            self.assertIsNotNone(report["loss"])
        if preflight is not None:
            self.assertEqual(preflight.get("status"), "COMPLETED")


if __name__ == "__main__":
    unittest.main()
