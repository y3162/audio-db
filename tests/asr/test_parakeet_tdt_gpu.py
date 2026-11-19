"""Real Parakeet TDT v2 checks. They load the cached 0.6B checkpoint on one GPU.

Run with .venvs/se and set CUDA_VISIBLE_DEVICES to one free GPU.
Set HF_HUB_OFFLINE=1 to use only cached model files.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _sine(samples: int, frequency: float) -> "torch.Tensor":
    import torch

    time = torch.arange(samples, dtype=torch.float32) / 16000.0
    return (0.1 * torch.sin(2.0 * torch.pi * frequency * time)).contiguous()


def _payload(recognition) -> dict:
    return {
        "text": recognition.text,
        "token_ids": [token.token_id for token in recognition.tokens],
        "start_offsets": [token.start_offset for token in recognition.tokens],
        "end_offsets": [token.end_offset for token in recognition.tokens],
        "durations": [token.end_offset - token.start_offset for token in recognition.tokens],
    }


def _child() -> None:
    import torch

    from src.asr.models.parakeet_tdt import ParakeetTDT

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ParakeetTDT().to(device)
    wave = _sine(16000, 220.0).to(device)
    length = torch.tensor([wave.shape[0]], device=wave.device)
    recognition = model.recognize(wave.view(1, -1), length)[0]
    from src.asr.models.parakeet import _live_cuda_graph

    live = _live_cuda_graph(model.model)
    print(
        "PARAKEET_TDT_CHILD "
        + json.dumps(
            {
                "payload": _payload(recognition),
                "decoding": model.effective_decoding,
                "live_cuda_graph": None if live is None else bool(live),
                "fingerprint": model.software_fingerprint(),
            },
            sort_keys=True,
        )
    )


def _nemo_available() -> bool:
    try:
        import nemo  # noqa: F401
    except Exception:
        return False
    return True


@unittest.skipUnless(_nemo_available(), "NeMo is not installed in this environment")
@unittest.skipUnless(os.environ.get("CUDA_VISIBLE_DEVICES", "") not in {"", "-1"}, "set CUDA_VISIBLE_DEVICES to one free GPU")
class ParakeetTDTGPUTest(unittest.TestCase):
    def test_decode_repeatability_batch_and_waveform_grad(self) -> None:
        import torch

        from src.asr.models.parakeet import _live_cuda_graph
        from src.asr.models.parakeet_tdt import ParakeetTDT

        self.assertTrue(torch.cuda.is_available())
        device = torch.device("cuda")
        model = ParakeetTDT().to(device)
        live = _live_cuda_graph(model.model)
        self.assertIs(live, False)
        self.assertEqual(model.effective_decoding["strategy"], "greedy_batch")
        self.assertFalse(model.effective_decoding["compute_timestamps"])
        self.assertTrue(model.effective_decoding["tdt_include_token_duration"])
        self.assertFalse(model.effective_decoding["use_cuda_graph_decoder"])
        self.assertEqual(model.samples_per_encoder_frame, 1280)
        self.assertEqual(model.omega, 0.1)
        self.assertEqual(model.durations, [0, 1, 2, 3, 4])

        first_wave = _sine(16000, 220.0).to(device)
        length = torch.tensor([first_wave.shape[0]], device=device)
        first = _payload(model.recognize(first_wave.view(1, -1), length)[0])
        second = _payload(model.recognize(first_wave.view(1, -1), length)[0])
        self.assertEqual(first, second)
        self.assertTrue(first["token_ids"], "recognition produced no tokens to compare")

        other = _sine(16000, 330.0).to(device)
        batched = model.recognize(torch.stack((first_wave, other)), torch.tensor([16000, 16000], device=device))
        alone = model.recognize(other.view(1, -1), length)[0]
        self.assertEqual(_payload(batched[0]), first)
        self.assertEqual(_payload(batched[1]), _payload(alone))

        env = os.environ.copy()
        env["PARAKEET_TDT_CHILD"] = "1"
        env.setdefault("HF_HUB_OFFLINE", "1")
        children = []
        for _ in range(2):
            completed = subprocess.run(
                [sys.executable, str(Path(__file__).resolve())],
                cwd=_ROOT,
                env=env,
                check=False,
                capture_output=True,
                text=True,
                timeout=900,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr[-2000:] + completed.stdout[-500:])
            line = next(item for item in completed.stdout.splitlines() if item.startswith("PARAKEET_TDT_CHILD "))
            children.append(json.loads(line.split(" ", 1)[1]))
        self.assertEqual(children[0]["payload"], children[1]["payload"])
        self.assertEqual(children[0]["payload"], first)
        self.assertIs(children[0]["live_cuda_graph"], False)

        wave = first_wave.detach().clone().requires_grad_(True)
        before = {name: parameter.detach().clone() for name, parameter in model.model.named_parameters()}
        model.enable_tdt_loss()
        ids = first["token_ids"]
        deterministic = []
        sampled = []
        for _ in range(2):
            deterministic.append(
                float(
                    model.loss_from_ids(
                        wave.view(1, -1),
                        length,
                        [ids],
                        deterministic_tdt=True,
                    ).detach()
                )
            )
        for _ in range(2):
            sampled.append(
                float(
                    model.loss_from_ids(
                        wave.view(1, -1),
                        length,
                        [ids],
                        deterministic_tdt=False,
                    ).detach()
                )
            )
        loss = model.loss_from_ids(wave.view(1, -1), length, [ids], deterministic_tdt=True)
        loss.sum().backward()
        self.assertIsNotNone(wave.grad)
        self.assertTrue(torch.isfinite(wave.grad).all())
        self.assertGreater(float(wave.grad.abs().sum()), 0.0)
        for name, parameter in model.model.named_parameters():
            self.assertFalse(parameter.requires_grad, name)
            self.assertIsNone(parameter.grad, name)
            self.assertTrue(torch.equal(before[name], parameter), name)
        report = {
            "gpu": torch.cuda.get_device_name(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "samples_per_encoder_frame": model.samples_per_encoder_frame,
            "omega": model.omega,
            "sigma": model.sigma,
            "durations": model.durations,
            "blank": model.blank_id,
            "decoding": model.effective_decoding,
            "live_cuda_graph": bool(live),
            "repeatable_text": first["text"],
            "token_count": len(first["token_ids"]),
            "process_reproducible": True,
            "batch_invariant": True,
            "deterministic_losses": deterministic,
            "sampled_losses": sampled,
            "waveform_grad_abs_sum": float(wave.grad.abs().sum()),
            "fingerprint": model.software_fingerprint(),
        }
        print("PARAKEET_TDT_REPORT " + json.dumps(report, sort_keys=True, default=str))


if os.environ.get("PARAKEET_TDT_CHILD") == "1":
    _child()
    raise SystemExit(0)

if __name__ == "__main__":
    unittest.main()
