"""TDT v2 decode rules without loading NeMo or a checkpoint."""

import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from src.asr.catalog import ModelCard
from src.asr.core.recognizer import SpeechRecognizer
from src.asr.models.parakeet import (
    TDT_V2_DECODING,
    TDT_V2_MODEL_ID,
    apply_tdt_v2_decoding,
    configure_parakeet_decoding,
    effective_tdt_v2_decoding,
)

_ROOT = Path(__file__).resolve().parents[2]


class _Decode(SimpleNamespace):
    def __init__(self):
        super().__init__()
        self.cfg = SimpleNamespace(decoding={"strategy": "beam", "greedy": {"max_symbols": 10}})
        self.changed = 0
        self.decoding = SimpleNamespace(decoding=SimpleNamespace())

    def change_decoding_strategy(self, decoding, verbose=False):
        del verbose
        self.changed += 1
        self.cfg.decoding = decoding


class ParakeetDecodingTest(unittest.TestCase):
    def test_tdt_v2_writes_the_fixed_decode_and_calls_the_model(self) -> None:
        model = _Decode()
        observed = apply_tdt_v2_decoding(model)
        self.assertEqual(observed, TDT_V2_DECODING)
        self.assertEqual(model.changed, 1)
        self.assertEqual(effective_tdt_v2_decoding(model), TDT_V2_DECODING)
        self.assertFalse(model.cfg.decoding["greedy"]["use_cuda_graph_decoder"])

    def test_a_failed_change_is_not_swallowed(self) -> None:
        class Broken(_Decode):
            def change_decoding_strategy(self, decoding, verbose=False):
                del decoding, verbose
                raise RuntimeError("decoder rejected the config")

        with self.assertRaisesRegex(RuntimeError, "could not be applied"):
            apply_tdt_v2_decoding(Broken())

    def test_an_ineffective_change_fails(self) -> None:
        class Reverts(_Decode):
            def change_decoding_strategy(self, decoding, verbose=False):
                del verbose
                self.changed += 1
                decoding["strategy"] = "beam"

        with self.assertRaisesRegex(RuntimeError, "effective decoding differs"):
            apply_tdt_v2_decoding(Reverts())

    def test_a_live_cuda_graph_flag_must_be_off(self) -> None:
        class StillOn(_Decode):
            def change_decoding_strategy(self, decoding, verbose=False):
                super().change_decoding_strategy(decoding, verbose=verbose)
                self.decoding.decoding.use_cuda_graph_decoder = True

        with self.assertRaisesRegex(RuntimeError, "CUDA-graph decoder"):
            apply_tdt_v2_decoding(StillOn())

    def test_other_families_keep_the_old_optional_path(self) -> None:
        model = _Decode()
        self.assertIsNone(configure_parakeet_decoding(model, "nvidia/parakeet-ctc-0.6b"))
        self.assertEqual(model.changed, 0)
        self.assertEqual(configure_parakeet_decoding(model, TDT_V2_MODEL_ID)["strategy"], "greedy_batch")
        self.assertEqual(model.changed, 1)

    def test_recognizer_keeps_backend_metadata_and_flattens_decode(self) -> None:
        card = ModelCard(
            family="parakeet",
            model_id=TDT_V2_MODEL_ID,
            backend="nemo",
            decode="tdt",
            languages=("en",),
            asr_finetuned=True,
            provider="NVIDIA",
            license="CC-BY-4.0",
        )
        seen = SpeechRecognizer(
            card,
            device="cpu",
            language="en",
            runner=lambda _audio: "ok",
            extra={"decoding": dict(TDT_V2_DECODING)},
        ).transcribe(torch.zeros(160), 16000)
        self.assertEqual(seen.extra["backend"], "nemo")
        self.assertEqual(seen.extra["decoding"]["strategy"], "greedy_batch")
        self.assertEqual(seen.extra["sample_rate"], 16000)
        self.assertEqual(seen.text, "ok")
        settings = SpeechRecognizer(
            card,
            device="cpu",
            language="en",
            runner=lambda _audio: "ok",
            extra={"decoding": dict(TDT_V2_DECODING)},
        ).settings()
        self.assertEqual(settings["backend"], "nemo")
        self.assertEqual(settings["decoding.strategy"], "greedy_batch")
        self.assertEqual(settings["decoding.use_cuda_graph_decoder"], "False")

    def test_importing_the_public_asr_package_does_not_load_nemo(self) -> None:
        code = (
            "import src.asr, src.models, src.research.asr_guided_se\n"
            "import sys\n"
            "assert 'nemo' not in sys.modules\n"
            "assert 'src.asr.models.parakeet_tdt' not in sys.modules\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
