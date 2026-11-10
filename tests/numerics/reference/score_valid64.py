"""Enhance the fixed 64-utterance validation slice from one saved generator.

Metrics are applied later with the lab environment. This process only writes
waveforms. Existing outputs are left in place.

    CUDA_VISIBLE_DEVICES=<free> .venvs/semamba/bin/python \\
        tests/numerics/reference/score_valid64.py official \\
        data/results/training/semamba-200step-20261101/official-a/state_00000200.pt \\
        data/results/training/semamba-valid64-20261101/official-a-step200.pt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src/se/official/semambapp"))

from model.semambapp import SEMambapp
from src.se.models.semambapp.network import SEMambaPP
from src.experiments.calc.dataset import load_recipe_dataset
from src.experiments.calc.stft import semamba_mag_phase_istft, semamba_mag_phase_stft
from tests.numerics.reference.compare_semamba_step import _cfg, _flat, _official_parts

DB = ROOT / "data/db/semamba-voicebank-train-20261101-compat.duckdb"
SELECTION = ROOT / "data/results/training/semamba-valid64-20261101/selection.json"


def main() -> None:
    side, state_path, destination = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    if side not in ("official", "independent") or destination.exists():
        raise SystemExit("usage: score_valid64.py official|independent STATE NEW_PT")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    selected = json.loads(SELECTION.read_text(encoding="utf-8"))
    wanted = [row["case_id"] for row in selected["utterances"]]
    valid = load_recipe_dataset(DB, "voicebank_28_valid", role="validation", seed=1234, sample_rate=16000, segment_size=None, training=False)
    positions = {example.case_id: index for index, example in enumerate(valid.examples)}
    missing = [case_id for case_id in wanted if case_id not in positions]
    if missing:
        raise SystemExit(f"missing case ids: {missing[:3]}")
    device = torch.device("cuda")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    generator = _generator(side, state, device)
    rows = []
    enhanced = []
    with torch.no_grad():
        for case_id in wanted:
            batch = valid.collate([positions[case_id]])
            clean = batch["clean"].to(device)
            noisy = batch["noisy"].to(device)
            audio = _enhance(side, generator, clean, noisy)
            enhanced.append(audio[0].detach().cpu())
            rows.append({"case_id": case_id, "samples": int(clean.shape[-1]), "enhanced_samples": int(audio.shape[-1])})
            print(len(rows), case_id[:12], audio.shape[-1], flush=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"side": side, "state": str(state_path), "rows": rows, "enhanced": enhanced}, destination)


def _generator(side: str, state: dict, device: torch.device) -> torch.nn.Module:
    if side == "official":
        module = SEMambapp(_cfg(48, 4, 16, 4))
    else:
        module = SEMambaPP(_flat(_cfg(48, 4, 16, 4)))
    module.load_state_dict(state["generator"], strict=True)
    return module.to(device).eval()


def _enhance(side: str, generator: torch.nn.Module, clean: torch.Tensor, noisy: torch.Tensor) -> torch.Tensor:
    if side == "official":
        return _official_parts(generator, clean, noisy, _cfg(48, 4, 16, 4))["audio"]
    clean_mag, _, _ = semamba_mag_phase_stft(clean, 400, 100, 400, 0.3, addeps=False)
    noisy_mag, noisy_phase, _ = semamba_mag_phase_stft(noisy, 400, 100, 400, 0.3, addeps=False)
    magnitude, phase = generator(noisy_mag, noisy_phase)
    return semamba_mag_phase_istft(magnitude, phase, 400, 100, 400, 0.3, length=clean.shape[-1])


if __name__ == "__main__":
    main()
