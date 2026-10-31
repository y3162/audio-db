"""Compare the independent SEMamba++ discriminators with the reference modules.

The reference tree is imported only here. Run with the semamba environment,
which has nnAudio, librosa, and torchaudio:

    .venvs/semamba/bin/python tests/training/reference/compare_semamba_disc.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
OFFICIAL = ROOT / "src" / "se" / "official" / "semambapp"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OFFICIAL))

from src.training.se.semamba_disc import (  # noqa: E402
    MultiResolutionDiscriminator,
    MultiScaleSubbandCQTDiscriminator,
)
from model.discriminator import (  # noqa: E402
    MultiResolutionDiscriminator as OfficialMRD,
    MultiScaleSubbandCQTDiscriminator as OfficialCQT,
)


def _count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _load(official: torch.nn.Module, independent: torch.nn.Module) -> dict[str, list[str]]:
    official_keys = set(official.state_dict())
    independent_keys = set(independent.state_dict())
    report = {
        "missing_in_independent": sorted(official_keys - independent_keys),
        "unexpected_in_independent": sorted(independent_keys - official_keys),
    }
    if report["missing_in_independent"] or report["unexpected_in_independent"]:
        return report
    independent.load_state_dict(official.state_dict(), strict=True)
    return report


def _gap(reference: torch.Tensor, other: torch.Tensor) -> dict[str, float]:
    delta = (reference.detach().float() - other.detach().float()).abs()
    denominator = reference.detach().float().abs().clamp_min(1e-8)
    return {"max_abs": float(delta.max()), "max_rel": float((delta / denominator).max())}


def _flatten(scores, features) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    flat_scores = []
    flat_features = []
    for score, maps in zip(scores, features, strict=True):
        flat_scores.append(score)
        flat_features.extend(maps)
    return flat_scores, flat_features


def _compare_pair(official, independent, waveform: torch.Tensor, train: bool) -> dict:
    if train:
        official.train()
        independent.train()
    else:
        official.eval()
        independent.eval()
    with torch.random.fork_rng():
        torch.manual_seed(19)
        real_s, fake_s, real_f, fake_f = official(waveform, waveform * 0.5)
    with torch.random.fork_rng():
        torch.manual_seed(19)
        our_real_s, our_real_f = independent(waveform)
        our_fake_s, our_fake_f = independent(waveform * 0.5)
    official_scores, official_features = _flatten(real_s, real_f)
    report = {
        "mode": "train" if train else "eval",
        "score_count": [len(real_s), len(our_real_s)],
        "feature_count": [sum(len(item) for item in real_f), len(our_real_f)],
        "scores": [],
        "features": [],
    }
    for index, (left, right) in enumerate(zip(official_scores, our_real_s, strict=True)):
        report["scores"].append({"index": index, **_gap(left, right), "shape": list(left.shape)})
    for index, (left, right) in enumerate(zip(official_features, our_real_f, strict=True)):
        report["features"].append({"index": index, **_gap(left, right)})
    fake_scores, fake_features = _flatten(fake_s, fake_f)
    report["fake_score_max_abs"] = max(_gap(left, right)["max_abs"] for left, right in zip(fake_scores, our_fake_s, strict=True))
    report["fake_feature_max_abs"] = max(
        _gap(left, right)["max_abs"] for left, right in zip(fake_features, our_fake_f, strict=True)
    )
    return report


def main() -> None:
    torch.manual_seed(3)
    waveform = torch.randn(2, 1, 8192)
    official_cqt = OfficialCQT()
    independent_cqt = MultiScaleSubbandCQTDiscriminator()
    official_mrd = OfficialMRD()
    independent_mrd = MultiResolutionDiscriminator()
    report = {
        "parameters": {
            "cqt_official": _count(official_cqt),
            "cqt_independent": _count(independent_cqt),
            "mrd_official": _count(official_mrd),
            "mrd_independent": _count(independent_mrd),
        },
        "cqt_keys": _load(official_cqt, independent_cqt),
        "mrd_keys": _load(official_mrd, independent_mrd),
    }
    if not report["cqt_keys"]["missing_in_independent"] and not report["mrd_keys"]["missing_in_independent"]:
        report["cqt_eval"] = _compare_pair(official_cqt, independent_cqt, waveform, False)
        report["cqt_train"] = _compare_pair(official_cqt, independent_cqt, waveform, True)
        report["mrd_eval"] = _compare_pair(official_mrd, independent_mrd, waveform, False)
        report["mrd_train"] = _compare_pair(official_mrd, independent_mrd, waveform, True)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
