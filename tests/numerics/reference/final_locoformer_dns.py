"""Three DNS-width TF-Locoformer steps against the official separator.

The architecture is ``egs2/dns_ins20``: 6 layers, emb_dim 128, ffn 384.
The official chunk is 64000 samples and batch 4. This run uses batch 1 and
4096 samples so the width can be compared on one GPU. It is not the official
chunk length.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get("AUDIO_DB_NUMERICS_OUT", "/tmp/audio-db-official-final"))
sys.path.insert(0, str(ROOT))

from src.experiments.calc.rng import seed_model_init  # noqa: E402
from src.experiments.calc.spectral import multi_resolution_l1  # noqa: E402
from src.experiments.official.tf_locoformer.reference import TFLocoformerTask  # noqa: E402
from tests.se.test_tf_locoformer import OFFICIAL, _frontend  # noqa: E402

DNS = dict(
    num_spk=1,
    n_layers=6,
    emb_dim=128,
    norm_type="rmsgroupnorm",
    num_groups=4,
    tf_order="ft",
    n_heads=4,
    flash_attention=False,
    attention_dim=128,
    pos_enc="rope",
    ffn_type=["swiglu_conv1d", "swiglu_conv1d"],
    ffn_hidden_dim=[384, 384],
    conv1d_kernel=4,
    conv1d_shift=1,
    dropout=0.0,
    eps=1.0e-5,
    n_fft=256,
    hop_length=128,
    normalize_variance=True,
)


def _separator():
    root = str(OFFICIAL)
    if root not in sys.path:
        sys.path.insert(0, root)
    from standalone.tflocoformer_separator import TFLocoformerSeparator

    options = {key: value for key, value in DNS.items() if key not in ("n_fft", "hop_length", "normalize_variance")}
    return TFLocoformerSeparator(**options)


def _max_abs(left, right) -> dict[str, float | str]:
    named = dict(right.named_parameters())
    worst = 0.0
    worst_name = ""
    for name, value in left.named_parameters():
        delta = float((value.detach() - named[name].detach()).abs().max())
        if delta > worst:
            worst = delta
            worst_name = name
    return {"max_abs": worst, "worst": worst_name}


def _grad(left, right) -> dict[str, float | str]:
    named = dict(right.named_parameters())
    worst = 0.0
    worst_name = ""
    for name, value in left.named_parameters():
        if value.grad is None or named[name].grad is None:
            continue
        delta = float((value.grad.detach() - named[name].grad.detach()).abs().max())
        if delta > worst:
            worst = delta
            worst_name = name
    return {"max_abs": worst, "worst": worst_name}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(3)
    official = _separator().to(device)
    architecture = dict(DNS)
    config = {
        "seed": 0,
        "loss": "official",
        "grad_clip": 5.0,
        "optimizer": {"name": "adamw", "lr": 1e-3, "betas": [0.9, 0.999], "weight_decay": 0.01},
        "scheduler": {"name": "none"},
        "model": {
            "name": "tf_locoformer",
            "variant": "dns",
            "sampling_rate": 16000,
            "architecture": architecture,
            "stft": {},
            "components": {},
        },
    }
    with seed_model_init(0):
        task = TFLocoformerTask(config, device)
    task.generator.separator.load_state_dict(official.state_dict(), strict=True)
    official_opt = torch.optim.AdamW(official.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)
    length = int(os.environ.get("AUDIO_DB_LOCO_SAMPLES", "64000"))
    torch.manual_seed(11)
    rows = []
    for step in range(3):
        clean = (0.2 * torch.sin(2 * torch.pi * (120 + 10 * step) * torch.arange(length, device=device) / 16000)).unsqueeze(0)
        noisy = clean * 0.8 + 0.05 * torch.sin(2 * torch.pi * (380 + 15 * step) * torch.arange(length, device=device) / 16000).unsqueeze(0)
        official.train()
        official_opt.zero_grad(set_to_none=True)
        official_wave = _frontend(official, noisy, 256, 128)
        official_loss = multi_resolution_l1(clean, official_wave, time_domain_weight=0.5, normalize_variance=False)
        official_loss.backward()
        torch.nn.utils.clip_grad_norm_(official.parameters(), 5.0)
        captured = {}
        original = task.optim_g.step

        def step_and_capture(*args, **kwargs):
            captured["grad"] = _grad(official, task.generator.separator)
            return original(*args, **kwargs)

        task.optim_g.step = step_and_capture
        metrics = task.train_update([{"clean": clean, "noisy": noisy}], 1)
        task.optim_g.step = original
        official_opt.step()
        rows.append({
            "step": step + 1,
            "loss_gap": abs(float(metrics["mr_l1_tfd"]) - float(official_loss.detach())),
            "grad": captured.get("grad"),
            "weight": _max_abs(official, task.generator.separator),
        })
        print("loco", rows[-1], flush=True)
    report = {
        "device": str(device),
        "official_commit": "7a615460d347ff7334a13dbb831d16280da72cdc",
        "source": "egs2/dns_ins20/enh1/conf/tuning/train_enh_tflocoformer.yaml",
        "width": "dns medium: n_layers=6 emb_dim=128 ffn=384 attention_dim=128",
        "batch_size": 1,
        "samples": length,
        "official_chunk_length": 64000,
        "official_batch_size": 4,
        "official_function": "standalone.tflocoformer_separator.TFLocoformerSeparator",
        "loss": "multi_resolution_l1 windows 256/512/768/1024, time_domain_weight 0.5",
        "steps": rows,
    }
    path = OUT / "locoformer_dns.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
