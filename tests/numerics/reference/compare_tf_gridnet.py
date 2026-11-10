"""TF-GridNet ms-snsd training versus the ESPnet separator and ``mr_l1_tfd``.

Tolerances, fixed before reading measurements:

- loss on identical waveforms: atol 1e-5
- separator forward: atol 1e-5
- parameter gradients: atol 1e-4, cosine >= 0.999 when |grad| max > 1e-5

The ms-snsd recipe uses Adam (lr 1e-3, betas 0.9/0.999, eps 1e-8, weight decay 0),
``mr_l1_tfd`` windows (256, 512, 768, 1024) without variance normalization, and
SI-SNR weight 0. ESPnet's default gradient clip is 5.0 because that yaml does
not set ``grad_clip``. The urgent24 yaml in this repo is a different recipe.

    CUDA_VISIBLE_DEVICES=<free> CUBLAS_WORKSPACE_CONFIG=:4096:8 \\
        .venvs/lab/bin/python tests/numerics/reference/compare_tf_gridnet.py
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import torch

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.se import build_model, load_enhancer  # noqa: E402
from src.experiments.calc.spectral import multi_resolution_l1  # noqa: E402
from src.experiments.official.tf_gridnet.reference import TFGridNetTask  # noqa: E402
from tests.se.test_tf_gridnet import reference_v1  # noqa: E402
from tests.numerics.reference.common import (  # noqa: E402
    configured_steps,
    environment,
    gap,
    grad_report,
    load_cached,
    optim_report,
    reference_scores,
    REPORT_DIR,
    save_report,
    weight_report,
)

LOSS_ATOL = 1e-5
GRAD_ATOL = 1e-4
WINDOWS = (256, 512, 768, 1024)
LR = 1e-3
BETAS = (0.9, 0.999)
CLIP = 5.0
ARCHITECTURE = {
    "n_fft": 512,
    "stride": 256,
    "n_layers": 4,
    "lstm_hidden_units": 128,
    "attn_n_head": 4,
    "attn_approx_qk_dim": 512,
    "emb_dim": 32,
    "emb_ks": 4,
    "emb_hs": 4,
}


def _official_loss_class():
    root = ROOT / "src" / "se" / "official" / "espnet"
    def package(name: str, path: Path) -> None:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        module.__package__ = name
        sys.modules[name] = module
    package("espnet2", root / "espnet2")
    package("espnet2.enh", root / "espnet2" / "enh")
    package("espnet2.enh.layers", root / "espnet2" / "enh" / "layers")
    package("espnet2.enh.loss", root / "espnet2" / "enh" / "loss")
    package("espnet2.enh.loss.criterions", root / "espnet2" / "enh" / "loss" / "criterions")
    package("espnet2.layers", root / "espnet2" / "layers")
    package("espnet2.legacy", root / "espnet2" / "legacy")
    package("espnet2.legacy.nets", root / "espnet2" / "legacy" / "nets")
    package("espnet2.legacy.nets.pytorch_backend", root / "espnet2" / "legacy" / "nets" / "pytorch_backend")
    torch_complex = types.ModuleType("torch_complex")
    torch_complex.functional = types.ModuleType("torch_complex.functional")
    torch_complex.tensor = types.ModuleType("torch_complex.tensor")
    torch_complex.tensor.ComplexTensor = object
    sys.modules["torch_complex"] = torch_complex
    sys.modules["torch_complex.functional"] = torch_complex.functional
    sys.modules["torch_complex.tensor"] = torch_complex.tensor
    typeguard = types.ModuleType("typeguard")
    typeguard.typechecked = lambda fn: fn
    sys.modules["typeguard"] = typeguard
    sys.path.insert(0, str(root))
    from espnet2.enh.loss.criterions.time_domain import MultiResL1SpecLoss

    return MultiResL1SpecLoss


def _pair(device: torch.device):
    official_cls = reference_v1()
    official = official_cls(input_dim=0, n_srcs=1, use_builtin_complex=True, **ARCHITECTURE).to(device)
    independent = build_model("tf_gridnet", ARCHITECTURE, variant="ms-snsd").to(device)
    independent.load_state_dict(official.state_dict(), strict=True)
    return official, independent


def _adam(module: torch.nn.Module) -> torch.optim.Adam:
    return torch.optim.Adam(module.parameters(), lr=LR, betas=BETAS, eps=1e-8, weight_decay=0.0)


def _official_value(criterion, clean: torch.Tensor, estimate: torch.Tensor) -> torch.Tensor:
    # fixed_order wraps the per-utterance vector with a batch mean. SI-SNR weight is 0.
    return criterion(clean, estimate).mean()


def _enhance(official, independent, noisy: torch.Tensor):
    lengths = torch.tensor([noisy.shape[-1]] * noisy.shape[0], device=noisy.device)
    official_wave = official(noisy, lengths)[0][0]
    independent_wave = independent(noisy)
    return official_wave, independent_wave


def _step(official, independent, optim_o, optim_i, criterion, clean, noisy) -> dict:
    official.train()
    independent.train()
    optim_o.zero_grad(set_to_none=True)
    optim_i.zero_grad(set_to_none=True)
    official_wave, independent_wave = _enhance(official, independent, noisy)
    official_loss = _official_value(criterion, clean, official_wave)
    independent_loss = multi_resolution_l1(clean, independent_wave, WINDOWS, normalize_variance=False)
    same_wave_gap = abs(
        float(_official_value(criterion, clean, official_wave.detach()).detach())
        - float(multi_resolution_l1(clean, official_wave.detach(), WINDOWS, normalize_variance=False).detach())
    )
    official_loss.backward()
    independent_loss.backward()
    before_clip = grad_report(official, independent, atol=GRAD_ATOL)
    torch.nn.utils.clip_grad_norm_(official.parameters(), CLIP)
    torch.nn.utils.clip_grad_norm_(independent.parameters(), CLIP)
    optim_o.step()
    optim_i.step()
    return {
        "forward": gap(official_wave, independent_wave),
        "loss_gap": abs(float(official_loss.detach()) - float(independent_loss.detach())),
        "official_loss": float(official_loss.detach()),
        "independent_loss": float(independent_loss.detach()),
        "loss_on_official_wave": same_wave_gap,
        "grad": before_clip,
        "weight": weight_report(official, independent),
        "optim": optim_report(optim_o, optim_i),
    }


def _task_config(device_name: str) -> dict:
    return {
        "seed": 7,
        "loss": "official",
        "grad_clip": CLIP,
        "optimizer": {"name": "adam", "lr": LR, "betas": list(BETAS), "weight_decay": 0.0},
        "scheduler": {"name": "plateau", "factor": 0.7, "patience": 1},
        "model": {"variant": "ms-snsd", "sampling_rate": 16000, "architecture": ARCHITECTURE},
    }


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = _official_loss_class()(
        window_sz=list(WINDOWS),
        hop_sz=None,
        eps=1e-8,
        time_domain_weight=0.5,
        normalize_variance=False,
        reduction="sum",
    ).to(device)
    report = {
        "environment": environment(device),
        "official_commit": "0d136c5456d403cb2c0a4c13e76e8136e32e9b7a",
        "variant": "ms-snsd",
        "architecture": ARCHITECTURE,
        "optimizer": {"name": "adam", "lr": LR, "betas": list(BETAS), "eps": 1e-8, "weight_decay": 0.0},
        "grad_clip": CLIP,
        "scheduler": "ReduceLROnPlateau is per validation epoch and is not stepped in these 20 updates",
        "si_snr_weight": 0.0,
        "tolerances": {"loss_atol": LOSS_ATOL, "grad_atol": GRAD_ATOL, "large_cosine_min": 0.999},
        "segment_size": 16000,
        "batch_size": 1,
        "note": "Official chunk_length is 48000 and batch_size is 4. This run uses 1 second and batch 1.",
    }
    torch.manual_seed(3)
    probe_wave = torch.randn(2, 2048, device=device)
    report["loss_identity"] = {
        "max_abs": abs(float(_official_value(criterion, probe_wave, probe_wave * 0.5).detach())
        - float(multi_resolution_l1(probe_wave, probe_wave * 0.5, WINDOWS).detach()))
    }
    official, independent = _pair(device)
    report["parameters"] = sum(parameter.numel() for parameter in independent.parameters())
    from src.experiments.official.tf_gridnet.model import TFGridNetModel

    yaml_model = TFGridNetModel(seed=7, architecture=ARCHITECTURE, variant="ms-snsd")
    yaml_model.generator.load_state_dict(official.state_dict(), strict=True)
    yaml_model.to(device).eval()
    official.eval()
    independent.eval()
    with torch.no_grad():
        yaml_wave = yaml_model(probe_wave)
        official_wave, independent_wave = _enhance(official, independent, probe_wave)
    report["yaml_model_forward"] = {
        "class": "src.experiments.official.tf_gridnet.model.TFGridNetModel",
        "versus_official": gap(official_wave, yaml_wave),
        "versus_build_model": gap(independent_wave, yaml_wave),
    }
    official.train()
    independent.train()
    try:
        train = load_cached("tf_train")
        valid = load_cached("valid")
        report["input_protocol"] = "historical_speech_cache_tf_train"
    except FileNotFoundError as exc:
        generator = torch.Generator().manual_seed(3)
        clean = torch.randn(4, 4096, generator=generator)
        noisy = clean * 0.5 + 0.05 * torch.randn(4, 4096, generator=generator)
        train = {"clean": clean, "noisy": noisy, "case_id": [f"synthetic-tf-{i}" for i in range(4)]}
        valid = {"clean": clean[:1], "noisy": noisy[:1], "case_id": ["synthetic-tf-valid"]}
        report["input_protocol"] = "synthetic_fixed_seed"
        report["input_protocol_detail"] = (
            "torch.Generator seed 3; 4 utterances x 4096 samples. "
            "Not speech_cache.pt and not training.duckdb. Shorter than the official 48000-sample chunk."
        )
        report["cache_missing"] = str(exc)
    report["train_case_ids"] = train["case_id"]
    report["valid_case_id"] = valid["case_id"]
    first = {"clean": train["clean"][:1], "noisy": train["noisy"][:1], "case_id": [train["case_id"][0]]}
    task = TFGridNetTask(_task_config(str(device)), device)
    task.generator.load_state_dict(independent.state_dict(), strict=True)
    task.optim_g = _adam(task.generator)
    held = {name: value.detach().clone() for name, value in official.state_dict().items()}
    task_metrics = task.train_update([{"clean": first["clean"].to(device), "noisy": first["noisy"].to(device)}], 1)
    official.load_state_dict(held, strict=True)
    independent.load_state_dict(held, strict=True)
    manual = _step(official, independent, _adam(official), _adam(independent), criterion, first["clean"].to(device), first["noisy"].to(device))
    report["task_versus_manual"] = {
        "task_loss": task_metrics["mr_l1_tfd"],
        "manual_independent_loss": manual["independent_loss"],
        "weight": weight_report(independent, task.generator),
    }
    official, independent = _pair(device)
    optim_o, optim_i = _adam(official), _adam(independent)
    history = []
    steps = configured_steps(20)
    report["steps"] = steps
    for step in range(steps):
        index = step % train["clean"].shape[0]
        batch = {
            "clean": train["clean"][index : index + 1],
            "noisy": train["noisy"][index : index + 1],
            "case_id": [train["case_id"][index]],
        }
        record = _step(
            official,
            independent,
            optim_o,
            optim_i,
            criterion,
            batch["clean"].to(device),
            batch["noisy"].to(device),
        )
        history.append(
            {
                "step": step,
                "case_id": batch["case_id"],
                "official_loss": record["official_loss"],
                "independent_loss": record["independent_loss"],
                "loss_gap": record["loss_gap"],
                "loss_on_official_wave": record["loss_on_official_wave"],
                "wave_max_abs": record["forward"]["max_abs"],
                "grad_max_abs": record["grad"]["max_abs"],
                "large_cosine": record["grad"]["large_cosine_min"],
                "weight_max_abs": record["weight"]["max_abs"],
            }
        )
        report["history"] = history
        report["last_step"] = record
        save_report("tf_gridnet.json", report)
        print(f"tf-gridnet step {step} loss {record['official_loss']:.6f} gap {record['loss_gap']:.6e}", flush=True)
    official.eval()
    independent.eval()
    with torch.no_grad():
        official_wave, independent_wave = _enhance(official, independent, valid["noisy"].to(device))
    clean = valid["clean"][0].cpu()
    report["probe_case_id"] = valid["case_id"]
    report["probe"] = {
        "wave": gap(official_wave, independent_wave),
        "official": reference_scores(clean, official_wave[0], 16000),
        "independent": reference_scores(clean, independent_wave[0], 16000),
    }
    exported = REPORT_DIR / "tf_gridnet_generator.pt"
    torch.save({"separator." + name: value.detach().cpu() for name, value in independent.state_dict().items()}, exported)
    enhancer = load_enhancer("tf_gridnet", {"model": str(exported)}, device=str(device), config=ARCHITECTURE, variant="ms-snsd")
    with torch.no_grad():
        reloaded = enhancer.enhance(valid["noisy"][0].detach().cpu(), 16000).to(device)
    report["reload"] = gap(independent_wave[0].cpu(), reloaded.cpu())
    path = save_report("tf_gridnet.json", report)
    print(path)


if __name__ == "__main__":
    main()
