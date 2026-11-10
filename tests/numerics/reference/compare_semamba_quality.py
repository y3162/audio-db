"""Train official and independent SEMamba++ together and save quality waveforms.

PESQ is not computed here. ``.venvs/semamba`` has no PESQ library, and the
reference loss module imports ``pesq`` at import time. A dummy module is
installed only so that import succeeds. Metric values come from
``semamba_quality.py score`` under ``.venvs/lab``.

    CUDA_VISIBLE_DEVICES=<free> CUBLAS_WORKSPACE_CONFIG=:4096:8 \\
        .venvs/semamba/bin/python tests/numerics/reference/compare_semamba_quality.py \\
        --out data/results/training/semamba-quality-20261101/phase-a --steps 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import torch
from torch import nn

if "pesq" not in sys.modules:
    # Import stub for model.loss. Not a metric. Do not read its return value.
    sys.modules["pesq"] = types.ModuleType("pesq")
    sys.modules["pesq"].pesq = lambda *args, **kwargs: 0.0

ROOT = Path(__file__).resolve().parents[3]
OFFICIAL = ROOT / "src" / "se" / "official" / "semambapp"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OFFICIAL))

from model.discriminator import MultiResolutionDiscriminator as OfficialMRD  # noqa: E402
from model.discriminator import MultiScaleSubbandCQTDiscriminator as OfficialCQT  # noqa: E402
from model.loss import MultiScaleMelSpectrogramLoss  # noqa: E402
from model.semambapp import SEMambapp  # noqa: E402
from src.se.models.semambapp.network import SEMambaPP  # noqa: E402
from src.experiments.official.semambapp.discriminator import (  # noqa: E402
    MultiResolutionDiscriminator,
    MultiScaleSubbandCQTDiscriminator,
)
from tests.numerics.reference.compare_semamba_step import (  # noqa: E402
    BETAS,
    LR,
    WEIGHT_DECAY,
    _cfg,
    _flat,
    _independent_losses,
    _independent_parts,
    _official_losses,
    _official_parts,
)
from tests.numerics.reference.semamba_quality import (  # noqa: E402
    JUDGED_UPDATES,
    SAMPLE_RATE,
    TOLERANCES,
    assert_tensor_roundtrip,
    module_sha256,
    save_json,
    waveform_gap,
)
from src.experiments.calc.rng import capture_rng, restore_rng, seed_model_init, seed_training_stream  # noqa: E402

INIT_SEED = 1234
TRAIN_SEED = 1234
WIDTH = 48
LAYERS = 4
STATE = 16
CONV = 4
SCHEDULER_GAMMA = 0.99
DEFAULT_CHECKPOINTS = (0, 1, 5, 10, 20)


def _require_environment() -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise SystemExit("set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting Python")
    if not torch.cuda.is_available():
        raise SystemExit("SEMamba++ fused kernel comparison requires CUDA")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _kernel_report(generator: torch.nn.Module) -> dict:
    import importlib.metadata as metadata

    import causal_conv1d
    import selective_scan_cuda
    from mamba_ssm.modules.mamba_simple import Mamba

    mambas = [module for module in generator.modules() if isinstance(module, Mamba)]
    if not mambas:
        raise SystemExit("official generator has no Mamba modules")
    if any(not module.use_fast_path for module in mambas):
        raise SystemExit("official Mamba use_fast_path is off; refusing the PyTorch scan fallback")
    if causal_conv1d.causal_conv1d_fn is None:
        raise SystemExit("causal_conv1d fused convolution is unavailable")
    return {
        "mamba_modules": len(mambas),
        "use_fast_path": True,
        "fused_kernel": "mamba_inner_fn",
        "selective_scan_cuda": selective_scan_cuda.__file__,
        "mamba_ssm": metadata.version("mamba-ssm"),
        "causal_conv1d": metadata.version("causal-conv1d"),
        "capability": ".".join(str(part) for part in torch.cuda.get_device_capability()),
        "gpu": torch.cuda.get_device_name(),
    }


def _dropout_count(*modules: torch.nn.Module) -> int:
    return sum(1 for module in modules for layer in module.modules() if isinstance(layer, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)))


def _run_pair(official_fn, independent_fn):
    """Give both implementations the same generator state, then keep one draw."""
    snap = capture_rng(cuda=True)
    official = official_fn()
    after = capture_rng(cuda=True)
    restore_rng(snap, cuda=True)
    independent = independent_fn()
    restore_rng(after, cuda=True)
    return official, independent


def _finite_number(value: torch.Tensor) -> float | None:
    number = float(value.detach())
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _loss_pair(official: dict, independent: dict, names: tuple[str, ...]) -> dict:
    gaps = {}
    official_values = {}
    independent_values = {}
    for name in names:
        left = _finite_number(official[name])
        right = _finite_number(independent[name])
        official_values[name] = left
        independent_values[name] = right
        gaps[name] = None if left is None or right is None else abs(left - right)
    return {"gap": gaps, "official": official_values, "independent": independent_values}


def _same_weights(left: torch.nn.Module, right: torch.nn.Module) -> bool:
    official = left.state_dict()
    independent = right.state_dict()
    if official.keys() != independent.keys():
        return False
    return all(torch.equal(official[name].detach().cpu(), independent[name].detach().cpu()) for name in official)


def _build(device: torch.device, out: Path):
    cfg = _cfg(WIDTH, LAYERS, STATE, CONV)
    with seed_model_init(INIT_SEED):
        official_g = SEMambapp(cfg)
    with seed_model_init(INIT_SEED):
        independent_g = SEMambaPP(_flat(cfg))
    with seed_model_init(INIT_SEED):
        official_cqt = OfficialCQT()
        official_mrd = OfficialMRD()
    with seed_model_init(INIT_SEED):
        independent_cqt = MultiScaleSubbandCQTDiscriminator()
        independent_mrd = MultiResolutionDiscriminator()
    before = {
        "generator": [module_sha256(official_g), module_sha256(independent_g)],
        "cqt": [module_sha256(official_cqt), module_sha256(independent_cqt)],
        "mrd": [module_sha256(official_mrd), module_sha256(independent_mrd)],
    }
    independent_g.load_state_dict(official_g.state_dict(), strict=True)
    independent_cqt.load_state_dict(official_cqt.state_dict(), strict=True)
    independent_mrd.load_state_dict(official_mrd.state_dict(), strict=True)
    modules = {
        "official_g": official_g.to(device),
        "independent_g": independent_g.to(device),
        "official_cqt": official_cqt.to(device),
        "independent_cqt": independent_cqt.to(device),
        "official_mrd": official_mrd.to(device),
        "independent_mrd": independent_mrd.to(device),
    }
    pairs = (
        ("generator", modules["official_g"], modules["independent_g"]),
        ("cqt", modules["official_cqt"], modules["independent_cqt"]),
        ("mrd", modules["official_mrd"], modules["independent_mrd"]),
    )
    for name, left, right in pairs:
        if not _same_weights(left, right):
            raise SystemExit(f"initial {name} weights differ after copy")
    after = {name: module_sha256(left) for name, left, _right in pairs}
    copied = {name: module_sha256(right) for name, _left, right in pairs}
    if after != copied:
        raise SystemExit(f"initial weight hashes differ after copy: {after} vs {copied}")
    identity = {
        "init_seed": INIT_SEED,
        "before_copy_sha256": before,
        "sha256": after,
        "copied_matches": True,
        "dropout_modules": _dropout_count(*modules.values()),
        "generator_parameters": sum(parameter.numel() for parameter in modules["official_g"].parameters()),
    }
    save_json(out / "initial_weights.json", identity)
    return cfg, modules, identity


def _optimizers(modules: dict):
    official_g = torch.optim.AdamW(modules["official_g"].parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)
    independent_g = torch.optim.AdamW(modules["independent_g"].parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)
    official_d = torch.optim.AdamW(
        list(modules["official_cqt"].parameters()) + list(modules["official_mrd"].parameters()),
        lr=LR,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
    )
    independent_d = torch.optim.AdamW(
        list(modules["independent_cqt"].parameters()) + list(modules["independent_mrd"].parameters()),
        lr=LR,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
    )
    schedulers = [
        torch.optim.lr_scheduler.ExponentialLR(optim, gamma=SCHEDULER_GAMMA)
        for optim in (official_g, independent_g, official_d, independent_d)
    ]
    return {
        "official_g": official_g,
        "independent_g": independent_g,
        "official_d": official_d,
        "independent_d": independent_d,
        "schedulers": schedulers,
    }


def _state_max_abs(reference: dict[str, torch.Tensor], module: torch.nn.Module) -> float:
    current = module.state_dict()
    worst = 0.0
    for name, left in reference.items():
        delta = (left.detach().float().cpu() - current[name].detach().float().cpu()).abs().max()
        worst = max(worst, float(delta))
    return worst


def _checkpoint(path: Path, modules: dict, optimizers: dict, updates: int, experiment: dict, rng: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    official = "official" in path.name
    torch.save(
        {
            "updates_completed": updates,
            "generator": modules["official_g"].state_dict() if official else modules["independent_g"].state_dict(),
            "cqt": modules["official_cqt"].state_dict() if official else modules["independent_cqt"].state_dict(),
            "mrd": modules["official_mrd"].state_dict() if official else modules["independent_mrd"].state_dict(),
            "optim_g": optimizers["official_g"].state_dict() if official else optimizers["independent_g"].state_dict(),
            "optim_d": optimizers["official_d"].state_dict() if official else optimizers["independent_d"].state_dict(),
            "scheduler_g": optimizers["schedulers"][0].state_dict() if official else optimizers["schedulers"][1].state_dict(),
            "scheduler_d": optimizers["schedulers"][2].state_dict() if official else optimizers["schedulers"][3].state_dict(),
            "rng": rng,
            "experiment": experiment,
            "generator_sha256": module_sha256(modules["official_g"] if official else modules["independent_g"]),
        },
        path,
    )


def _save_pair(out: Path, modules: dict, optimizers: dict, updates: int, experiment: dict, rng: dict) -> dict[str, str]:
    official = out / "checkpoints" / f"step_{updates:04d}_official.pt"
    independent = out / "checkpoints" / f"step_{updates:04d}_independent.pt"
    _checkpoint(official, modules, optimizers, updates, experiment, rng)
    _checkpoint(independent, modules, optimizers, updates, experiment, rng)
    return {"official": str(official.relative_to(out)), "independent": str(independent.relative_to(out))}


def _enhance(out: Path, modules: dict, cfg: dict, validation: list[dict], updates: int, checkpoints: dict[str, str], device: torch.device) -> None:
    tracked = (
        modules["official_g"],
        modules["independent_g"],
        modules["official_cqt"],
        modules["independent_cqt"],
        modules["official_mrd"],
        modules["independent_mrd"],
    )
    training = [module.training for module in tracked]
    for module in tracked:
        module.eval()
    rows = []
    with torch.no_grad():
        for utterance in validation:
            print(f"enhance step {updates} {utterance['case_id'][:12]} {utterance['seconds']:.3f}s", flush=True)
            clean = utterance["clean"].to(device).view(1, -1)
            noisy = utterance["noisy"].to(device).view(1, -1)
            official = _official_parts(modules["official_g"], clean, noisy, cfg)["audio"].squeeze(0).detach().float().cpu()
            independent = _independent_parts(modules["independent_g"], clean, noisy)["audio"].squeeze(0).detach().float().cpu()
            torch.cuda.empty_cache()
            rows.append(
                {
                    "case_id": utterance["case_id"],
                    "source_case_id": utterance["source_case_id"],
                    "seconds": utterance["seconds"],
                    "snr_db": utterance["snr_db"],
                    "clean": utterance["clean"].detach().float().cpu(),
                    "noisy": utterance["noisy"].detach().float().cpu(),
                    "official": official,
                    "independent": independent,
                    "wave": waveform_gap(official, independent),
                }
            )
    for module, was in zip(tracked, training, strict=True):
        module.train(was)
    payload = {
        "updates_completed": updates,
        "sample_rate": SAMPLE_RATE,
        "mode": "eval",
        "clipping": None,
        "resampled": False,
        "normalization": "training_energy_v1",
        "checkpoints": checkpoints,
        "generator_sha256": {
            "official": module_sha256(modules["official_g"]),
            "independent": module_sha256(modules["independent_g"]),
        },
        "utterances": rows,
    }
    digest = assert_tensor_roundtrip(out / "waves" / f"step_{updates:04d}.pt", payload)
    print(f"saved step {updates} waveforms {digest[:12]}", flush=True)


def _one_update(modules, optimizers, cfg, clean, noisy, mel, device):
    clean = clean.to(device)
    noisy = noisy.to(device)
    for module in modules.values():
        if isinstance(module, torch.nn.Module):
            module.train()

    def official_forward():
        return _official_parts(modules["official_g"], clean, noisy, cfg)

    def independent_forward():
        return _independent_parts(modules["independent_g"], clean, noisy)

    official_parts, independent_parts = _run_pair(official_forward, independent_forward)
    for optim in (optimizers["official_d"], optimizers["independent_d"], optimizers["official_g"], optimizers["independent_g"]):
        optim.zero_grad(set_to_none=True)
    official_d, independent_d = _run_pair(
        lambda: _official_losses({**official_parts, "audio": official_parts["audio"].detach()}, clean, modules["official_cqt"], modules["official_mrd"], mel, cfg)["discriminator"],
        lambda: _independent_losses({**independent_parts, "audio": independent_parts["audio"].detach()}, clean, modules["independent_cqt"], modules["independent_mrd"])["discriminator"],
    )
    official_d.backward()
    independent_d.backward()
    optimizers["official_d"].step()
    optimizers["independent_d"].step()
    for optim in (optimizers["official_g"], optimizers["independent_g"]):
        optim.zero_grad(set_to_none=True)
    official_parts, independent_parts = _run_pair(official_forward, independent_forward)
    official_loss, independent_loss = _run_pair(
        lambda: _official_losses(official_parts, clean, modules["official_cqt"], modules["official_mrd"], mel, cfg),
        lambda: _independent_losses(independent_parts, clean, modules["independent_cqt"], modules["independent_mrd"]),
    )
    official_loss["generator"].backward()
    independent_loss["generator"].backward()
    optimizers["official_g"].step()
    optimizers["independent_g"].step()
    names = ("generator", "discriminator", "adversarial", "matching", "mel", "magnitude", "phase", "complex", "consistency")
    losses = _loss_pair(official_loss, independent_loss, names)
    losses["official"]["discriminator"] = _finite_number(official_d)
    losses["independent"]["discriminator"] = _finite_number(independent_d)
    if losses["official"]["discriminator"] is not None and losses["independent"]["discriminator"] is not None:
        losses["gap"]["discriminator"] = abs(losses["official"]["discriminator"] - losses["independent"]["discriminator"])
    return {
        "losses": losses,
        "wave": waveform_gap(official_parts["audio"], independent_parts["audio"]),
        "clean": clean.squeeze(0).detach().float().cpu(),
        "noisy": noisy.squeeze(0).detach().float().cpu(),
        "official": official_parts["audio"].squeeze(0).detach().float().cpu(),
        "independent": independent_parts["audio"].squeeze(0).detach().float().cpu(),
        "generator_sha256": {
            "official": module_sha256(modules["official_g"]),
            "independent": module_sha256(modules["independent_g"]),
        },
    }


def _load_training_batches(steps: int) -> list[dict]:
    path = ROOT / "data" / "results" / "training" / "official-compare-20261101" / "semamba_waves.pt"
    saved = torch.load(path, map_location="cpu", weights_only=True)
    count = len(saved["clean"])
    rows = []
    for index in range(steps):
        slot = index % count
        rows.append(
            {
                "clean": saved["clean"][slot],
                "noisy": saved["noisy"][slot],
                "case_id": saved["case_id"][slot],
                "slot": slot,
            }
        )
    return rows


def _resume_probe(out: Path, device: torch.device) -> dict:
    """Reload the step-10 checkpoint and compare one further update with the saved step-11 weights."""
    probe_path = out / "probe_step11.pt"
    checkpoint = out / "checkpoints" / "step_0010_official.pt"
    other = out / "checkpoints" / "step_0010_independent.pt"
    if not probe_path.is_file() or not checkpoint.is_file():
        return {"status": "na", "reason": "probe_not_saved"}
    probe = torch.load(probe_path, map_location="cpu", weights_only=False)
    torch.cuda.empty_cache()
    cfg, modules, _identity = _build(device, out / "resume_rebuild")
    official = torch.load(checkpoint, map_location="cpu", weights_only=False)
    independent = torch.load(other, map_location="cpu", weights_only=False)
    modules["official_g"].load_state_dict(official["generator"])
    modules["independent_g"].load_state_dict(independent["generator"])
    modules["official_cqt"].load_state_dict(official["cqt"])
    modules["independent_cqt"].load_state_dict(independent["cqt"])
    modules["official_mrd"].load_state_dict(official["mrd"])
    modules["independent_mrd"].load_state_dict(independent["mrd"])
    optimizers = _optimizers(modules)
    optimizers["official_g"].load_state_dict(official["optim_g"])
    optimizers["independent_g"].load_state_dict(independent["optim_g"])
    optimizers["official_d"].load_state_dict(official["optim_d"])
    optimizers["independent_d"].load_state_dict(independent["optim_d"])
    optimizers["schedulers"][0].load_state_dict(official["scheduler_g"])
    optimizers["schedulers"][1].load_state_dict(independent["scheduler_g"])
    optimizers["schedulers"][2].load_state_dict(official["scheduler_d"])
    optimizers["schedulers"][3].load_state_dict(independent["scheduler_d"])
    mel = MultiScaleMelSpectrogramLoss(sampling_rate=SAMPLE_RATE).to(device)
    restore_rng(official["rng"], cuda=True)
    batch = _load_training_batches(11)[10]
    _one_update(modules, optimizers, cfg, batch["clean"], batch["noisy"], mel, device)
    result = {
        "status": "ok",
        "from_updates": 10,
        "compared_updates": 11,
        "official_generator_max_abs": _state_max_abs(probe["official"], modules["official_g"]),
        "independent_generator_max_abs": _state_max_abs(probe["independent"], modules["independent_g"]),
        "scheduler_last_epoch": [int(item.last_epoch) for item in optimizers["schedulers"]],
    }
    save_json(out / "resume_check.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=JUDGED_UPDATES)
    parser.add_argument("--checkpoints", type=str, default=",".join(str(item) for item in DEFAULT_CHECKPOINTS))
    parser.add_argument("--skip-resume-check", action="store_true")
    args = parser.parse_args()
    _require_environment()
    if args.steps < 1:
        raise SystemExit("--steps must be positive")
    checkpoints = tuple(sorted({int(item) for item in args.checkpoints.split(",") if item}))
    if any(item < 0 or item > args.steps for item in checkpoints):
        raise SystemExit("checkpoints must lie between 0 and --steps")
    out = args.out
    if not (out / "validation.pt").is_file():
        raise SystemExit(f"missing {out / 'validation.pt'}; run semamba_quality.py prepare first")
    device = torch.device("cuda:0")
    validation_blob = torch.load(out / "validation.pt", map_location="cpu", weights_only=False)
    validation = validation_blob["utterances"]
    batches = _load_training_batches(args.steps)
    train_ids = {item["case_id"] for item in batches}
    for utterance in validation:
        if utterance["source_case_id"] in train_ids or utterance["case_id"] in train_ids:
            raise SystemExit("validation audio is in the training batches")
    cfg, modules, identity = _build(device, out)
    kernel = _kernel_report(modules["official_g"])
    optimizers = _optimizers(modules)
    mel = MultiScaleMelSpectrogramLoss(sampling_rate=SAMPLE_RATE).to(device)
    seed_training_stream(TRAIN_SEED, cuda=True)
    experiment = {
        "init_seed": INIT_SEED,
        "train_seed": TRAIN_SEED,
        "lr": LR,
        "betas": list(BETAS),
        "weight_decay": WEIGHT_DECAY,
        "scheduler": {"name": "exponential", "gamma": SCHEDULER_GAMMA, "stepped": False, "reason": "official ExponentialLR steps once per epoch"},
        "grad_clip": None,
        "update_order": "discriminator on detached audio, then a fresh generator forward and generator step",
        "batch": "libri_demand_smoke_train 8000-sample crops from official-compare-20261101/semamba_waves.pt",
        "dropout_modules": identity["dropout_modules"],
        "kernel": kernel,
        "tolerances": TOLERANCES,
        "pesq": "not computed in this process",
    }
    config_path = out / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    config["training"] = experiment
    config["steps"] = args.steps
    config["checkpoints"] = list(checkpoints)
    save_json(config_path, config)
    log_path = out / "train_log.jsonl"
    train_rows = []
    probe = None
    started = time.perf_counter()
    if 0 in checkpoints:
        rng_after = capture_rng(cuda=True)
        names = _save_pair(out, modules, optimizers, 0, experiment, rng_after)
        _enhance(out, modules, cfg, validation, 0, names, device)
        restore_rng(rng_after, cuda=True)
    for index in range(args.steps):
        step_started = time.perf_counter()
        batch = batches[index]
        result = _one_update(modules, optimizers, cfg, batch["clean"], batch["noisy"], mel, device)
        updates = index + 1
        record = {
            "updates_completed": index,
            "batch_case_id": batch["case_id"],
            "seconds": time.perf_counter() - step_started,
            "losses": result["losses"],
            "wave": result["wave"],
            "generator_sha256": result["generator_sha256"],
            "nonfinite": any(
                value is None
                for side in (result["losses"]["official"], result["losses"]["independent"])
                for value in side.values()
            ),
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        train_rows.append(
            {
                "updates_completed": index,
                "case_id": batch["case_id"],
                "clean": result["clean"],
                "noisy": result["noisy"],
                "official": result["official"],
                "independent": result["independent"],
                "wave": result["wave"],
            }
        )
        print(
            f"update {updates}/{args.steps} loss_gap={result['losses']['gap']['generator']} cosine={result['wave'].get('cosine')} ({record['seconds']:.1f}s)",
            flush=True,
        )
        if updates == 11:
            probe = {
                "official": {key: value.detach().cpu().clone() for key, value in modules["official_g"].state_dict().items()},
                "independent": {key: value.detach().cpu().clone() for key, value in modules["independent_g"].state_dict().items()},
            }
            torch.save(probe, out / "probe_step11.pt")
        if updates in checkpoints:
            rng_after = capture_rng(cuda=True)
            names = _save_pair(out, modules, optimizers, updates, experiment, rng_after)
            _enhance(out, modules, cfg, validation, updates, names, device)
            restore_rng(rng_after, cuda=True)
        if record["nonfinite"]:
            raise SystemExit(f"non-finite loss at update {updates}")
    assert_tensor_roundtrip(out / "train_waves.pt", {"steps": train_rows, "sample_rate": SAMPLE_RATE, "quality_source": False})
    elapsed = time.perf_counter() - started
    summary = {
        "steps": args.steps,
        "seconds": elapsed,
        "seconds_per_step": elapsed / args.steps,
        "max_memory_mib": torch.cuda.max_memory_allocated(device) / (1024 * 1024),
        "scheduler_last_epoch": [scheduler.last_epoch for scheduler in optimizers["schedulers"]],
        "kernel": kernel,
        "initial_weights": identity["sha256"],
        "generator_loss_gap": [row["losses"]["gap"]["generator"] for row in _read_log(log_path)],
        "train_wave_cosine": [row["wave"].get("cosine") for row in train_rows],
    }
    save_json(out / "train_summary.json", summary)
    print(json.dumps({"seconds": elapsed, "seconds_per_step": summary["seconds_per_step"]}, indent=2), flush=True)
    if not args.skip_resume_check and args.steps >= 11 and 10 in checkpoints:
        del modules, optimizers, mel
        torch.cuda.empty_cache()
        resume = _resume_probe(out, device)
        print(json.dumps({"resume": resume}, indent=2), flush=True)


def _read_log(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    main()
