"""DeepFilterNet3: three official DfNet steps, then one call of ``df.train.run_epoch``.

``libdfdata`` is not importable here. The import is stubbed so the official
function can be called. ``summary_write`` is replaced with a no-op because it
needs the libdf synthesis logger and would write waveforms. The forward, loss,
backward, clip, and AdamW step stay the official function.
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get("AUDIO_DB_NUMERICS_OUT", "/tmp/audio-db-official-final"))
sys.path.insert(0, str(ROOT))
libdf_dir = os.environ.get("AUDIO_DB_LIBDF_PY", "/tmp/audio-db-official-numerics/libdf-py")
if libdf_dir not in sys.path:
    sys.path.insert(0, libdf_dir)

from tests.numerics.reference import compare_deepfilternet as cmp  # noqa: E402
from tests.numerics.reference.common import grad_report, optim_report, weight_report  # noqa: E402


def _steps(device) -> dict:
    MultiResSpecLoss, LocalSnrLoss, local_snr, _cosine, libdf_status = cmp._install_official()
    ours_lr, ours_wd = cmp.deepfilter_cosine_schedules(cmp._schedule_config())
    task = cmp.DeepFilterNetTask(cmp._task_config(), device)
    independent = cmp.build_model("deepfilternet3", None).to(device)
    independent.load_state_dict(task.generator.state_dict(), strict=True)
    from df.deepfilternet3 import DfNet

    official = DfNet(independent.erb_fb.detach().clone(), independent.mask.erb_inv_fb.detach().clone()).to(device)
    official.load_state_dict(independent.state_dict(), strict=True)
    official.params = independent.params
    generator = torch.Generator().manual_seed(5)
    clean_bank = torch.randn(3, 1600, generator=generator)
    noisy_bank = clean_bank * 0.5 + 0.05 * torch.randn(3, 1600, generator=generator)
    mr = MultiResSpecLoss([256, 512, 1024, 2048], 0.3, 500.0, 500.0).to(device)
    local = LocalSnrLoss(1e-3).to(device)
    opt_o = torch.optim.AdamW(official.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-12, amsgrad=True)
    opt_i = torch.optim.AdamW(independent.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-12, amsgrad=True)
    rows = []
    for step in range(3):
        clean = cmp._resample(clean_bank[step : step + 1]).to(device)
        noisy = cmp._resample(noisy_bank[step : step + 1]).to(device)
        learning_rate = float(ours_lr[step])
        decay = float(ours_wd[step])
        for optim in (opt_o, opt_i):
            for group in optim.param_groups:
                group["lr"] = learning_rate
                group["weight_decay"] = decay
        for module, optim in ((official, opt_o), (independent, opt_i)):
            module.train()
            optim.zero_grad(set_to_none=True)
        loss_o, spectral_o, local_o = cmp._manual_loss(official, task, mr, local, local_snr, clean, noisy)
        loss_i, spectral_i, local_i = cmp._manual_loss(independent, task, mr, local, local_snr, clean, noisy)
        loss_o.backward()
        loss_i.backward()
        torch.nn.utils.clip_grad_norm_(official.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(independent.parameters(), 1.0)
        grads = grad_report(official, independent, atol=1e-4)
        opt_o.step()
        opt_i.step()
        rows.append({
            "step": step,
            "loss_gap": abs(float(loss_o.detach()) - float(loss_i.detach())),
            "spectral_official": spectral_o,
            "spectral_independent": spectral_i,
            "local_official": local_o,
            "local_independent": local_i,
            "grad_max_abs": grads["max_abs"],
            "grad_worst": grads["worst"],
            "weight_max_abs": weight_report(official, independent)["max_abs"],
            "optim_max_abs": optim_report(opt_o, opt_i)["max_abs"],
            "lr": learning_rate,
        })
        print("df step", step, rows[-1]["loss_gap"], rows[-1]["grad_max_abs"], rows[-1]["weight_max_abs"], flush=True)
    return {"libdf": libdf_status, "steps": rows, "task": task, "official": official, "independent": independent}


def _run_epoch(device, built) -> dict:
    libdfdata = types.ModuleType("libdfdata")
    libdfdata.PytorchDataLoader = type("PytorchDataLoader", (), {})
    sys.modules["libdfdata"] = libdfdata
    import df.train as train_mod
    from df.loss import Istft, Loss
    from df.model import ModelParams
    from libdf import DF

    train_mod.summary_write = lambda *args, **kwargs: None
    params = ModelParams()
    state = DF(params.sr, params.fft_size, params.hop_size, params.nb_erb, params.min_nb_freqs)
    window = torch.as_tensor(state.fft_window().copy())
    istft = Istft(params.fft_size, params.hop_size, window).to(device)
    losses = Loss(state, istft).to(device)
    task = built["task"]
    official = built["official"]
    independent = built["independent"]
    # One fresh pair so this call is not stacked on the three manual steps.
    official.load_state_dict(independent.state_dict(), strict=True)
    generator = torch.Generator().manual_seed(11)
    clean = cmp._resample(torch.randn(1, 1600, generator=generator)).to(device)
    noisy = cmp._resample(torch.randn(1, 1600, generator=generator)).to(device)
    task.stream.reset()
    clean_spec, _erb, _feat, _complex = task._analyze_features(clean)
    task.stream.reset()
    noisy_spec, noisy_erb, noisy_feat, _noisy_complex = task._analyze_features(noisy)

    def _complex(spec: torch.Tensor) -> torch.Tensor:
        if torch.is_complex(spec):
            return spec
        return torch.view_as_complex(spec)

    class Batch:
        feat_erb = noisy_erb
        feat_spec = noisy_feat
        noisy = _complex(noisy_spec)
        speech = _complex(clean_spec)
        snr = torch.zeros(clean.shape[0], device=device)
        ids = [0]
        timings = torch.zeros(2, device=device)

    class Loader:
        def get_batch_size(self, split):
            del split
            return 1

        def len(self, split):
            del split
            return 1

        def iter_epoch(self, split, seed):
            del split, seed
            yield Batch()

    summary = OUT / "df-summary"
    summary.mkdir(parents=True, exist_ok=True)
    official.load_state_dict(independent.state_dict(), strict=True)
    opt_o = torch.optim.AdamW(official.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-12, amsgrad=True)
    opt_i = torch.optim.AdamW(independent.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-12, amsgrad=True)
    value_o = train_mod.run_epoch(
        model=official,
        epoch=0,
        loader=Loader(),
        split="train",
        opt=opt_o,
        losses=losses,
        summary_dir=str(summary),
    )
    value_i = train_mod.run_epoch(
        model=independent,
        epoch=0,
        loader=Loader(),
        split="train",
        opt=opt_i,
        losses=losses,
        summary_dir=str(summary),
    )
    return {
        "status": "called df.train.run_epoch on official DfNet and on the audio-db module",
        "official_mean_loss": value_o,
        "independent_mean_loss": value_i,
        "loss_gap": abs(float(value_o) - float(value_i)),
        "weight_max_abs": weight_report(official, independent)["max_abs"],
        "optim_max_abs": optim_report(opt_o, opt_i)["max_abs"],
        "mask_factor": float(losses.ml_f),
        "multires_factor": float(losses.mrsl_f),
        "local_snr_factor": float(losses.lsnr_f),
        "features_from": "audio-db task frontend, not libdfdata",
        "summary_write": "noop",
        "libdfdata": "import stubbed; PytorchDataLoader was not constructed",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    report = {
        "device": str(device),
        "cudnn_deterministic": True,
        "tf32": False,
        "official_commit": "d375b2d8309e0935d165700c91da9de862a99c31",
    }
    built = _steps(device)
    report["libdf"] = built.pop("libdf")
    report["official_dfnet_three_steps"] = built["steps"]
    try:
        report["run_epoch"] = _run_epoch(device, built)
    except Exception as exc:
        report["run_epoch"] = {"status": "FAILED", "reason": f"{type(exc).__name__}: {exc}"}
    path = OUT / "df_official_loop.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path)
    print(json.dumps(report["run_epoch"], indent=2))


if __name__ == "__main__":
    main()
