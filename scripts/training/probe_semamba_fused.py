"""Run a short fused SEMamba++ GAN update using the selected CUDA device."""

import json

import torch
import causal_conv1d_cuda
import selective_scan_cuda

from src.training.se.config import load_training_config
from src.training.se.run import build_task


def main():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    config = load_training_config("configs/training/semambapp_seed17.yaml")
    task = build_task(config, torch.device("cuda"))
    modules = (task.generator, task.discriminator_mrd, task.discriminator_band)
    before = [next(module.parameters()).detach().clone() for module in modules]
    clean = torch.randn(1, 8000, device="cuda") * 0.1
    batch = {"clean": clean, "noisy": clean + torch.randn_like(clean) * 0.03, "case_id": ["synthetic"]}
    losses = task.train_update([batch], 1)
    torch.cuda.synchronize()
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    for module, original in zip(modules, before):
        assert all(torch.isfinite(parameter).all() for parameter in module.parameters())
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in module.parameters()
        )
        assert not torch.equal(original, next(module.parameters()))
    for optimizer in (task.optim_g, task.optim_d):
        assert optimizer.state
        for state in optimizer.state.values():
            assert all(
                not isinstance(value, torch.Tensor) or torch.isfinite(value).all()
                for value in state.values()
            )
    print(json.dumps({
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "causal": causal_conv1d_cuda.__file__,
        "scan": selective_scan_cuda.__file__,
        "loss": losses,
        "peak_bytes": torch.cuda.max_memory_allocated(),
    }, indent=2))


if __name__ == "__main__":
    main()
