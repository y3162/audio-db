"""Measure sensitivity of the phase loss to observed backend output differences."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch
from src.se.models.semambapp.network import SelectiveSSM
from src.training.se.spectral import matrix_phase_losses
from tests.training.reference.run_semamba_official_batch import _load, _waves
from tests.training.reference.compare_semamba_quality import _require_environment

_require_environment()
waves = _waves(Path(sys.argv[1]))
task = _load('independent')['task']
clean, noisy = (waves[k][:1].cuda() for k in ('clean', 'noisy'))
outputs = {}
with torch.no_grad():
    for backend in ('reference', 'fused'):
        for module in task.generator.modules():
            if isinstance(module, SelectiveSSM):
                module.scan_backend = backend
        parts = task._generate(clean, noisy, task.generator)
        outputs[backend] = (parts['clean_phase'].detach(), parts['phase'].detach())
values = {}
grads = {}
for backend, (target, phase) in outputs.items():
    phase = phase.detach().requires_grad_(True)
    loss = sum(matrix_phase_losses(target, phase))
    grads[backend] = torch.autograd.grad(loss, phase)[0].detach().double().flatten()
    values[backend] = loss.item()
a, b = grads['reference'], grads['fused']
delta = a - b
print(json.dumps({'loss': values, 'phase_max_abs': (outputs['reference'][1] - outputs['fused'][1]).abs().max().item(),
                  'phase_grad_relative_l2': (delta.norm() / a.norm()).item(),
                  'phase_grad_cosine': torch.nn.functional.cosine_similarity(a, b, dim=0).item(),
                  'phase_grad_sign_flips': int((a * b < 0).sum()), 'elements': a.numel()}, indent=2))
