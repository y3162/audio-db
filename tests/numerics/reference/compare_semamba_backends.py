"""Compare one mixer with common weights/input; GPU and external kernels required."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch
import torch.nn.functional as F
from mamba_ssm.modules.mamba_simple import Mamba
from src.se.models.semambapp.network import SelectiveSSM

# Fixed before running: outputs atol=2e-5, rtol=2e-4; gradients atol=2e-5,
# rtol=3e-4. Norm-relative error and cosine accompany elementwise tolerance.
torch.manual_seed(5)
modules = {'reference': SelectiveSSM(48, 16, 4, 2),
           'fused': SelectiveSSM(48, 16, 4, 2, 'fused'),
           'official_mamba': Mamba(d_model=48, d_state=16, d_conv=4, expand=2)}
state = modules['reference'].state_dict()
for m in modules.values():
    m.load_state_dict(state, strict=True)
    m.cuda()
source = torch.randn(2, 64, 48, device='cuda')
results = {}
for name, module in modules.items():
    x = source.detach().clone().requires_grad_(True)
    y = module(x)
    y.square().mean().backward()
    results[name] = {'output': y.detach().cpu(), 'input_grad': x.grad.detach().cpu(),
                     **{n: p.grad.detach().cpu() for n, p in module.named_parameters()}}

def stats(a, b, output=False):
    a, b = a.double().flatten(), b.double().flatten()
    diff = a - b
    return {'max_abs': diff.abs().max().item(),
            'relative_l2': (diff.norm() / a.norm().clamp_min(1e-30)).item(),
            'cosine': F.cosine_similarity(a, b, dim=0).item(),
            'allclose': torch.allclose(a, b, rtol=2e-4 if output else 3e-4, atol=2e-5)}
report = {}
for a, b in [('reference', 'fused'), ('official_mamba', 'fused'), ('official_mamba', 'reference')]:
    report[f'{a}_vs_{b}'] = {k: stats(v, results[b][k], k == 'output') for k, v in results[a].items()}
print(json.dumps(report, indent=2))
