"""Synchronously time coarse one-step regions; instrumentation adds overhead."""
import contextlib
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch
from src.se.models.semambapp.network import SelectiveSSM
from tests.numerics.reference.compare_semamba_quality import _require_environment
from tests.numerics.reference.run_semamba_official_batch import _load, _update, _waves, _slice

_require_environment()
side, root = sys.argv[1], Path(sys.argv[2])
kind = 'official' if side == 'official' else 'independent'
t0 = time.perf_counter(); batch = _slice(_waves(root), 0); data = time.perf_counter() - t0
with contextlib.redirect_stdout(sys.stderr):
    bundle = _load(kind)
if kind == 'official':
    generator, discriminators, optimizers = bundle['generator'], [bundle['mrd'], bundle['cqt']], [bundle['optim_d'], bundle['optim_g']]
else:
    task = bundle['task']
    generator, discriminators, optimizers = task.generator, [task.discriminator_mrd, task.discriminator_band], [task.optim_d, task.optim_g]
    if side == 'fused':
        for m in generator.modules():
            if isinstance(m, SelectiveSSM): m.scan_backend = 'fused'
times = {'generator_forward': 0., 'discriminator_forward': 0., 'stft_calls': 0., 'optimizer': 0.}
def before(module, args):
    torch.cuda.synchronize(); module._probe_time = time.perf_counter()
def after(name):
    def hook(module, args, output):
        torch.cuda.synchronize(); times[name] += time.perf_counter() - module._probe_time
    return hook
for module, name in [(generator, 'generator_forward'), *((m, 'discriminator_forward') for m in discriminators)]:
    module.register_forward_pre_hook(before); module.register_forward_hook(after(name))
original_stft = torch.stft
def stft(*args, **kwargs):
    torch.cuda.synchronize(); start = time.perf_counter(); out = original_stft(*args, **kwargs); torch.cuda.synchronize()
    times['stft_calls'] += time.perf_counter() - start
    return out
torch.stft = stft
original_backward = torch.Tensor.backward
backwards = []
def backward(self, *args, **kwargs):
    torch.cuda.synchronize(); start = time.perf_counter(); out = original_backward(self, *args, **kwargs); torch.cuda.synchronize()
    backwards.append(time.perf_counter() - start)
    return out
torch.Tensor.backward = backward
for optimizer in optimizers:
    original = optimizer.step
    def step(*args, _original=original, **kwargs):
        torch.cuda.synchronize(); start = time.perf_counter(); out = _original(*args, **kwargs); torch.cuda.synchronize()
        times['optimizer'] += time.perf_counter() - start
        return out
    optimizer.step = step
torch.cuda.synchronize(); start = time.perf_counter(); _update(kind, bundle, [batch], 1, capture=False); torch.cuda.synchronize()
print(json.dumps({'side': side, 'data_seconds': data, 'instrumented_total_seconds': time.perf_counter() - start,
                  **times, 'backward_calls_in_order': backwards, 'peak_allocated_bytes': torch.cuda.max_memory_allocated()}, indent=2))
