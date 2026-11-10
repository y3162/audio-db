"""Profile one real-pair GAN update with CPU and CUDA operator totals."""
import json
import contextlib
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch
from torch.profiler import profile, ProfilerActivity
from src.se.models.semambapp.network import SelectiveSSM
from tests.numerics.reference.compare_semamba_quality import _require_environment
from tests.numerics.reference.run_semamba_official_batch import _load, _update, _waves, _slice

_require_environment()
side, root = sys.argv[1], Path(sys.argv[2])
kind = 'official' if side == 'official' else 'independent'
t0 = time.perf_counter(); waves = _waves(root); batch = _slice(waves, 0); data_seconds = time.perf_counter() - t0
with contextlib.redirect_stdout(sys.stderr):
    bundle = _load(kind)
if side == 'fused':
    for m in bundle['task'].generator.modules():
        if isinstance(m, SelectiveSSM): m.scan_backend = 'fused'
torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
    metrics, _ = _update(kind, bundle, [batch], 1, capture=False)
    torch.cuda.synchronize()
elapsed = time.perf_counter() - start
rows = []
for event in prof.key_averages():
    rows.append({'name': event.key, 'calls': event.count, 'self_cpu_us': event.self_cpu_time_total,
                 'self_cuda_us': event.self_device_time_total, 'cpu_us': event.cpu_time_total,
                 'cuda_us': event.device_time_total, 'self_cuda_memory_bytes': event.self_device_memory_usage})
print(json.dumps({'backend': side, 'data_seconds': data_seconds, 'profiled_update_seconds': elapsed,
                  'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'metrics': metrics,
                  'operators': sorted(rows, key=lambda x: x['self_cuda_us'], reverse=True)}, indent=2))
