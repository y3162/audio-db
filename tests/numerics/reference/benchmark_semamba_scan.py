"""Time an independent SEMamba++ generator, using synthetic 3 s STFT inputs.

CUDA_VISIBLE_DEVICES=<free> OMP_NUM_THREADS=1 python tests/numerics/reference/benchmark_semamba_scan.py reference
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch
from src.se.models.semambapp.inference import build

backend = sys.argv[1]
warmup = len(sys.argv) > 2 and sys.argv[2] == 'warmup'
torch.manual_seed(1234)
torch.cuda.set_device(0)
model = build({'scan_backend': backend}).cuda().train()
mag = torch.rand(1, 201, 481, device='cuda')
phase = torch.rand_like(mag)
if warmup:
    sum(x.square().mean() for x in model(mag, phase)).backward()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
start = time.perf_counter()
out = model(mag, phase)
torch.cuda.synchronize()
forward = time.perf_counter() - start
start = time.perf_counter()
sum(x.square().mean() for x in out).backward()
torch.cuda.synchronize()
backward = time.perf_counter() - start
print(json.dumps({'backend': backend, 'seconds_forward': forward, 'seconds_backward': backward,
                  'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'device': torch.cuda.get_device_name(), 'synthetic': True, 'warmup': warmup}))
