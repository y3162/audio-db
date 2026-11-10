"""One 3 s real-pair GAN update from a shared checkpoint, saving comparisons."""
import hashlib
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch
from tests.numerics.reference.run_semamba_official_batch import _load, _update, _waves, _slice, _generator_state
from tests.numerics.reference.compare_semamba_quality import _require_environment
from src.se.models.semambapp.network import SelectiveSSM
from src.experiments.calc.rng import seed_training_stream

side, directory = sys.argv[1], Path(sys.argv[2])
_require_environment()
dest = directory / f'update-{side}.pt'
if dest.exists():
    raise SystemExit(f'refusing to overwrite {dest}')
kind = 'official' if side == 'official' else 'independent'
t0 = time.perf_counter()
waves = _waves(directory)
batch = _slice(waves, 0)
data_seconds = time.perf_counter() - t0
bundle = _load(kind)
if side == 'fused':
    for module in bundle['task'].generator.modules():
        if isinstance(module, SelectiveSSM):
            module.scan_backend = 'fused'
seed_training_stream(1234, cuda=True)
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
metrics, capture = _update(kind, bundle, [batch], 1, capture=True)
torch.cuda.synchronize()
seconds = time.perf_counter() - t0
audio = capture['audio']
if isinstance(audio, list):
    audio = torch.cat(audio)
payload = {'side': side, 'metrics': metrics, 'audio': audio.cpu(),
           'generator_grad': capture['generator_grad'], 'discriminator_grad': capture['discriminator_grad'],
           'generator_after': _generator_state(kind, bundle),
           'input_sha256': {key: hashlib.sha256(batch[key].contiguous().numpy().tobytes()).hexdigest() for key in ('clean', 'noisy')},
           'seconds': seconds, 'data_seconds': data_seconds, 'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
torch.save(payload, dest)
print(json.dumps({k: v for k, v in payload.items() if k not in ('audio', 'generator_grad', 'discriminator_grad', 'generator_after')}, indent=2))
