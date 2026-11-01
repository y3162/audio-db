"""Short real-model DDP probe using saved VoiceBank crops and initial state.

Run only on idle GPUs via torch.distributed.run; outputs must be a new directory.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch
import torch.distributed as dist
from src.training.core.distributed import setup_process, unwrap
from src.training.core.rng import capture_rng, restore_rng, seed_training_stream
from src.se.models.semambapp.network import SelectiveSSM
from tests.training.reference.train_semamba_200 import INIT, _independent
from tests.training.reference.compare_semamba_quality import _require_environment

root, steps, micro, accum = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
resume = Path(sys.argv[5]) if len(sys.argv) > 5 else None
start_step = int(sys.argv[6]) if len(sys.argv) > 6 else 0
_require_environment()
process = setup_process('cuda')
rank = process.rank
assert process.world_size == 2 and micro * accum * 2 == 8
if (root / f'rank-{rank}.json').exists():
    raise SystemExit(f'refusing to overwrite results in {root}')
waves = torch.load(root.parent / 'waves.pt', map_location='cpu', weights_only=False)
initial = torch.load(INIT, map_location='cpu', weights_only=False)
task = _independent(process.device, {'generator': initial['generator'], 'cqt': initial['cqt'], 'mrd': initial['mrd']})['task']
resume_state = torch.load(resume, map_location='cpu', weights_only=False) if resume is not None else None
if resume_state is not None:
    task.load_state_dict(resume_state)
for module in task.generator.modules():
    if isinstance(module, SelectiveSSM):
        module.scan_backend = 'fused'
task.distribute(process)
seed_training_stream(1234, cuda=True)
if resume_state is not None and 'rng' in resume_state:
    restore_rng(resume_state['rng'], cuda=True)
task.capture_update = True
task.capture_gradients = True
reports = []
torch.cuda.reset_peak_memory_stats()
for step in range(start_step, steps):
    batches = []
    hashes = []
    for i in range(accum):
        first = (step * 8 + rank * micro * accum + i * micro) % len(waves['clean'])
        indices = [(first + j) % len(waves['clean']) for j in range(micro)]
        batch = {key: waves[key][indices] for key in ('clean', 'noisy')}
        batch['case_id'] = [waves['case_id'][j] for j in indices]
        batches.append(batch)
        hashes.append({key: hashlib.sha256(batch[key].contiguous().numpy().tobytes()).hexdigest() for key in ('clean', 'noisy')})
    dist.barrier()
    torch.cuda.synchronize()
    start = time.perf_counter()
    metrics = task.train_update(batches, accum)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    # Compare actual generator and both discriminator parameters across ranks.
    sync = {}
    grad_sync = {}
    for name, module in [('generator', task.generator), ('mrd', task.discriminator_mrd), ('band', task.discriminator_band)]:
        flat = torch.cat([p.detach().flatten() for p in unwrap(module).parameters()])
        other = flat.clone()
        dist.broadcast(other, src=0)
        sync[name] = (flat - other).abs().max().item()
        grads = torch.cat([p.grad.detach().flatten() for p in unwrap(module).parameters() if p.grad is not None])
        other_grads = grads.clone()
        dist.broadcast(other_grads, src=0)
        grad_sync[name] = (grads - other_grads).abs().max().item()
    rng_state = capture_rng(cuda=True)
    rng_hash = {'torch': hashlib.sha256(rng_state['torch'].numpy().tobytes()).hexdigest(),
                'cuda': [hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in rng_state['cuda']]}
    reports.append({'step': step + 1, 'metrics': metrics, 'seconds': seconds, 'hashes': hashes, 'rng': rng_hash,
                    'rank_max_abs_vs_zero': sync, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                    'gradient_max_abs_vs_zero': grad_sync,
                    'learning_rates': task.learning_rates()})
    if rank == 0 and step + 1 in (1, 2, 4, 20):
        state = task.state_dict()
        state['generator_grad'] = task.last_capture['generator_grad']
        state['discriminator_grad'] = task.last_capture['discriminator_grad']
        state['rng'] = rng_state
        torch.save(state, root / f'checkpoint-{step + 1}.pt')
    (root / f'rank-{rank}.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
    print(json.dumps({'rank': rank, **reports[-1]}), flush=True)
    dist.barrier()
dist.destroy_process_group()
