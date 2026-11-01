"""Run the official reference GAN on exactly the real crops used by the DDP probe."""
import hashlib
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch
import torch.distributed as dist
from src.training.core.distributed import setup_process, unwrap, wrap_module
from src.training.core.rng import seed_training_stream, capture_rng
from tests.training.reference.compare_semamba_quality import _require_environment
from tests.training.reference.run_semamba_official_batch import _load, _update

root, steps = Path(sys.argv[1]), int(sys.argv[2])
_require_environment()
process = setup_process('cuda')
rank = process.rank
assert process.world_size == 2
if (root / f'rank-{rank}.json').exists():
    raise SystemExit(f'refusing to overwrite {root}')
waves = torch.load(root.parent / 'waves.pt', map_location='cpu', weights_only=False)
bundle = _load('official')
for key in ('generator', 'mrd', 'cqt'):
    bundle[key] = wrap_module(bundle[key], process)
seed_training_stream(1234, cuda=True)
torch.cuda.reset_peak_memory_stats()
reports = []
for step in range(steps):
    indices = [(step * 8 + rank * 4 + j) % len(waves['clean']) for j in range(4)]
    batch = {key: waves[key][indices] for key in ('clean', 'noisy')}
    batch['case_id'] = [waves['case_id'][j] for j in indices]
    hashes = [{key: hashlib.sha256(batch[key].contiguous().numpy().tobytes()).hexdigest() for key in ('clean', 'noisy')}]
    dist.barrier(); torch.cuda.synchronize(); start = time.perf_counter()
    metrics, captured = _update('official', bundle, [batch], 1, capture=True)
    torch.cuda.synchronize(); seconds = time.perf_counter() - start
    sync, grad_sync = {}, {}
    for name in ('generator', 'mrd', 'cqt'):
        module = unwrap(bundle[name])
        flat = torch.cat([p.detach().flatten() for p in module.parameters()])
        other = flat.clone(); dist.broadcast(other, 0)
        sync[name] = (flat - other).abs().max().item()
        flat = torch.cat([p.grad.detach().flatten() for p in module.parameters() if p.grad is not None])
        other = flat.clone(); dist.broadcast(other, 0)
        grad_sync[name] = (flat - other).abs().max().item()
    rng_state = capture_rng(cuda=True)
    rng_hash = {'torch': hashlib.sha256(rng_state['torch'].numpy().tobytes()).hexdigest(),
                'cuda': [hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in rng_state['cuda']]}
    reports.append({'step': step + 1, 'metrics': metrics, 'seconds': seconds, 'hashes': hashes, 'rng': rng_hash,
                    'rank_max_abs_vs_zero': sync, 'gradient_max_abs_vs_zero': grad_sync,
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                    'learning_rates': {'generator': bundle['optim_g'].param_groups[0]['lr'],
                                       'discriminator': bundle['optim_d'].param_groups[0]['lr']}})
    if rank == 0 and step + 1 in (1, 4, 20):
        torch.save({'generator': unwrap(bundle['generator']).state_dict(),
                    'discriminator_mrd': unwrap(bundle['mrd']).state_dict(),
                    'discriminator_band': unwrap(bundle['cqt']).state_dict(),
                    'optim_g': bundle['optim_g'].state_dict(), 'optim_d': bundle['optim_d'].state_dict(),
                    'rng': rng_state, 'generator_grad': captured['generator_grad'],
                    'discriminator_grad': captured['discriminator_grad']}, root / f'checkpoint-{step + 1}.pt')
    (root / f'rank-{rank}.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
    print(json.dumps({'rank': rank, **reports[-1]}), flush=True)
    dist.barrier()
dist.destroy_process_group()
