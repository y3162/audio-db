"""One implementation per process, with step-10 resume diagnostics."""
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tests.training.reference.compare_semamba_quality import (
    _require_environment, _cfg, _flat, SEMambapp, SEMambaPP, OfficialCQT,
    OfficialMRD, MultiScaleSubbandCQTDiscriminator, MultiResolutionDiscriminator,
    MultiScaleMelSpectrogramLoss, _official_parts, _independent_parts,
    _official_losses, _independent_losses, _load_training_batches, module_sha256,
    LR, BETAS, WEIGHT_DECAY,
)
from src.training.core.rng import capture_rng, restore_rng, seed_training_stream

side, output = sys.argv[1], Path(sys.argv[2])
if side not in ('official', 'independent') or output.exists():
    raise SystemExit('invalid side or output exists')
output.mkdir(parents=True)
_require_environment()
device = torch.device('cuda')
cfg = _cfg(48, 4, 16, 4)
if side == 'official':
    g, cqt, mrd = SEMambapp(cfg), OfficialCQT(), OfficialMRD()
else:
    g, cqt, mrd = SEMambaPP(_flat(cfg)), MultiScaleSubbandCQTDiscriminator(), MultiResolutionDiscriminator()
initial = torch.load(ROOT / 'data/results/training/semamba-repro-20261101/run2/checkpoints/step_0000_official.pt', map_location='cpu', weights_only=False)
for model, key in ((g, 'generator'), (cqt, 'cqt'), (mrd, 'mrd')):
    model.load_state_dict(initial[key], strict=True)
    model.to(device).train()
optim_g = torch.optim.AdamW(g.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)
optim_d = torch.optim.AdamW(list(cqt.parameters()) + list(mrd.parameters()), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)
sched_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=.99)
sched_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=.99)
mel = MultiScaleMelSpectrogramLoss(sampling_rate=16000).to(device)
start = 0
if len(sys.argv) > 3:
    cp = torch.load(sys.argv[3], map_location='cpu', weights_only=False)
    for model, key in ((g, 'generator'), (cqt, 'cqt'), (mrd, 'mrd')): model.load_state_dict(cp[key])
    optim_g.load_state_dict(cp['optim_g']); optim_d.load_state_dict(cp['optim_d'])
    sched_g.load_state_dict(cp['scheduler_g']); sched_d.load_state_dict(cp['scheduler_d'])
    restore_rng(cp['rng'], cuda=True)
    start = cp['updates_completed']
else:
    seed_training_stream(1234, cuda=True)

def save(step):
    before = capture_rng(cuda=True)
    torch.save({'updates_completed': step, 'generator': g.state_dict(), 'cqt': cqt.state_dict(), 'mrd': mrd.state_dict(), 'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(), 'scheduler_g': sched_g.state_dict(), 'scheduler_d': sched_d.state_dict(), 'rng': before, 'experiment': {'side': side}}, output / f'step_{step:04d}.pt')
    after = capture_rng(cuda=True)
    with (output / 'save_rng.jsonl').open('a') as f:
        f.write(json.dumps({'step': step, 'cpu_equal': torch.equal(before['torch'], after['torch']), 'cuda_equal': all(torch.equal(a, b) for a, b in zip(before['cuda'], after['cuda']))}) + '\n')

def parts(clean, noisy):
    return _official_parts(g, clean, noisy, cfg) if side == 'official' else _independent_parts(g, clean, noisy)
def losses(p, clean):
    return _official_losses(p, clean, cqt, mrd, mel, cfg) if side == 'official' else _independent_losses(p, clean, cqt, mrd)

save(start)
for i, batch in enumerate(_load_training_batches(20)[start:], start):
    clean, noisy = batch['clean'].to(device), batch['noisy'].to(device)
    optim_d.zero_grad(set_to_none=True); optim_g.zero_grad(set_to_none=True)
    p = parts(clean, noisy)
    ld = losses({**p, 'audio': p['audio'].detach()}, clean)['discriminator']
    ld.backward(); optim_d.step(); optim_g.zero_grad(set_to_none=True)
    p = parts(clean, noisy)
    lg = losses(p, clean)
    lg['generator'].backward()
    if i == start:
        torch.save({'forward': p['audio'].detach().cpu(), 'g_grad': {k: v.grad.detach().cpu() for k, v in g.named_parameters() if v.grad is not None}, 'd_grad': {f'{prefix}.{k}': v.grad.detach().cpu() for prefix, model in (('cqt', cqt), ('mrd', mrd)) for k, v in model.named_parameters() if v.grad is not None}}, output / 'first_step.pt')
    optim_g.step()
    with (output / 'loss.jsonl').open('a') as f:
        f.write(json.dumps({'step': i + 1, 'case_id': batch['case_id'], 'discriminator': float(ld.detach()), 'generator': float(lg['generator'].detach())}) + '\n')
    if i + 1 in (1, 10, 20): save(i + 1)
    print(side, i + 1, flush=True)
