"""Single-process official/independent paired VoiceBank training diagnostic."""
import json
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tests.training.reference.compare_semamba_quality import (
    _require_environment, _cfg, _flat, SEMambapp, SEMambaPP, OfficialCQT, OfficialMRD,
    MultiScaleSubbandCQTDiscriminator, MultiResolutionDiscriminator, MultiScaleMelSpectrogramLoss,
    _official_parts, _independent_parts, _official_losses, _independent_losses, LR, BETAS, WEIGHT_DECAY,
)
from src.training.data.dataset import load_recipe_dataset
from src.training.data.sampler import epoch_indices
from src.training.core.rng import capture_rng, restore_rng, seed_training_stream

side, out, steps = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
if side not in ('official', 'independent') or steps < 1 or out.exists(): raise SystemExit('invalid arguments or output exists')
out.mkdir(parents=True)
_require_environment()
device = torch.device('cuda')
cfg = _cfg(48, 4, 16, 4)
if side == 'official': g, cqt, mrd = SEMambapp(cfg), OfficialCQT(), OfficialMRD()
else: g, cqt, mrd = SEMambaPP(_flat(cfg)), MultiScaleSubbandCQTDiscriminator(), MultiResolutionDiscriminator()
initial = torch.load(ROOT / 'data/results/training/semamba-repro-20261101/run2/checkpoints/step_0000_official.pt', map_location='cpu', weights_only=False)
for model, key in ((g, 'generator'), (cqt, 'cqt'), (mrd, 'mrd')):
    model.load_state_dict(initial[key], strict=True); model.to(device).train()
og = torch.optim.AdamW(g.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)
od = torch.optim.AdamW(list(cqt.parameters()) + list(mrd.parameters()), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)
sg = torch.optim.lr_scheduler.ExponentialLR(og, gamma=.99)
sd = torch.optim.lr_scheduler.ExponentialLR(od, gamma=.99)
mel = MultiScaleMelSpectrogramLoss(sampling_rate=16000).to(device)
ds = load_recipe_dataset(ROOT / 'data/db/semamba-voicebank-train-20261101-compat.duckdb', 'voicebank_28_train', role='train', seed=1234, sample_rate=16000, segment_size=8000, training=True)
seed_training_stream(1234, cuda=True)
start = 0
if len(sys.argv) > 4:
    cp = torch.load(sys.argv[4], map_location='cpu', weights_only=False)
    if cp.get('side') != side: raise SystemExit('checkpoint side differs')
    for model, key in ((g, 'generator'), (cqt, 'cqt'), (mrd, 'mrd')): model.load_state_dict(cp[key])
    og.load_state_dict(cp['optim_g']); od.load_state_dict(cp['optim_d'])
    sg.load_state_dict(cp['scheduler_g']); sd.load_state_dict(cp['scheduler_d'])
    restore_rng(cp['rng'], cuda=True)
    start = cp['updates_completed']
    if start >= steps: raise SystemExit('checkpoint already at requested step')
def save(step, epoch, cursor):
    torch.save({'side': side, 'updates_completed': step, 'epoch': epoch, 'cursor': cursor, 'generator': g.state_dict(), 'cqt': cqt.state_dict(), 'mrd': mrd.state_dict(), 'optim_g': og.state_dict(), 'optim_d': od.state_dict(), 'scheduler_g': sg.state_dict(), 'scheduler_d': sd.state_dict(), 'rng': capture_rng(cuda=True)}, out / f'step_{step:08d}.pt')
save(start, *divmod(start, len(ds.examples)))
for i in range(start, steps):
    epoch, cursor = divmod(i, len(ds.examples))
    if i == start or cursor == 0:
        ds.set_epoch(epoch); order = epoch_indices(len(ds.examples), 1234, epoch, 1, 0)
    batch = ds.collate([order[cursor]])
    clean, noisy = batch['clean'].to(device), batch['noisy'].to(device)
    def parts(): return _official_parts(g, clean, noisy, cfg) if side == 'official' else _independent_parts(g, clean, noisy)
    def losses(p): return _official_losses(p, clean, cqt, mrd, mel, cfg) if side == 'official' else _independent_losses(p, clean, cqt, mrd)
    og.zero_grad(set_to_none=True); od.zero_grad(set_to_none=True)
    p = parts(); ld = losses({**p, 'audio': p['audio'].detach()})['discriminator']
    if not torch.isfinite(ld): raise RuntimeError(f'non-finite discriminator loss at step {i+1}')
    ld.backward(); od.step(); og.zero_grad(set_to_none=True)
    lg = losses(parts())['generator']
    if not torch.isfinite(lg): raise RuntimeError(f'non-finite generator loss at step {i+1}')
    lg.backward(); og.step()
    if cursor + 1 == len(ds.examples): sg.step(); sd.step()
    with (out / 'loss.jsonl').open('a') as f: f.write(json.dumps({'step': i+1, 'epoch': epoch, 'case_id': batch['case_id'][0], 'generator': float(lg.detach()), 'discriminator': float(ld.detach())})+'\n')
    if (i+1) % 1000 == 0 or i+1 == steps: save(i+1, epoch, cursor+1)
    print(side, i+1, flush=True)
