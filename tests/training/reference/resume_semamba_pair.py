"""Resume a paired reference experiment at step 10 through step 20 on CUDA."""
import argparse
import json
from pathlib import Path
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tests.training.reference.compare_semamba_quality import (
    _require_environment, _build, _optimizers, _load_training_batches,
    _one_update, _save_pair, MultiScaleMelSpectrogramLoss, SAMPLE_RATE,
)
from src.training.core.rng import restore_rng, capture_rng


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    _require_environment()
    device = torch.device('cuda:0')
    cfg, modules, _ = _build(device, args.out)
    optim = _optimizers(modules)
    checkpoints = {side: torch.load(args.run / 'checkpoints' / f'step_0010_{side}.pt', map_location='cpu', weights_only=False) for side in ('official', 'independent')}
    for side in ('official', 'independent'):
        cp = checkpoints[side]
        if cp['updates_completed'] != 10: raise RuntimeError('not a step-10 checkpoint')
        for component, key in (('g','generator'), ('cqt','cqt'), ('mrd','mrd')):
            modules[f'{side}_{component}'].load_state_dict(cp[key])
        optim[f'{side}_g'].load_state_dict(cp['optim_g'])
        optim[f'{side}_d'].load_state_dict(cp['optim_d'])
    for index, (side, kind) in enumerate((('official','g'),('independent','g'),('official','d'),('independent','d'))):
        optim['schedulers'][index].load_state_dict(checkpoints[side][f'scheduler_{kind}'])
    mel = MultiScaleMelSpectrogramLoss(sampling_rate=SAMPLE_RATE).to(device)
    restore_rng(checkpoints['official']['rng'], cuda=True)
    batches = _load_training_batches(20)
    start = time.perf_counter()
    for i in range(10, 20):
        batch = batches[i]
        result = _one_update(modules, optim, cfg, batch['clean'], batch['noisy'], mel, device)
        with (args.out / 'train_log.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps({'updates_completed': i + 1, 'case_id': batch['case_id'], 'losses': result['losses'], 'wave': result['wave']}) + '\n')
        print(f'update {i + 1}/20', flush=True)
    rng = capture_rng(cuda=True)
    _save_pair(args.out, modules, optim, 20, checkpoints['official']['experiment'], rng)
    (args.out / 'resume.json').write_text(json.dumps({'source': str(args.run), 'from_updates': 10, 'updates_completed': 20, 'seconds': time.perf_counter()-start}, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__': main()
