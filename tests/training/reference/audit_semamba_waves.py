"""Audit saved quality waves without changing or excluding any utterances."""
import argparse, json, sys
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tests.training.reference.semamba_quality import score_conditions, score_waveform

p = argparse.ArgumentParser()
p.add_argument('input', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
if a.output.exists():
    raise SystemExit('refusing to overwrite output')
data = torch.load(a.input, map_location='cpu', weights_only=False)
sr = data['sample_rate']
rows = []
for u in data['utterances']:
    waves = {k: u[k].flatten() for k in ('clean','noisy','official','independent')}
    details = {}
    for k, x in waves.items():
        details[k] = {'samples': x.numel(), 'finite': bool(torch.isfinite(x).all()), 'peak': float(x.abs().max()), 'rms': float(x.square().mean().sqrt()), 'samples_abs_gt_1': int((x.abs()>1).sum())}
    length = min(x.numel() for x in waves.values())
    prefix = score_conditions(*(waves[k][:length] for k in ('clean','noisy','official','independent')), sr)
    padded = {k:F.pad(x,(0,waves['clean'].numel()-x.numel())) for k,x in waves.items()}
    full = score_conditions(*(padded[k] for k in ('clean','noisy','official','independent')), sr)
    segments = {k: [score_waveform(waves['clean'][i:i+sr],waves[k][i:i+sr],sr) for i in range(0,length-sr+1,sr)] for k in ('noisy','official','independent')}
    rows.append({'case_id':u['case_id'],'details':details,'common_prefix_samples':length,'common_prefix':prefix,'zero_padded_to_clean_length':full,'one_second_segments':segments})
a.output.write_text(json.dumps({'source':str(a.input),'sample_rate':sr,'rows':rows},indent=2),encoding='utf-8')
