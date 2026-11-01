"""Compute waveform comparison in float64 for all named-set inference outputs."""
import json
import sys
from pathlib import Path
import torch

root = Path('data/results/training')
out = Path(sys.argv[1])
if out.exists(): raise SystemExit('output already exists')
sources = [root / 'semamba-voicebank-inference-20261101-v2'] + [root / f'semamba-voicebank-shard-{i}-20261101' for i in (1, 2, 3)]
paths = sorted((p for directory in sources for p in directory.glob('*.pt')), key=lambda p: p.name)
if [p.name for p in paths] != [f'{i:04d}.pt' for i in range(824)]: raise SystemExit('incomplete or duplicate outputs')
rows = []
for path in paths:
    saved = torch.load(path, map_location='cpu', weights_only=True)
    a, b = saved['official'].double(), saved['independent'].double()
    n = min(a.numel(), b.numel()); a, b = a[:n], b[:n]
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    cosine = None if denom == 0 else float(torch.dot(a, b) / denom)
    rows.append({'case_id': saved['case_id'], 'cosine': cosine, 'max_abs': float((a-b).abs().max()), 'common_prefix_samples': n})
out.write_text(json.dumps(rows))
