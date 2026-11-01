"""Diagnostic common-prefix scoring; PESQ is not used as a GO criterion."""
import json
import sys
from pathlib import Path

import torch
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tests.training.reference.semamba_quality import score_conditions, summarize_step, tolerance_result

target = Path(sys.argv[-1])
sources = list(map(Path, sys.argv[1:-1]))
if target.exists():
    raise SystemExit('output already exists')
paths = sorted((p for source in sources for p in source.glob('*.pt')), key=lambda p: p.name)
if len(paths) != 824 or [p.name for p in paths] != [f'{i:04d}.pt' for i in range(824)]:
    raise SystemExit(f'expected 824 completed waveforms, got {len(paths)}')
rows = []
for path in paths:
    sample = torch.load(path, map_location='cpu', weights_only=True)
    waves = [sample[name] for name in ('clean', 'noisy', 'official', 'independent')]
    lengths = [int(x.numel()) for x in waves]
    n = min(lengths)
    scored = score_conditions(*(x[:n] for x in waves), 16000)
    scored['case_id'] = sample['case_id']
    scored['original_lengths'] = lengths
    scored['common_prefix_samples'] = n
    rows.append(scored)
    print(path.name, flush=True)
target.write_text(json.dumps({'warning': 'diagnostic common prefix; PESQ anomaly unresolved; checkpoint training overlap unknown', 'summary': summarize_step(rows), 'tolerance_on_prefix': tolerance_result(rows), 'rows': rows}, indent=2))
