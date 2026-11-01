"""Reproduce the PESQ boundary on saved waveforms; output must be new."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from pesq import pesq

source = Path('data/results/training/semamba-quality-20261101/phase-a/waves/step_0010.pt')
out = Path(sys.argv[1])
if out.exists():
    raise SystemExit('output already exists')
row = torch.load(source, map_location='cpu', weights_only=False)['utterances'][0]
ref = row['clean'].numpy().astype(np.float64)
est = row['independent'].numpy().astype(np.float64)
def score(a, b):
    return float(pesq(16000, a, b, 'wb'))
lengths = {str(n): score(ref[:n], est[:n]) for n in range(93600, 93681)}
padding = {}
for n in (93600, 93620, 93640):
    padding[str(n)] = score(np.pad(ref[:n], (0, len(ref)-n)), np.pad(est[:n], (0, len(est)-n)))
reverse = {str(n): score(ref[:n], est[:n]) for n in reversed(range(93600, 93681))}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({'source': str(source), 'case_id': row['case_id'], 'prefix': lengths,
                           'zero_pad_to_full': padding, 'reverse_order': reverse}, indent=2))
