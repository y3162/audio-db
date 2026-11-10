"""Aggregate all 824 diagnostic prefix scores without omitting failures."""
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tests.numerics.reference.semamba_quality import summarize_step, tolerance_result

out = Path(sys.argv[1])
if out.exists(): raise SystemExit('output already exists')
paths = sorted((p for i in range(3) for p in (ROOT / f'data/results/training/semamba-voicebank-scores-{i}-20261101').glob('*.json')), key=lambda p: p.name)
if [p.name for p in paths] != [f'{i:04d}.json' for i in range(824)]: raise SystemExit('incomplete or duplicate score files')
rows = [json.loads(p.read_text()) for p in paths]
dist = {}
for metric, limit in [('pesq', .1), ('stoi', .02), ('si_sdr', 1.)]:
    values = [abs(r['official_minus_independent'][metric]['value']) for r in rows if r['official_minus_independent'][metric]['value'] is not None]
    dist[metric] = {'finite': len(values), 'missing': len(rows)-len(values), 'violations': sum(x > limit for x in values), 'max': max(values) if values else None, 'quantiles_50_90_99': np.quantile(values, [.5,.9,.99]).tolist() if values else None}
out.write_text(json.dumps({'warning': 'Diagnostic common prefix. PESQ alignment pathology and checkpoint training overlap unresolved. Not a quality GO.', 'summary': summarize_step(rows), 'tolerance_on_prefix': tolerance_result(rows), 'distribution': dist, 'original_length_mismatches': sum(len(set(r['original_lengths'])) != 1 for r in rows), 'rows': rows}, indent=2))
