"""Summarize two 2-rank runs, including all data hashes and per-step losses."""
import json
import statistics
import sys
from pathlib import Path

left, right = map(Path, sys.argv[1:3])
L = [json.loads((left / f'rank-{r}.json').read_text()) for r in (0, 1)]
R = [json.loads((right / f'rank-{r}.json').read_text()) for r in (0, 1)]
n = min(*(len(x) for x in L + R))
rows = []
for i in range(n):
    a, b = [x[i] for x in L], [x[i] for x in R]
    keys = set(a[0]['metrics']) & set(b[0]['metrics'])
    losses = {k: [sum(float(x['metrics'][k]) for x in a) / 2,
                  sum(float(x['metrics'][k]) for x in b) / 2]
              for k in keys if isinstance(a[0]['metrics'][k], float)}
    rows.append({'step': i + 1, 'hashes_equal': all(x['hashes'] == y['hashes'] for x, y in zip(a, b)),
                 'losses_and_grad_norms': losses,
                 'lr_equal': all(x['learning_rates'] == y['learning_rates'] for x, y in zip(a, b)),
                 'rank_weight_max': [max(max(x['rank_max_abs_vs_zero'].values()) for x in run) for run in (a, b)]})
print(json.dumps({'steps': n, 'all_hashes_equal': all(r['hashes_equal'] for r in rows),
                  'left_seconds_mean_excluding_first': statistics.mean(x['seconds'] for x in L[0][1:n]),
                  'right_seconds_mean_excluding_first': statistics.mean(x['seconds'] for x in R[0][1:n]),
                  'left_peak': max(x['peak_allocated_bytes'] for rank in L for x in rank),
                  'right_peak': max(x['peak_allocated_bytes'] for rank in R for x in rank), 'rows': rows}, indent=2))
