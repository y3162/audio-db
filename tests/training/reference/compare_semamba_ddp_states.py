"""Compare generator and discriminator task checkpoints without modifying them."""
import json
import sys
import torch

a, b = (torch.load(path, map_location='cpu', weights_only=False) for path in sys.argv[1:3])
report = {}
for group in ('generator', 'discriminator_mrd', 'discriminator_band'):
    left, right = a[group], b[group]
    if set(left) != set(right):
        raise SystemExit(f'checkpoint keys differ in {group}')
    aa = bb = dd = dot = maximum = 0.0
    for key in left:
        x, y = left[key].double().flatten(), right[key].double().flatten()
        d = x - y
        maximum = max(maximum, d.abs().max().item())
        aa += x.dot(x).item(); bb += y.dot(y).item(); dd += d.dot(d).item(); dot += x.dot(y).item()
    report[group] = {'max_abs': maximum, 'relative_l2': (dd / max(aa, 1e-60)) ** 0.5,
                     'cosine': dot / max((aa * bb) ** 0.5, 1e-60)}
print(json.dumps(report, indent=2))
