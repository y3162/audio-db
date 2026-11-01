"""Summarize saved real-pair update comparisons including norm-relative errors."""
import json
import sys
from pathlib import Path
import torch

root = Path(sys.argv[1])
a = torch.load(root / f'update-{sys.argv[2]}.pt', map_location='cpu', weights_only=False)
b = torch.load(root / f'update-{sys.argv[3]}.pt', map_location='cpu', weights_only=False)

def compare(left, right, strip=False):
    if not isinstance(left, dict):
        left, right = {'value': left}, {'value': right}
    if strip:
        left = {k.split('.', 1)[1]: v for k, v in left.items()}
        right = {k.split('.', 1)[1]: v for k, v in right.items()}
    missing = sorted(set(left) ^ set(right))
    dot = aa = bb = dd = max_abs = 0.0
    significant_flips = significant_count = 0
    worst = ''
    for key in sorted(set(left) & set(right)):
        x, y = left[key].double().flatten(), right[key].double().flatten()
        delta = x - y
        threshold = x.abs().max().item() * 1e-3
        significant = x.abs() > max(threshold, 1e-8)
        significant_count += int(significant.sum())
        significant_flips += int(((x * y < 0) & significant).sum())
        peak = delta.abs().max().item()
        if peak > max_abs:
            max_abs, worst = peak, key
        dot += x.dot(y).item(); aa += x.dot(x).item(); bb += y.dot(y).item(); dd += delta.dot(delta).item()
    return {'max_abs': max_abs, 'worst': worst, 'relative_l2': (dd / max(aa, 1e-60)) ** 0.5,
            'cosine': dot / max((aa * bb) ** 0.5, 1e-60), 'missing': missing,
            'significant_sign_flips': significant_flips, 'significant_elements': significant_count,
            'significant_definition': 'abs(reference) > max(1e-8, 0.001 * tensor max abs)'}

report = {'input_match': a['input_sha256'] == b['input_sha256'],
          'losses': {k: [v, b['metrics'].get(k)] for k, v in a['metrics'].items()},
          'audio': compare(a['audio'], b['audio']),
          'generator_grad': compare(a['generator_grad'], b['generator_grad'], True),
          'discriminator_grad': compare(a['discriminator_grad'], b['discriminator_grad']),
          'generator_after': compare(a['generator_after'], b['generator_after'])}
print(json.dumps(report, indent=2))
