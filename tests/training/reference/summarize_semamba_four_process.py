"""Summarize component gradients from a completed four-process result."""
import argparse
import itertools
import json
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    runs = {}
    for name in ('official_a', 'official_b', 'independent_a', 'independent_b'):
        data = torch.load(args.directory / name / 'result.pt', map_location='cpu', weights_only=False)
        runs[name] = data['snapshots'][-1]['grad']
        del data
    result = {}
    for left, right in itertools.combinations(runs, 2):
        groups = {}
        for group in ('generator', 'discriminator_band', 'discriminator_mrd'):
            keys = [k for k in runs[left] if k.startswith(group + '.')]
            if not keys:
                continue
            aa = bb = dot = diff = 0.0
            maximum = 0.0
            for key in keys:
                a, b = runs[left][key].double(), runs[right][key].double()
                aa += float(torch.sum(a*a)); bb += float(torch.sum(b*b)); dot += float(torch.sum(a*b))
                diff += float(torch.sum((a-b)**2)); maximum = max(maximum, float((a-b).abs().max()))
            groups[group] = {'norm_a': aa**0.5, 'norm_b': bb**0.5,
                             'max_abs': maximum, 'relative_l2': (diff/aa)**0.5 if aa else None,
                             'cosine': dot/(aa*bb)**0.5 if aa and bb else None}
        result[left + ' vs ' + right] = groups
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
    print(args.output)

if __name__ == '__main__':
    main()
