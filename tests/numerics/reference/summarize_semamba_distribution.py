"""Compare final gradients within and across independent four-process trials."""
import argparse
import itertools
import json
from pathlib import Path
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('directories', type=Path, nargs='+')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    runs = []
    expected = None
    for directory in args.directories:
        cfg = json.loads((directory / 'config.json').read_text())
        signature = tuple(cfg.get(k) for k in ('model', 'seed', 'input_sha256', 'steps',
                          'scan_backend', 'official_width', 'scheduler_step', 'segment_size'))
        if expected is None:
            expected = signature
        elif signature != expected:
            raise ValueError('trial settings or input hash differ')
        for name in ('official_a', 'official_b', 'independent_a', 'independent_b'):
            item = torch.load(directory / name / 'result.pt', map_location='cpu', weights_only=False)
            runs.append((str(directory), name, item['snapshots'][-1]['grad']))
            del item
    out = []
    for (dir_a, name_a, grad_a), (dir_b, name_b, grad_b) in itertools.combinations(runs, 2):
        if set(grad_a) != set(grad_b):
            raise ValueError('gradient names differ')
        aa = bb = ab = dd = 0.0
        maximum = 0.0
        for key in grad_a:
            a, b = grad_a[key].double(), grad_b[key].double()
            aa += float((a*a).sum()); bb += float((b*b).sum()); ab += float((a*b).sum())
            dd += float(((a-b)**2).sum()); maximum = max(maximum, float((a-b).abs().max()))
        kind_a, kind_b = name_a.split('_')[0], name_b.split('_')[0]
        out.append({'a': [dir_a, name_a], 'b': [dir_b, name_b],
                    'kind': kind_a if kind_a == kind_b else 'cross',
                    'max_abs': maximum, 'norm_a': aa**0.5, 'norm_b': bb**0.5,
                    'relative_l2': (dd/aa)**0.5 if aa else None,
                    'cosine': ab/(aa*bb)**0.5 if aa and bb else None})
    with args.output.open('x') as stream:
        json.dump(out, stream, indent=2)
    print(args.output)

if __name__ == '__main__':
    main()
