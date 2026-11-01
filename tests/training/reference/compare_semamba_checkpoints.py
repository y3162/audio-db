"""Compare two saved SEMamba checkpoints, including optimizer and RNG state."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch


def compare(a, b):
    report = {'equal': True, 'tensor_count': 0, 'different_tensors': 0, 'max_abs': 0.0, 'worst_path': None, 'structural_differences': []}
    def walk(x, y, path):
        if torch.is_tensor(x) or isinstance(x, np.ndarray):
            if not (torch.is_tensor(y) or isinstance(y, np.ndarray)):
                report['equal'] = False; report['structural_differences'].append(path); return
            x = torch.as_tensor(x).detach().cpu(); y = torch.as_tensor(y).detach().cpu()
            report['tensor_count'] += 1
            if x.shape != y.shape or x.dtype != y.dtype:
                report['equal'] = False; report['structural_differences'].append(path); return
            if not torch.equal(x, y):
                report['equal'] = False; report['different_tensors'] += 1
                if x.numel():
                    delta = float((x.to(torch.float64) - y.to(torch.float64)).abs().max())
                    if not np.isfinite(delta):
                        report['structural_differences'].append(path + ':nonfinite')
                    elif delta > report['max_abs']:
                        report['max_abs'] = delta; report['worst_path'] = path
        elif isinstance(x, dict) and isinstance(y, dict):
            if x.keys() != y.keys():
                report['equal'] = False; report['structural_differences'].append(path + ':keys')
            for key in x.keys() & y.keys(): walk(x[key], y[key], f'{path}.{key}')
        elif isinstance(x, (tuple, list)) and isinstance(y, (tuple, list)):
            if len(x) != len(y):
                report['equal'] = False; report['structural_differences'].append(path + ':length')
            for i, (left, right) in enumerate(zip(x, y)): walk(left, right, f'{path}[{i}]')
        elif type(x) != type(y) or x != y:
            report['equal'] = False; report['structural_differences'].append(path)
    walk(a, b, 'root')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('left', type=Path); p.add_argument('right', type=Path); p.add_argument('output', type=Path)
    args = p.parse_args()
    if args.output.exists(): raise SystemExit('refusing to overwrite output')
    left = torch.load(args.left, map_location='cpu', weights_only=False)
    right = torch.load(args.right, map_location='cpu', weights_only=False)
    keys = ['updates_completed','generator','cqt','mrd','optim_g','optim_d','scheduler_g','scheduler_d','rng']
    result = {'left':str(args.left),'right':str(args.right),'groups':{k:compare(left[k],right[k]) for k in keys}}
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__': main()
