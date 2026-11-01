"""Compare stored DDP gradients, reporting norm-relative differences and cosine."""
import json
import sys
import torch

a, b = (torch.load(path, map_location='cpu', weights_only=False) for path in sys.argv[1:3])

def normalize(key):
    if key.startswith('DistributedDataParallel.module.'):
        return key[len('DistributedDataParallel.module.'):]
    return key.split('.', 1)[1]

def summary(left, right, generator=False):
    x = {(normalize(k) if generator else k): v for k, v in left.items()}
    y = {(normalize(k) if generator else k): v for k, v in right.items()}
    missing = sorted(set(x) ^ set(y))
    aa = bb = dd = dot = maximum = 0.0
    for key in set(x) & set(y):
        l, r = x[key].double().flatten(), y[key].double().flatten()
        d = l - r
        maximum = max(maximum, d.abs().max().item())
        aa += l.dot(l).item(); bb += r.dot(r).item(); dd += d.dot(d).item(); dot += l.dot(r).item()
    return {'max_abs': maximum, 'relative_l2': (dd / max(aa, 1e-60)) ** 0.5,
            'cosine': dot / max((aa * bb) ** 0.5, 1e-60), 'missing': missing}

print(json.dumps({'generator': summary(a['generator_grad'], b['generator_grad'], True),
                  'discriminator': summary(a['discriminator_grad'], b['discriminator_grad']),
                  'rng_torch_equal': bool(torch.equal(a['rng']['torch'], b['rng']['torch'])),
                  'rng_cuda_equal': all(torch.equal(x, y) for x, y in zip(a['rng']['cuda'], b['rng']['cuda']))}, indent=2))
