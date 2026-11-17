"""Compare official and SEMamba++ joint feature-matching gradients on CUDA.

Run in a fresh process with CUDA_VISIBLE_DEVICES set; output is JSON lines.
The legacy path recreates per-sub-discriminator channel views. The audio path
uses the normal paired API with 2-D inputs, including its shared family view.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src/official/semambapp'))

from model.discriminator import (  # noqa: E402
    MultiResolutionDiscriminator as OfficialMRD,
    MultiScaleSubbandCQTDiscriminator as OfficialCQT,
    feature_loss,
)
from src.experiments.calc.spectral import feature_matching  # noqa: E402
from src.models.semambapp.discriminator import (  # noqa: E402
    MultiResolutionDiscriminator as AudioMRD,
    MultiScaleSubbandCQTDiscriminator as AudioCQT,
)


def compare(left, right):
    difference = (left - right).abs()
    li = left.contiguous().view(torch.int32).to(torch.int64)
    ri = right.contiguous().view(torch.int32).to(torch.int64)
    li = torch.where(li < 0, -2147483648 - li, li)
    ri = torch.where(ri < 0, -2147483648 - ri, ri)
    return {
        'max': float(difference.max()),
        'mean': float(difference.mean()),
        'mismatch': int((left != right).sum()),
        'ulp': int((li - ri).abs().max()),
        'nonfinite': int((~torch.isfinite(left)).sum() + (~torch.isfinite(right)).sum()),
    }


def legacy_family(module, clean, fake):
    pairs = [(discriminator(clean), discriminator(fake)) for discriminator in module.discriminators]
    real_rows = [pair[0] for pair in pairs]
    fake_rows = [pair[1] for pair in pairs]
    return (
        [row[0] for row in real_rows],
        [row[0] for row in fake_rows],
        [item for row in real_rows for item in row[1]],
        [item for row in fake_rows for item in row[1]],
    )


def run(kind, case, seed, modules):
    torch.manual_seed(seed)
    clean = torch.randn(1, 8000, device='cuda')
    fake = torch.randn(1, 8000, device='cuda', requires_grad=True)
    cqt, mrd = modules[:2] if kind == 'official' else modules[2:]
    if kind == 'official':
        qout = cqt(clean.unsqueeze(1), fake.unsqueeze(1))
        rout = mrd(clean.unsqueeze(1), fake.unsqueeze(1))
    elif kind == 'audio':
        qout = cqt(clean, fake)
        rout = mrd(clean, fake)
    else:
        qout = legacy_family(cqt, clean, fake)
        rout = legacy_family(mrd, clean, fake)
    if kind == 'official':
        lq = feature_loss(qout[2], qout[3])
        lr = feature_loss(rout[2], rout[3])
        maps = [item for group in qout[3] + rout[3] for item in group]
    else:
        lq = feature_matching(qout[2], qout[3], scale=2, reduction='sum')
        lr = feature_matching(rout[2], rout[3], scale=2, reduction='sum')
        maps = qout[3] + rout[3]
    loss = {'q': lq, 'r': lr, 'joint': lq + lr}[case]
    nodes = collections.Counter()
    seen = set()
    stack = [loss.grad_fn]
    while stack:
        node = stack.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        nodes[type(node).__name__] += 1
        stack.extend(item[0] for item in node.next_functions)
    gradient = torch.autograd.grad(loss, fake)[0]
    return loss.detach().cpu(), gradient.detach().cpu(), [item.detach().cpu() for item in maps], dict(nodes)


def main():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(17)
    oq, om, aq, am = OfficialCQT().cuda(), OfficialMRD().cuda(), AudioCQT().cuda(), AudioMRD().cuda()
    aq.load_state_dict(oq.state_dict())
    am.load_state_dict(om.state_dict())
    modules = (oq, om, aq, am)
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    for seed in (17, 23):
        results = {}
        for case in ('q', 'r', 'joint'):
            official = run('official', case, seed, modules)
            legacy = run('legacy', case, seed, modules)
            audio = run('audio', case, seed, modules)
            repeat = run('official', case, seed, modules)
            results[case] = (official[1], audio[1])
            nodes = [{k: v for k, v in row[3].items() if 'squeeze' in k.lower() or 'AddBackward' in k}
                     for row in (official, legacy, audio)]
            print(json.dumps({
                'seed': seed, 'case': case,
                'loss': [float(row[0]) for row in (official, legacy, audio)],
                'maps_diff': sum(not torch.equal(a, b) for a, b in zip(official[2], legacy[2], strict=True)),
                'audio_maps_diff': sum(not torch.equal(a, b) for a, b in zip(official[2], audio[2], strict=True)),
                'legacy': compare(official[1], legacy[1]), 'audio': compare(official[1], audio[1]),
                'repeat': compare(official[1], repeat[1]), 'nodes': nodes,
            }), flush=True)
        for index, kind in enumerate(('official', 'audio')):
            separate = results['q'][index] + results['r'][index]
            print(json.dumps({'seed': seed, 'case': 'separate_sum_vs_joint', 'kind': kind,
                              'stats': compare(separate, results['joint'][index])}), flush=True)


if __name__ == '__main__':
    main()
