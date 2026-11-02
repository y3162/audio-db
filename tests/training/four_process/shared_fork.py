"""Create a new four-worker run from one saved checkpoint, preserving its state."""
import argparse
import copy
import json
import shutil
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def remap_optimizer(state, old, new):
    if len(old) != len(new) or len(set(old)) != len(old) or set(old) != set(new):
        raise ValueError('parameter names do not match')
    if len(state['param_groups']) != 1:
        raise ValueError('unexpected optimizer groups')
    ids = state['param_groups'][0]['params']
    if len(ids) != len(old):
        raise ValueError('unexpected optimizer parameter count')
    original = state['state']
    by_name = {name: original[identifier] for name, identifier in zip(old, ids) if identifier in original}
    state['state'] = {identifier: by_name[name] for name, identifier in zip(new, ids) if name in by_name}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('source', type=Path)
    p.add_argument('parent', type=Path)
    p.add_argument('--worker', default='official_a')
    p.add_argument('--fork-step', type=int, required=True)
    p.add_argument('--steps', type=int, required=True)
    p.add_argument('--remap-semamba', action='store_true',
                   help='map official optimizer moments to independent parameter names')
    p.add_argument('--remap-mp', action='store_true',
                   help='map MP-SENet optimizer moments by parameter name')
    a = p.parse_args()
    if a.steps <= a.fork_step:
        p.error('steps must exceed fork-step')
    checkpoint = torch.load(a.source / a.worker / f'fork-step-{a.fork_step}.pt',
                            map_location='cpu', weights_only=False)
    if checkpoint['data_position'] != a.fork_step:
        raise ValueError('checkpoint step mismatch')
    cfg = json.loads((a.source / 'config.json').read_text())
    if cfg['model'] not in ('semambapp', 'mp_senet'):
        raise ValueError('cross-implementation optimizer mapping is not implemented for this model')
    if cfg['model'] == 'semambapp' and not a.remap_semamba:
        raise ValueError('SEMamba++ cross-implementation forks require --remap-semamba')
    if cfg['model'] == 'mp_senet' and not a.remap_mp:
        raise ValueError('MP-SENet cross-implementation forks require --remap-mp')
    a.parent.mkdir(exist_ok=False)
    dst = a.parent / a.source.name
    dst.mkdir()
    for name in ('input.pt', 'environment.json'):
        shutil.copy2(a.source / name, dst / name)
    cfg['steps'] = a.steps
    (dst / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    independent_checkpoint = checkpoint
    if a.remap_semamba or a.remap_mp:
        if not a.worker.startswith('official_'):
            raise ValueError('remapping requires an official source')
        if a.remap_semamba:
            if cfg['model'] != 'semambapp': raise ValueError('wrong model for SEMamba++ mapping')
            from tests.training.four_process.real_models import SEMambaPP
            adapter = SEMambaPP(); adapter.configure(cfg)
            official = adapter.build('official', torch.device('cpu'))
            independent = adapter.build('independent', torch.device('cpu'))
            groups = (('generator', ('generator',)), ('discriminator', ('discriminator_band', 'discriminator_mrd')))
        else:
            if cfg['model'] != 'mp_senet': raise ValueError('wrong model for MP-SENet mapping')
            from tests.training.reference.compare_mp_senet import _hyper, _build
            independent_g, independent_d, official_g, official_d = _build(_hyper(), torch.device('cpu'))
            official = {'generator': official_g, 'discriminator': official_d}
            independent = {'generator': independent_g, 'discriminator': independent_d}
            groups = (('generator', ('generator',)), ('discriminator', ('discriminator',)))
        independent_checkpoint = copy.deepcopy(checkpoint)
        for opt_name, modules in groups:
            def names(models):
                return [module + '.' + name for module in modules
                        for name, _ in models[module].named_parameters()]
            old, new = names(official), names(independent)
            state = independent_checkpoint['optimizers'][opt_name]
            remap_optimizer(state, old, new)
    for name in ('official_a', 'official_b', 'independent_a', 'independent_b'):
        (dst / name).mkdir()
        selected = independent_checkpoint if name.startswith('independent_') else checkpoint
        selected['worker'] = name
        torch.save(selected, dst / name / 'checkpoint.pt')
    print(dst)

if __name__ == '__main__':
    main()
