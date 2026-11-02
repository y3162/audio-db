"""Compare TF-GridNet against ESPnet's real STFT encoder/decoder from Git.

Uses built-in complex tensors. The temporary torch_complex and typeguard shims
only satisfy unused legacy-complex imports and runtime type annotations.
"""
import argparse
import copy
import json
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
ESP = ROOT / 'src/se/official/espnet'
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--input', required=True, help='four-process input.pt containing VoiceBank clean/noisy')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--steps', type=int, default=1)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    commit = subprocess.check_output(['git', '-C', str(ESP), 'rev-parse', 'HEAD'], text=True).strip()
    with tempfile.TemporaryDirectory() as temp:
        paths = ['espnet2/enh/encoder', 'espnet2/enh/decoder', 'espnet2/enh/separator',
                 'espnet2/enh/layers', 'espnet2/layers', 'espnet2/legacy/nets/pytorch_backend',
                 'espnet2/torch_utils']
        archive = subprocess.Popen(['git', '-C', str(ESP), 'archive', 'HEAD', *paths], stdout=subprocess.PIPE)
        subprocess.run(['tar', '-x', '-C', temp], stdin=archive.stdout, check=True)
        archive.stdout.close()
        if archive.wait() != 0:
            raise RuntimeError('git archive failed')
        sys.path.insert(0, temp)
        tc = types.ModuleType('torch_complex'); tc.__path__ = []
        tensor = types.ModuleType('torch_complex.tensor')
        class UnusedLegacyComplex: pass
        tensor.ComplexTensor = UnusedLegacyComplex
        functional = types.ModuleType('torch_complex.functional')
        tc.tensor, tc.functional = tensor, functional
        sys.modules.update({'torch_complex': tc, 'torch_complex.tensor': tensor,
                            'torch_complex.functional': functional})
        typeguard = types.ModuleType('typeguard'); typeguard.typechecked = lambda fn: fn
        sys.modules['typeguard'] = typeguard
        from espnet2.enh.separator.tfgridnet_separator import TFGridNet
        from tests.training.reference.compare_tf_gridnet import ARCHITECTURE, _adam, _step, _official_loss_class
        from tests.training.reference.common import environment, optim_report
        from src.se import build_model
        torch.manual_seed(3)
        official = TFGridNet(input_dim=0, n_srcs=1, use_builtin_complex=True, **ARCHITECTURE).to(args.device)
        independent = build_model('tf_gridnet', ARCHITECTURE, variant='ms-snsd').to(args.device)
        independent.load_state_dict(official.state_dict(), strict=True)
        criterion = _official_loss_class()(window_sz=[256, 512, 768, 1024], hop_sz=None,
                                           eps=1e-8, time_domain_weight=0.5,
                                           normalize_variance=False, reduction='sum')
        data = torch.load(args.input, map_location='cpu', weights_only=False)
        if not 1 <= args.steps <= len(data['clean']):
            raise ValueError('steps must be covered by the prepared input')
        spectra = {}
        def capture(name):
            def hook(_module, inputs):
                spectra[name] = inputs[0].detach().cpu()
            return hook
        official.conv.register_forward_pre_hook(capture('official'))
        independent.conv.register_forward_pre_hook(capture('independent'))
        optim_o, optim_i = _adam(official), _adam(independent)
        history = []
        checkpoint = None
        for index in range(args.steps):
            clean, noisy = (data['clean'][index].unsqueeze(0).to(args.device),
                            data['noisy'][index].unsqueeze(0).to(args.device))
            result = _step(official, independent, optim_o, optim_i, criterion, clean, noisy)
            result['preprocessed_conv_input_max_abs'] = float(
                (spectra['official'] - spectra['independent']).abs().max())
            history.append(result)
            if args.steps >= 4 and index == 1:
                checkpoint = copy.deepcopy((official.state_dict(), independent.state_dict(),
                                            optim_o.state_dict(), optim_i.state_dict(), torch.get_rng_state(),
                                            torch.cuda.get_rng_state(args.device) if args.device.startswith('cuda') else None))
        resume = None
        if checkpoint is not None:
            resumed_o = TFGridNet(input_dim=0, n_srcs=1, use_builtin_complex=True, **ARCHITECTURE).to(args.device)
            resumed_i = build_model('tf_gridnet', ARCHITECTURE, variant='ms-snsd').to(args.device)
            resumed_o.load_state_dict(checkpoint[0]); resumed_i.load_state_dict(checkpoint[1])
            resumed_optim_o, resumed_optim_i = _adam(resumed_o), _adam(resumed_i)
            resumed_optim_o.load_state_dict(checkpoint[2]); resumed_optim_i.load_state_dict(checkpoint[3])
            torch.set_rng_state(checkpoint[4])
            if checkpoint[5] is not None:
                torch.cuda.set_rng_state(checkpoint[5], args.device)
            for index in range(2, args.steps):
                clean, noisy = (data['clean'][index].unsqueeze(0).to(args.device),
                                data['noisy'][index].unsqueeze(0).to(args.device))
                _step(resumed_o, resumed_i, resumed_optim_o, resumed_optim_i, criterion, clean, noisy)
            resume = {name: max(float((a - b).abs().max()) for a, b in zip(
                original.parameters(), restarted.parameters(), strict=True))
                for name, original, restarted in (('official', official, resumed_o),
                                                   ('independent', independent, resumed_i))}
            resume['official_optimizer'] = optim_report(optim_o, resumed_optim_o)
            resume['independent_optimizer'] = optim_report(optim_i, resumed_optim_i)
        report = {'official_commit': commit, 'input': args.input, 'device': args.device,
                  'length': noisy.shape[-1], 'cases': data['items'][:args.steps], 'steps': history,
                  'resume_weight_max_abs': resume,
                  'environment': environment(torch.device(args.device)),
                  'seed': 3, 'architecture': ARCHITECTURE,
                  'optimizer': {'name': 'Adam', 'lr': 0.001, 'betas': [0.9, 0.999],
                                'eps': 1e-8, 'weight_decay': 0, 'grad_clip': 5.0},
                  'scheduler': 'not stepped in these short updates',
                  'note': 'Official STFT encoder/decoder; legacy complex import shim, built-in complex path.'}
        with output.open('x') as stream:
            json.dump(report, stream, indent=2, default=str)
        print(output)


if __name__ == '__main__':
    main()
