"""Compare TF-GridNet full-batch and accumulated updates on fixed CPU inputs."""
import argparse
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tests.numerics.test_model_tasks import _base
from src.experiments.official.tf_gridnet.reference import TFGridNetTask


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--input', type=Path, help='prepared VoiceBank input.pt; use official width')
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    architecture = {'n_fft': 32, 'stride': 16, 'n_layers': 1, 'lstm_hidden_units': 8,
                    'attn_n_head': 1, 'attn_approx_qk_dim': 8, 'emb_dim': 8, 'emb_ks': 1, 'emb_hs': 1}
    if args.input:
        architecture = {}
    config = _base(Path('/tmp'), 'tf_gridnet', variant='ms-snsd', architecture=architecture,
                   segment_size=2048, scheduler={'name': 'plateau', 'factor': 0.7, 'patience': 1})
    config['optimizer'] = {'name': 'adam', 'lr': 0.001, 'betas': [0.9, 0.999], 'weight_decay': 0.0}
    config['grad_clip'] = 5.0
    torch.manual_seed(5); a = TFGridNetTask(config, torch.device('cpu')); b = TFGridNetTask(config, torch.device('cpu'))
    b.generator.load_state_dict(a.generator.state_dict()); initial = {n: p.detach().clone() for n, p in a.generator.named_parameters()}
    if args.input:
        data = torch.load(args.input, map_location='cpu', weights_only=False)
        clean = torch.stack(data['clean'][:2]); noisy = torch.stack(data['noisy'][:2])
    else:
        torch.manual_seed(9); clean = torch.randn(2, 2048); noisy = torch.randn(2, 2048)
    a.train_update([{'clean': clean, 'noisy': noisy}], 1)
    b.train_update([{'clean': clean[:1], 'noisy': noisy[:1]}, {'clean': clean[1:], 'noisy': noisy[1:]}], 2)
    A = torch.cat([p.grad.flatten().double() for p in a.generator.parameters()]); B = torch.cat([p.grad.flatten().double() for p in b.generator.parameters()])
    worst = {'difference': -1}
    for (name, left), (_, right) in zip(a.generator.named_parameters(), b.generator.named_parameters(), strict=True):
        delta = (left-right).abs(); value = delta.max().item()
        if value > worst['difference']:
            index = delta.flatten().argmax(); ga = left.grad.flatten()[index].item(); gb = right.grad.flatten()[index].item()
            param = initial[name].flatten()[index].item()
            worst = {'name': name, 'difference': value, 'grad_a': ga, 'grad_b': gb, 'initial': param}
            p1 = torch.nn.Parameter(torch.tensor(param)); p2 = torch.nn.Parameter(torch.tensor(param))
            o1 = torch.optim.Adam([p1], lr=config['optimizer']['lr'], betas=tuple(config['optimizer']['betas']),
                                   weight_decay=config['optimizer']['weight_decay'])
            o2 = torch.optim.Adam([p2], lr=config['optimizer']['lr'], betas=tuple(config['optimizer']['betas']),
                                   weight_decay=config['optimizer']['weight_decay'])
            p1.grad = torch.tensor(ga); p2.grad = torch.tensor(gb); o1.step(); o2.step()
            worst['adam_predicted_difference'] = abs(p1.item()-p2.item())
    torch.manual_seed(10); q = torch.randn(1, 7, 8, dtype=torch.float64); k = torch.randn(1, 7, 8, dtype=torch.float64)
    beta = torch.randn(1, 1, 8, dtype=torch.float64, requires_grad=True)
    v = torch.randn(1, 7, 3, dtype=torch.float64)
    output = torch.softmax(q @ (k + beta).transpose(1, 2), dim=-1) @ v
    reference = torch.softmax(q @ k.transpose(1, 2), dim=-1) @ v
    output.sum().backward()
    report = {'gradient_norm_a': A.norm().item(), 'gradient_norm_b': B.norm().item(),
              'gradient_max_abs': (A-B).abs().max().item(), 'gradient_relative_l2': (A-B).norm().item()/A.norm().item(),
              'worst_weight': worst, 'optimizer': config['optimizer'],
              'key_shift_float64': {'output_max_abs': (output-reference).abs().max().item(), 'beta_gradient_max_abs': beta.grad.abs().max().item()},
              'scope': ('CPU VoiceBank official width' if args.input else 'CPU synthetic reduced width') +
                       ', full batch 2 versus two micro-batches',
              'input': str(args.input) if args.input else None}
    with args.output.open('x') as stream: json.dump(report, stream, indent=2)
    print(args.output)

if __name__ == '__main__': main()
