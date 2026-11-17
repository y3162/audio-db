"""Run in a fresh CUDA process with mode: before, after, off, or fixed.

Fixed is diagnostic only; it does not alter production autotuning.
"""
import os, sys, json, hashlib
import torch
mode=sys.argv[1]
if mode not in {'before', 'after', 'off', 'fixed'}:
 raise SystemExit(f'unknown mode: {mode}')
if mode=='before': torch.use_deterministic_algorithms(True, warn_only=True)
from mamba_ssm.ops.triton import layer_norm
from mamba_ssm.ops.triton.layer_norm import RMSNorm
if mode=='after': torch.use_deterministic_algorithms(True, warn_only=True)
ks=(layer_norm._layer_norm_fwd_1pass_kernel,layer_norm._layer_norm_bwd_kernel)
if mode=='fixed':
 import triton
 for k in ks: k.configs=[triton.Config({},num_warps=1)]
torch.manual_seed(23)
x=torch.randn(100,81,48,device='cuda',requires_grad=True); model=RMSNorm(48).cuda(); dy=torch.randn_like(x)
y=model(x); y.backward(dy)
def sha(t): return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()
print(json.dumps(dict(mode=mode, configs=[[c.num_warps for c in k.configs] for k in ks],selected=[None if k.best_config is None else k.best_config.num_warps for k in ks],output=sha(y),grad=sha(x.grad),weight_grad=sha(model.weight.grad))),flush=True)
