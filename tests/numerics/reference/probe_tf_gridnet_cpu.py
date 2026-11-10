"""Supplemental synthetic CPU diagnostic; not a four-process comparison.

Run with a new output filename. Existing results are never overwritten.
"""
import sys, json, hashlib, time
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import torch
from tests.numerics.reference import compare_tf_gridnet as c
from tests.numerics.reference.common import environment
from src.experiments.calc.spectral import _rect_magnitude_l1

if '--legacy' in sys.argv:
    # Reproduce the previously shared alignment node for causal comparison.
    def legacy(target, estimate, windows, **kwargs):
        reference, predicted = target.float(), estimate.float()
        scale = (predicted * reference).sum(-1, keepdim=True) / (predicted.square().sum(-1, keepdim=True) + 1e-8)
        aligned = predicted * scale
        time_loss = (aligned - reference).abs().sum(-1)
        spectral = torch.zeros_like(time_loss)
        for window in windows:
            spectral = spectral + _rect_magnitude_l1(reference, aligned, window, window // 2)
        return (time_loss * .5 + .5 * (spectral / len(windows))).mean()
    c.multi_resolution_l1 = legacy
torch.set_num_threads(1)
torch.manual_seed(3)
clean=torch.randn(1,2048)
noisy=clean*.5+.05
h=lambda t: hashlib.sha256(t.numpy().tobytes()).hexdigest()
start=time.monotonic()
o,i=c._pair(torch.device('cpu'))
criterion=c._official_loss_class()(window_sz=list(c.WINDOWS),hop_sz=None,eps=1e-8,time_domain_weight=.5,normalize_variance=False,reduction='sum')
ow, iw = c._enhance(o, i, noisy)
loss_o = c._official_value(criterion, clean, ow)
loss_i = c.multi_resolution_l1(clean, iw, c.WINDOWS, normalize_variance=False)
loss_o.backward(); loss_i.backward()
layers = {}; all_o = []; all_i = []
for (name, po), (name_i, pi) in zip(o.named_parameters(), i.named_parameters()):
    assert name == name_i
    a = po.grad.detach().double().flatten(); b = pi.grad.detach().double().flatten()
    d = (a-b).abs(); idx = int(d.argmax()); norm = a.norm().item()
    layers[name] = {'max_abs': d[idx].item(), 'index': [int(x) for x in torch.unravel_index(torch.tensor(idx), po.shape)],
                    'official_value': a[idx].item(), 'independent_value': b[idx].item(),
                    'official_norm': norm, 'independent_norm': b.norm().item(),
                    'relative_l2': (a-b).norm().item() / norm if norm else None,
                    'cosine': torch.nn.functional.cosine_similarity(a, b, dim=0, eps=1e-30).item(),
                    'sign_mismatch': int(((a*b)<0).sum())}
    all_o.append(a); all_i.append(b)
a=torch.cat(all_o); b=torch.cat(all_i); d=(a-b).abs()
bins={}
for low, high in [(0,1e-8),(1e-8,1e-6),(1e-6,1e-4),(1e-4,1e-2),(1e-2,float('inf'))]:
    mask=(a.abs()>=low)&(a.abs()<high)
    bins[f'{low:g}..{high:g}']={'count':int(mask.sum()), 'max_abs':d[mask].max().item() if mask.any() else None,
        'sign_mismatch':int(((a[mask]*b[mask])<0).sum()),
        'cosine':torch.nn.functional.cosine_similarity(a[mask],b[mask],dim=0,eps=1e-30).item() if mask.any() else None}
max_name=max(layers,key=lambda name: layers[name]['max_abs'])
details={'global':{'max_abs':d.max().item(), 'relative_l2':(a-b).norm().item()/a.norm().item(),
         'cosine':torch.nn.functional.cosine_similarity(a,b,dim=0).item(), 'official_norm':a.norm().item(),
         'independent_norm':b.norm().item(), 'sign_mismatch':int(((a*b)<0).sum()), 'max_parameter':max_name},
         'magnitude_bins':bins,'layers':layers}
o.zero_grad(set_to_none=True); i.zero_grad(set_to_none=True)
r=c._step(o,i,c._adam(o),c._adam(i),criterion,clean,noisy)
r['gradient_detail']=details
out={'model':'tf_gridnet ms-snsd','kind':'supplemental same-process synthetic CPU one-step','independent_base_commit':'7d434c6169c2dee67d28201d73dd8b26bc5fe0a3','loss_graph':'legacy' if '--legacy' in sys.argv else 'working tree','official_commit':'0d136c5456d403cb2c0a4c13e76e8136e32e9b7a','environment':environment(torch.device('cpu')),'seed':3,'input_sha256':{'clean':h(clean),'noisy':h(noisy)},'case_id':None,'tolerances':{'forward_atol':1e-5,'loss_atol':1e-5,'grad_atol':1e-4,'large_cosine_min':.999},'result':r,'elapsed_seconds':time.monotonic()-start,'limitations':['synthetic data; not a named dataset','same-process diagnostic; no A/B repeat or scheduler comparison','no GPU']}
p=Path(sys.argv[1])
with p.open('x') as stream:
    json.dump(out, stream, indent=2, default=str)
    stream.write('\n')
print(p); print(json.dumps(r,default=str))
