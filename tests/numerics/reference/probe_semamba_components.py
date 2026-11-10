import json, torch
from causal_conv1d import causal_conv1d_fn
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

torch.manual_seed(1234); torch.cuda.manual_seed_all(1234)
torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False

def compare(a,b):
 d=(a-b).abs(); return {'equal':torch.equal(a,b),'max_abs':float(d.max()),'different':int((d!=0).sum())}
def repeat(fn, inputs):
 rows=[]
 for i in range(5):
  xs=[x.detach().clone().requires_grad_() for x in inputs]
  y=fn(*xs); y.square().mean().backward(); torch.cuda.synchronize()
  rows.append((y.detach(),[x.grad.detach() for x in xs]))
 return [{'forward':compare(rows[0][0],row[0]),'gradients':[compare(a,b) for a,b in zip(rows[0][1],row[1])]} for row in rows[1:]]
d='cuda'; B,D,L,N=8,96,81,16
conv=[torch.randn(B,D,L,device=d),torch.randn(D,4,device=d),torch.randn(D,device=d)]
scan=[torch.randn(B,D,L,device=d),torch.rand(B,D,L,device=d),-torch.rand(D,N,device=d),torch.randn(B,N,L,device=d),torch.randn(B,N,L,device=d),torch.randn(D,device=d)]
out={}
for deterministic in (False,True):
 torch.use_deterministic_algorithms(deterministic)
 for name,fn,inputs in [('causal_conv1d',lambda x,w,b: causal_conv1d_fn(x,w,b,activation='silu'),conv),('selective_scan',lambda u,delta,A,B,C,D: selective_scan_fn(u,delta,A,B,C,D,delta_softplus=True),scan)]:
  try: out[f'{name}_deterministic_{deterministic}']=repeat(fn,inputs)
  except Exception as e: out[f'{name}_deterministic_{deterministic}']={'error':str(e)}
print(json.dumps(out,indent=2))
