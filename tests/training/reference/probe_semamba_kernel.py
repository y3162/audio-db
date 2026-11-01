import json, torch, os
from mamba_ssm.modules.mamba_simple import Mamba
from causal_conv1d import causal_conv1d_fn
import selective_scan_cuda

torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.manual_seed(1234); torch.cuda.manual_seed_all(1234)
m=Mamba(d_model=48,d_state=16,d_conv=4,expand=2).cuda().train()
x=torch.randn(8,81,48,device='cuda',requires_grad=True)
r=[]
for i in range(5):
 m.zero_grad(set_to_none=True); x.grad=None
 y=m(x); loss=y.square().mean(); loss.backward(); torch.cuda.synchronize()
 r.append((y.detach().clone(),x.grad.detach().clone(),{n:p.grad.detach().clone() for n,p in m.named_parameters()}))
def compare(a,b):
 d=(a-b).abs(); return {'equal':torch.equal(a,b),'max_abs':float(d.max()),'different':int((d!=0).sum())}
out={'gpu':torch.cuda.get_device_name(), 'cuda':torch.version.cuda,'torch':torch.__version__, 'workspace':os.environ.get('CUBLAS_WORKSPACE_CONFIG'), 'fast_path':m.use_fast_path,'selective_scan':selective_scan_cuda.__file__,'comparisons':[]}
for i in range(1,5):
 out['comparisons'].append({'run':i,'forward':compare(r[0][0],r[i][0]),'input_gradient':compare(r[0][1],r[i][1]),'parameter_gradients':{n:compare(g,r[i][2][n]) for n,g in r[0][2].items()}})
print(json.dumps(out,indent=2))
