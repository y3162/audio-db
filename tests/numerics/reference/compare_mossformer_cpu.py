"""CPU comparison against the pinned ClearerVoice source; writes no results files."""
import copy, json, sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src/se/official/clearervoice-studio/train/speech_enhancement'))
from models.mossformer_gan.discriminator import Discriminator as Official
from losses import loss as official_loss
from utils.misc import stft, power_compress, pesq_loss
from src.experiments.official.mossformergan.discriminator import Discriminator
from src.experiments.official.mossformergan.reference import MossFormerGANTask
from tests.numerics.test_model_tasks import _base

torch.set_num_threads(1)
torch.manual_seed(77)
def gap(a,b):
    a,b=a.detach().reshape(-1).double(),b.detach().reshape(-1).double()
    d=a-b
    return {'max_abs':d.abs().max().item(), 'relative_l2':(d.norm()/a.norm().clamp_min(1e-30)).item(), 'cosine':torch.nn.functional.cosine_similarity(a,b,dim=0).item(), 'norm_reference':a.norm().item(), 'norm_actual':b.norm().item()}
def flat(params): return torch.cat([p.detach().reshape(-1) for p in params])
def grads(module): return flat([p.grad for p in module.parameters() if p.grad is not None])

o=Official(16); m=Discriminator(16)
m.load_state_dict(o.state_dict(),strict=True)
assert list(o.state_dict()) == list(m.state_dict())
features=[{},{}]
for j, model in enumerate((o,m)):
    for name in ('cnn','att','pool','sigmoid'):
        getattr(model,name).register_forward_hook(lambda mod, inp, out, n=name, j=j: features[j].__setitem__(n,out.detach().clone()))
x=torch.rand(1,1,201,25,requires_grad=True); y=torch.rand_like(x,requires_grad=True)
x2=x.detach().clone().requires_grad_(); y2=y.detach().clone().requires_grad_()
rng=torch.get_rng_state(); a=o(x,y); torch.set_rng_state(rng); b=m(x2,y2)
((a - 1).square().mean()).backward(); ((b - 1).square().mean()).backward()
report={'keys':len(o.state_dict()), 'parameters':sum(p.numel() for p in o.parameters()), 'initial':gap(flat(o.parameters()),flat(m.parameters())), 'intermediate':{k:gap(features[0][k], features[1][k]) for k in features[0]}, 'forward':gap(a,b), 'parameter_grad':gap(grads(o),grads(m)), 'input_grad':gap(x.grad,x2.grad), 'generator_adversarial_grad':gap(y.grad,y2.grad)}
oo=torch.optim.AdamW(o.parameters(),lr=.001); mo=torch.optim.AdamW(m.parameters(),lr=.001); oo.step();mo.step();report['discriminator_step']=gap(flat(o.parameters()),flat(m.parameters()))
del o,m,oo,mo,x,y,x2,y2,a,b,features

config=_base(Path('/tmp'), 'mossformergan', variant='se_16k', architecture={'n_fft':400,'win_len':400,'hop':100,'num_channel':64},segment_size=8000,grad_clip=10.0)
config['optimizer']['lr']=.0005
config['optimizer']['betas']=[.9,.999]
task=MossFormerGANTask(config,torch.device('cpu')); og=copy.deepcopy(task.generator); od=Official(16);od.load_state_dict(task.discriminator.state_dict())
gen_out = [None, None]
def track(j):
    def hook(module, inputs, output):
        for value in output: value.retain_grad()
        gen_out[j] = output
    return hook
task.generator.register_forward_hook(track(0)); og.register_forward_hook(track(1))
ogopt=torch.optim.AdamW(og.parameters(),lr=.0005); odopt=torch.optim.AdamW(od.parameters(),lr=.001)
clean=torch.randn(1,8000)*.1; noisy=clean+torch.randn_like(clean)*.03
args=SimpleNamespace(batch_size=1,win_type='hamming',win_len=400,win_inc=100,fft_len=400)
# The official helper hardcodes CUDA. Preserve its PESQ and normalization but
# return the tensor on CPU so this reference comparison can run without a GPU.
def cpu_pesq(clean_rows, enhanced_rows):
    values = [pesq_loss(c, e) for c, e in zip(clean_rows, enhanced_rows)]
    if -1 in values: return None
    return torch.tensor([(value - 1) / 3.5 for value in values], dtype=torch.float32)
def capture(opt):
    saved={}
    orig=opt.step
    def step(*args,**kwargs):
        saved['grad']=flat([p.grad for group in opt.param_groups for p in group['params'] if p.grad is not None]).clone()
        return orig(*args,**kwargs)
    opt.step=step
    return saved
cg,cd=capture(task.optim_g),capture(task.optim_d)
rng=torch.get_rng_state()
losses=task.train_update([{'clean':clean,'noisy':noisy}],1)
torch.set_rng_state(rng)
c=torch.sqrt(noisy.shape[-1]/(noisy.square().sum(-1)+1e-6))
inputs=noisy*c.unsqueeze(-1)
out=og(power_compress(stft(inputs,args,center=True).float()).permute(0,1,3,2))
with patch.object(official_loss,'batch_pesq',side_effect=cpu_pesq):
    lg,ld=official_loss.loss_mossformergan_se_16k(args,inputs,clean,out,c,od,torch.device('cpu'))
def zero_manual():
    # ClearerVoice solver.py zeroes each AdamW after its step. The task uses
    # set_to_none=True. Leaving the previous .grad in place adds it to the next
    # backward, so steps 2 and 3 were not the same update.
    ogopt.zero_grad(set_to_none=True)
    odopt.zero_grad(set_to_none=True)
lg.backward();torch.nn.utils.clip_grad_norm_(og.parameters(),10.)
ld.backward();torch.nn.utils.clip_grad_norm_(od.parameters(),10.)
report['task']={'loss_g':gap(lg,torch.tensor(losses['generator'])), 'loss_d':gap(ld,torch.tensor(losses['discriminator'])), 'generator_grad':gap(grads(og),cg['grad']), 'discriminator_grad':gap(grads(od),cd['grad'])}
report['task']['generator_forward'] = [gap(a,b) for a,b in zip(gen_out[1],gen_out[0])]
report['task']['generator_output_grad'] = [gap(a.grad,b.grad) for a,b in zip(gen_out[1],gen_out[0])]
ogopt.step(); odopt.step(); zero_manual()
report['task']['generator_weights']=gap(flat(og.parameters()),flat(task.generator.parameters()))
report['task']['discriminator_weights']=gap(flat(od.parameters()),flat(task.discriminator.parameters()))
from src.experiments.official.mossformergan.models import MossFormerGANGenerator
yaml = MossFormerGANGenerator(seed=0, variant='se_16k', n_fft=400, win_len=400, hop=100, num_channel=64, sampling_rate=16000)
yaml.generator.load_state_dict(og.state_dict(), strict=True)
compressed = power_compress(stft(inputs,args,center=True).float()).permute(0,1,3,2)
rng=torch.get_rng_state()
same_weights = og(compressed)
torch.set_rng_state(rng)
yaml_out = yaml(compressed)
report['yaml_generator'] = {
    'class': 'src.experiments.official.mossformergan.models.MossFormerGANGenerator',
    'forward_vs_task_generator_same_weights': [gap(a,b) for a,b in zip(same_weights, yaml_out)],
}
extra=[]
for step in (2, 3):
    rng=torch.get_rng_state()
    losses=task.train_update([{'clean':clean,'noisy':noisy}],1)
    torch.set_rng_state(rng)
    out=og(compressed)
    with patch.object(official_loss,'batch_pesq',side_effect=cpu_pesq):
        lg,ld=official_loss.loss_mossformergan_se_16k(args,inputs,clean,out,c,od,torch.device('cpu'))
    zero_manual()
    lg.backward(); torch.nn.utils.clip_grad_norm_(og.parameters(),10.)
    ld.backward(); torch.nn.utils.clip_grad_norm_(od.parameters(),10.)
    record={'step':step,'loss_g':gap(lg,torch.tensor(losses['generator'])),'loss_d':gap(ld,torch.tensor(losses['discriminator'])), 'generator_grad':gap(grads(og),cg['grad']), 'discriminator_grad':gap(grads(od),cd['grad'])}
    ogopt.step(); odopt.step(); zero_manual()
    record['generator_weights']=gap(flat(og.parameters()),flat(task.generator.parameters()))
    record['discriminator_weights']=gap(flat(od.parameters()),flat(task.discriminator.parameters()))
    extra.append(record)
report['extra_steps']=extra
print(json.dumps(report,indent=2))
