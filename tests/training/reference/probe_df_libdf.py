"""DeepFilterNet official libdf frontend and independent DFStream, one CPU update."""
import argparse
import ast
import copy
import json
import math
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / '.venvs/tsubame-libdf'))
from libdf import DF, erb, erb_norm, unit_norm
from tests.training.four_process.real_models import DeepFilterNet3, _df_loss
from tests.training.reference.common import gap, grad_report, weight_report


def official_norm_alpha(params):
    path = ROOT / 'src/se/official/deepfilternet/DeepFilterNet/df/utils.py'
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and
             node.name in ('get_norm_alpha', '_calculate_norm_alpha')]
    if len(nodes) != 2:
        raise RuntimeError('official libdf normalization functions unavailable')
    scope = {'math': math, 'ModelParams': lambda: params}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope['get_norm_alpha'](False)


class OfficialStream:
    def __init__(self, params):
        self.params = params
        self.df = DF(params.sr, params.fft_size, params.hop_size, params.nb_erb, params.min_nb_freqs)
        self.alpha = official_norm_alpha(params)
    def reset(self): self.df.reset()
    def analyze(self, waveform):
        return torch.from_numpy(self.df.analysis(waveform.detach().cpu().numpy())).to(waveform.device)
    def features(self, spectrum):
        arr = spectrum.detach().cpu().numpy()
        e = torch.from_numpy(erb_norm(erb(arr, self.df.erb_widths()), self.alpha)).unsqueeze(1).to(spectrum.device)
        f = torch.from_numpy(unit_norm(arr[..., :self.params.nb_df], self.alpha)).unsqueeze(1).to(spectrum.device)
        return torch.view_as_real(spectrum).unsqueeze(1), e, torch.view_as_real(f)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--input', required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--same-official-model', action='store_true')
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    data=torch.load(a.input,map_location='cpu',weights_only=False)
    clean=data['clean'][0].unsqueeze(0); noisy=data['noisy'][0].unsqueeze(0)
    torch.manual_seed(3)
    official=DeepFilterNet3(); official.configure({}); om=official.build('official',torch.device('cpu'))
    independent=DeepFilterNet3(); independent.configure({}); im=independent.build('independent',torch.device('cpu'))
    om['generator'].load_state_dict(im['generator'].state_dict())
    if a.same_official_model:
        im['generator'] = copy.deepcopy(om['generator'])
    initial={n:p.detach().clone() for n,p in om['generator'].named_parameters()}
    stream=OfficialStream(official._params)
    from src.se.models.deepfilternet3.dsp import DFStream
    params=official._params
    local=DFStream(params.sr,params.fft_size,params.hop_size,params.nb_erb,params.nb_df,
                   params.min_nb_freqs,params.norm_tau,torch.device('cpu'))
    official_window=torch.from_numpy(stream.df.fft_window())
    first=noisy[0,:params.hop_size]
    buffer=torch.cat((torch.zeros(params.fft_size-params.hop_size),first))*local.window
    first_ref=torch.from_numpy(stream.df.analysis(first.unsqueeze(0).numpy()))[0,0]
    first32=torch.fft.rfft(buffer)*local.scale
    first64=(torch.fft.rfft(buffer.double())*local.scale).to(torch.complex64)
    fft_probe={'window_max_abs':float((official_window-local.window).abs().max()),
               'erb_widths_equal':list(stream.df.erb_widths())==list(local.widths),
               'first_frame_float32_max_abs':float((first_ref-first32).abs().max()),
               'first_frame_float64_max_abs':float((first_ref-first64).abs().max())}
    from tests.training.four_process.real_models import _df_analyze
    op=_df_analyze(stream,official._params,noisy); ip=_df_analyze(independent._stream,independent._params,noisy)
    front={name:gap(torch.view_as_real(a) if a.is_complex() else a,
                    torch.view_as_real(b) if b.is_complex() else b)
           for name,a,b in zip(('spectrum','erb','df','complex'),op,ip)}
    oo=official.optimizers(om,0)['generator']; io=independent.optimizers(im,0)['generator']
    ow,ol=_df_loss('official',om['generator'],stream,official._params,clean,noisy,official._official_loss)
    iw,il=_df_loss('official' if a.same_official_model else 'independent', im['generator'],
                   independent._stream,independent._params,clean,noisy,
                   official._official_loss if a.same_official_model else None)
    ol.backward(); il.backward(); grad=grad_report(om['generator'],im['generator'],atol=1e-4)
    named_a=dict(om['generator'].named_parameters()); named_b=dict(im['generator'].named_parameters())
    if set(named_a) != set(named_b): raise RuntimeError('parameter names differ')
    if any((named_a[k].grad is None) != (named_b[k].grad is None) for k in named_a):
        raise RuntimeError('gradient presence differs')
    active=[k for k in sorted(named_a) if named_a[k].grad is not None]
    ga=torch.cat([named_a[k].grad.detach().flatten().double() for k in active])
    gb=torch.cat([named_b[k].grad.detach().flatten().double() for k in active])
    grad['global_relative_l2']=(ga-gb).norm().item()/ga.norm().item()
    grad['global_norm_official']=ga.norm().item(); grad['global_norm_independent']=gb.norm().item()
    torch.nn.utils.clip_grad_norm_(om['generator'].parameters(),1.0); torch.nn.utils.clip_grad_norm_(im['generator'].parameters(),1.0)
    oo.step(); io.step()
    worst = weight_report(om['generator'], im['generator'])
    name=worst['worst']; left=named_a[name]; right=named_b[name]
    idx=(left-right).abs().flatten().argmax()
    worst['grad_official']=float(left.grad.flatten()[idx]); worst['grad_independent']=float(right.grad.flatten()[idx])
    worst['initial']=float(initial[name].flatten()[idx])
    p1=torch.nn.Parameter(torch.tensor(worst['initial'])); p2=torch.nn.Parameter(torch.tensor(worst['initial']))
    opt1=torch.optim.AdamW([p1],lr=0.001,betas=(0.9,0.999),weight_decay=1e-12,amsgrad=True)
    opt2=torch.optim.AdamW([p2],lr=0.001,betas=(0.9,0.999),weight_decay=1e-12,amsgrad=True)
    p1.grad=torch.tensor(worst['grad_official']); p2.grad=torch.tensor(worst['grad_independent'])
    opt1.step(); opt2.step(); worst['predicted_adamw_difference']=abs(float(p1.detach()-p2.detach()))
    result={'input':a.input,'case':data['items'][0],'frontend':front,'fft_probe':fft_probe,'forward':gap(ow,iw),
            'official_loss':float(ol.detach()),'independent_loss':float(il.detach()),
            'loss_abs':abs(float(ol.detach()-il.detach())),'gradient':grad,'weight':worst,
            'scope':('CPU one update, official model and loss on both sides' if a.same_official_model else
                     'CPU one update, official vs independent model and loss') +
                    ', libdf 0.5.6 vs independent DFStream, independent synthesis for both losses'}
    with a.output.open('x') as f: json.dump(result,f,indent=2,default=str)
    print(a.output)

if __name__=='__main__': main()
