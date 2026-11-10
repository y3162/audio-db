import tempfile, traceback, gc
from pathlib import Path
import torch
from src.se import load_enhancer
from tests.numerics.test_model_tasks import _base, _updated
from tests.numerics.support import paired_dataset, tiny_config
from src.experiments.calc.mp_senet import MPSENetTask
from src.experiments.calc.tf_gridnet import TFGridNetTask
from src.experiments.calc.semambapp import SEMambaTask
from src.experiments.calc.deepfilternet import DeepFilterNetTask
from src.experiments.calc.mossformergan import MossFormerGANTask

torch.set_num_threads(4)
print('gpu',torch.cuda.get_device_properties(0), flush=True)
failures = []
with tempfile.TemporaryDirectory() as d:
 root=Path(d)
 cases=[('mp_senet', MPSENetTask, tiny_config(root/'ckpt', loss='reconstruction'), 512, 16000), ('tf_gridnet',TFGridNetTask,_base(root,'tf_gridnet',variant='ms-snsd',architecture={'n_fft':32,'stride':16,'n_layers':1,'lstm_hidden_units':8,'attn_n_head':1,'attn_approx_qk_dim':8,'emb_dim':8,'emb_ks':1,'emb_hs':1},segment_size=800),800,16000),('semambapp',SEMambaTask,_base(root,'semambapp',architecture={'hid_feature':8,'num_tfmamba':1,'d_state':4,'d_conv':2,'expand':2,'n_fft':400},stft={'hop':100,'win':400,'disc_windows':[256,512]},segment_size=2048),2048,16000),('deepfilternet3',DeepFilterNetTask,_base(root,'deepfilternet3',architecture={},segment_size=4800,sampling_rate=48000,grad_clip=1.0),4800,48000),('mossformergan',MossFormerGANTask,_base(root,'mossformergan',variant='se_16k',architecture={'n_fft':400,'win_len':400,'hop':100,'num_channel':64},segment_size=8000,grad_clip=10.0),8000,16000)]
 for name,cls,cfg,frames,rate in cases:
  try:
   torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); task=cls(cfg,torch.device('cuda')); print(name,'constructed',torch.cuda.memory_allocated(),torch.cuda.memory_reserved(),flush=True)
   data=paired_dataset(root/name,1,frames=frames,segment_size=frames,seed=7)
   print(name,'update',_updated(task,data),'peak',torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved(),flush=True)
   ckpt=root/(name+'.pt'); torch.save(task.state_dict(),ckpt); task.load_state_dict(torch.load(ckpt,map_location='cpu',weights_only=False)); print(name,'checkpoint OK',flush=True)
   export=root/(name+'-inference.pt'); torch.save(task.export_for_inference(), export)
   model_cfg=cfg['model'] if name == 'mp_senet' else cfg['model'].get('architecture')
   enhancer=load_enhancer(name, export, device='cuda', config=model_cfg, variant=cfg['model'].get('variant'))
   enhanced=enhancer.enhance(torch.randn(frames), rate)
   assert enhanced.numel() == frames and torch.isfinite(enhanced).all()
   print(name,'inference OK',flush=True)
   del enhancer, enhanced
   del task,data; gc.collect()
  except Exception:
   failures.append(name)
   print(name,'FAILED',traceback.format_exc(),flush=True)
if failures:
 raise SystemExit('failed models: ' + ', '.join(failures))
