import json,pathlib,numpy as np,soundfile as sf
D=pathlib.Path('research_voicebank_snr_shift_20261112');rows=[json.loads(l) for l in (D/'recipes.jsonl').open()];sel=[x for x in rows if float(x['clean_gain_g'])!=1];res=[]
for r in sel:
 x,_=sf.read(r['clean_path']);y,_=sf.read(r['noisy_path']);
 with sf.SoundFile(r['noise_path']) as f:f.seek(int(r['noise_start_48k']));n=f.read(len(x))
 g=float(r['clean_gain_g']);b=float(r['noise_gain_after_peak']); residual=y-x;pure=b*n;pred=(g-1)*x+pure
 res.append({'id':r['utterance_id'],'g':g,'naive_residual_vs_noise_rmse':float(np.sqrt(np.mean((residual-pure)**2))),'identity_residual_rmse':float(np.sqrt(np.mean((residual-pred)**2)))})
(D/'residual_validation.json').write_text(json.dumps({'count':len(res),'cases':res},indent=2)+'\n');print('count',len(res),'max naive',max(x['naive_residual_vs_noise_rmse'] for x in res),'max identity',max(x['identity_residual_rmse'] for x in res))
