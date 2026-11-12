import json,collections,math,pathlib,random,hashlib
import numpy as np,soundfile as sf
D=pathlib.Path('research_voicebank_snr_shift_20261112');r=[json.loads(x) for x in (D/'recipes.jsonl').open()]; assert len(r)==12396 and len({(x['split'],x['utterance_id']) for x in r})==len(r)
ready=[x for x in r if x['status']=='ready'];rng=random.Random(42); selected=rng.sample(ready,min(40,len(ready)))
for kind in ('babble','ssn'):
 selected += [x for x in ready if x['noise_type']==kind][:2]
results=[]
for x in selected:
 clean,sr=sf.read(x['clean_path']); y,_=sf.read(x['noisy_path']);
 with sf.SoundFile(x['noise_path']) as f:f.seek(int(x['noise_start_48k'])); n=f.read(int(x['noise_end_exclusive_48k'])-int(x['noise_start_48k']))
 assert sr==48000 and len(n)==len(clean);g=float(x['clean_gain_g']);b=float(x['noise_gain_after_peak']);base=g*clean+b*n; shifted=g*clean+float(x['shifted_noise_gain'])*n
 rmse=float(np.sqrt(np.mean((base-y)**2))); snr=lambda p,q:10*np.log10(np.dot(p,p)/np.dot(q,q));delta=float(snr(g*clean,float(x['shifted_noise_gain'])*n)-snr(g*clean,b*n));assert rmse<2e-5 and abs(delta+15)<1e-9
 results.append(dict(id=x['utterance_id'],noise=x['noise_type'],rmse=rmse,component_snr_delta_db=delta,shifted_peak=float(np.max(abs(shifted))),reference='g * clean'))
(D/'verification.json').write_text(json.dumps({'method':'seed 42 random 40 ready plus first 2 each artificial; no output audio written','results':results},indent=2)+'\n');print('verified',len(results),'max_rmse',max(x['rmse'] for x in results),'max_peak',max(x['shifted_peak'] for x in results))
