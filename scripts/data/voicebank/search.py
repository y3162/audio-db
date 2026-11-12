import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import csv,sys,json,pathlib,hashlib,concurrent.futures
import numpy as np,soundfile as sf
from scipy.signal import fftconvolve,resample_poly,correlate
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from asl_P56_fast import asl_P56
D=pathlib.Path('research_voicebank_snr_shift_20261112')
rows=[r for r in csv.DictReader(open('research_voicebank_integrated_20261112/utterances.csv')) if r['obs_noise_type'] in ('babble','ssn')]
noise={k:sf.read(D/(k+'.wav'))[0] for k in ('babble','ssn')}; n16={k:resample_poly(v,1,3) for k,v in noise.items()}; sq={k:np.r_[0.,np.cumsum(v*v)] for k,v in n16.items()}
def run(r):
 try:
  k=r['obs_noise_type'];x,sr=sf.read(r['obs_clean_asset']);y,_=sf.read(r['obs_noisy_asset']);m=len(x);q=resample_poly(y-x,1,3);a=n16[k];e=sq[k][len(q):]-sq[k][:-len(q)];c=fftconvolve(a,q[::-1],mode='valid');score=c*c/np.maximum(e,1e-30);i=int(np.argmax(score));score[max(0,i-len(q)):i+len(q)+1]=0;second=float(np.sqrt(max(score)/np.dot(q,q)));lo=max(0,i*3-8);hi=min(len(noise[k])-m,i*3+8);z=noise[k][lo:hi+m];cc=fftconvolve(z,(y-x)[::-1],mode='valid');ee=np.r_[0.,np.cumsum(z*z)];ee=ee[m:]-ee[:-m];j=int(np.argmax(cc*cc/np.maximum(ee,1e-30)));start=lo+j;n=noise[k][start:start+m]
  rawpath=pathlib.Path('data/corpora/VCTK/wav48')/r['obs_utterance_id'].split('_')[0]/(r['obs_utterance_id']+'.wav');raw,_=sf.read(rawpath);offset=int(np.argmax(abs(correlate(raw,x,mode='valid',method='fft'))));raw=raw[offset:offset+m];raw=raw*np.max(abs(x))/np.max(abs(raw));ms,_,_=asl_P56(raw,sr,16);beta=np.sqrt(ms/np.mean(n*n)/10**(float(r['obs_original_snr_db'])/10));v=x+beta*n;g=1.;steps=0
  for factor in (1.1,1.2,1.3,1.4):
   if max(abs(v))*g<=1:break
   g/=factor;steps+=1
  diff=g*v-y;fit=np.linalg.lstsq(np.column_stack((x,n)),y,rcond=None)[0];ferr=y-fit[0]*x-fit[1]*n
  return dict(id=r['obs_utterance_id'],kind=k,start=start,end=start+m,ncc16=float(np.sqrt(max(c[i]*c[i]/max(e[i],1e-30),0))/np.linalg.norm(q)),second_ncc16=second,beta=float(beta),g=g,steps=steps,rmse=float(np.sqrt(np.mean(diff*diff))),max_abs=float(np.max(abs(diff))),fit_alpha=float(fit[0]),fit_beta=float(fit[1]),fit_rmse=float(np.sqrt(np.mean(ferr*ferr))),clean_sha256=hashlib.sha256(pathlib.Path(r['obs_clean_asset']).read_bytes()).hexdigest(),raw_path=str(rawpath),raw_sha256=hashlib.sha256(rawpath.read_bytes()).hexdigest(),raw_alignment_offset=offset,p56_ms=float(ms),status='ok')
 except Exception as ex:return dict(id=r['obs_utterance_id'],kind=r['obs_noise_type'],status='error',error=repr(ex))
if __name__=='__main__':
 with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool,open(D/'artificial_search.jsonl','w') as f:
  for i,v in enumerate(pool.map(run,rows,chunksize=1)):
   f.write(json.dumps(v)+'\n');f.flush()
   if i%100==0:print(i,v,flush=True)
