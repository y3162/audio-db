"""Research candidate: no noisy waveform required; refuses overwriting output."""
import argparse,os,pathlib,sys
import numpy as np,soundfile as sf
from scipy.signal import correlate
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from asl_P56_fast import asl_P56
p=argparse.ArgumentParser();p.add_argument('--clean',required=True);p.add_argument('--vctk',required=True);p.add_argument('--noise',required=True);p.add_argument('--start',required=True,type=int);p.add_argument('--snr',required=True,type=float);p.add_argument('--output',required=True);a=p.parse_args();dest=pathlib.Path(a.output)
if dest.exists():raise FileExistsError(dest)
x,sr=sf.read(a.clean);raw,sr2=sf.read(a.vctk)
if x.ndim!=1 or raw.ndim!=1 or len(raw)<len(x) or sr!=sr2:raise ValueError('incompatible clean/VCTK')
offset=int(np.argmax(abs(correlate(raw,x,mode='valid',method='fft'))));r=raw[offset:offset+len(x)];raw_norm=r*np.max(abs(x))/np.max(abs(r))
if np.sqrt(np.mean((raw_norm-x)**2))>1e-4:raise ValueError('VCTK alignment failed')
with sf.SoundFile(a.noise) as f:
 if f.samplerate!=sr:raise ValueError('noise sample rate mismatch')
 f.seek(a.start);n=f.read(len(x))
if len(n)!=len(x) or n.ndim!=1:raise ValueError('noise length/channel mismatch')
ms,_,_=asl_P56(raw_norm,sr,16);beta=10**((10*np.log10(ms)-a.snr)/20)/np.sqrt(np.mean(n*n));z=x+beta*n
for factor in (1.1,1.2,1.3,1.4):
 if np.max(abs(z))<=1:break
 z/=factor
if np.max(abs(z))>1:raise ValueError('unverified peak boundary')
q=np.clip(np.rint(z*32768),-32768,32767).astype(np.int16)
with dest.open('xb') as f:sf.write(f,q,sr,format='WAV',subtype='PCM_16')
print('vctk_offset=',offset,'level_db=',10*np.log10(ms),'beta=',beta,'output=',dest)
