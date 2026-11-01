"""Diagnostic only: compare pretrained models on previously rendered extended recipes.

These modified recipes are not a registered named evaluation set. This probe is
not a substitute for the required named-set evaluation or a held-out test.
"""
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'src/se/official/semambapp')]
from model.semambapp import SEMambapp
from model.stfts import mag_phase_stft, mag_phase_istft
from src.se.models.semambapp.inference import build, _enhance_16k
from tests.training.reference.compare_semamba_step import _cfg

out = Path(sys.argv[1])
if out.exists():
    raise SystemExit('output already exists')
out.mkdir(parents=True)
device = torch.device('cuda')
weights = torch.load(ROOT / 'data/checkpoints/semambapp/semambapp.pth', map_location='cpu', weights_only=True)['generator']
a = SEMambapp(_cfg(48, 4, 16, 4)).to(device).eval()
b = build(None).to(device).eval()
la = a.load_state_dict(weights, strict=True)
lb = b.load_state_dict(weights, strict=True)
saved = torch.load(ROOT / 'data/results/training/semamba-repro-20261101/run2/validation.pt', map_location='cpu', weights_only=False)
rows = []
with torch.inference_mode():
    for u in saved['utterances']:
        clean, noisy = u['clean'], u['noisy']
        x = noisy.to(device)
        peak = x.abs().max()
        x = x / (peak + 1e-9)
        mag, phase, _ = mag_phase_stft(x[None], 400, 100, 400, 0.3)
        am, ap, _ = a(mag, phase)
        official = (mag_phase_istft(am, ap, 400, 100, 400, 0.3)[0] * peak).cpu()
        independent = (_enhance_16k(b, x) * peak).cpu()
        rows.append(dict(case_id=u['case_id'], clean=clean, noisy=noisy, official=official, independent=independent, frames=mag.shape[-1], peak=float(peak)))
        print(u['case_id'], noisy.numel(), mag.shape[-1], official.numel(), independent.numel(), flush=True)
        del x, mag, phase, am, ap
        torch.cuda.empty_cache()
torch.save(dict(sample_rate=16000, utterances=rows), out / 'waves.pt')
(out / 'metadata.json').write_text(json.dumps(dict(official=str(la), independent=str(lb), device=torch.cuda.get_device_name(), validation='previously rendered extended recipes, not a registered named set or held-out VoiceBank', normalization='official peak + 1e-9', resampling='none; existing 16 kHz rendered input'), indent=2))
