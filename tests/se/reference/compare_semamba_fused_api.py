"""Compare checkpoint-backed public inference on one saved 3 s waveform."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch
from src.se import load_enhancer

root = Path(__file__).resolve().parents[3]
waves = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
wave = waves['noisy'][0, :47993]
checkpoint = root / 'data/checkpoints/semambapp/semambapp.pth'
reference = load_enhancer('semambapp', checkpoint, device='cuda')
fused = load_enhancer('semambapp', checkpoint, device='cuda', config={'scan_backend': 'fused'})
a = reference.enhance(wave, 16000)
b = fused.enhance(wave, 16000)
diff = (a - b).double()
print(json.dumps({'samples': wave.numel(), 'reference_samples': a.numel(), 'fused_samples': b.numel(),
                  'max_abs': diff.abs().max().item(),
                  'relative_l2': (diff.norm() / a.double().norm()).item(),
                  'cosine': torch.nn.functional.cosine_similarity(a.double(), b.double(), dim=0).item()}))
