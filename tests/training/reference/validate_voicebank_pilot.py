"""Eight fixed validation cases: pipeline and learning-curve diagnostic only."""
import json
import sys
import time
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.training.data.dataset import load_recipe_dataset
from src.se.models.semambapp.inference import build, _enhance_16k
from tests.training.reference.semamba_quality import score_waveform

out = Path(sys.argv[1])
if out.exists(): raise SystemExit('output already exists')
out.mkdir(parents=True)
ds = load_recipe_dataset(ROOT / 'data/db/semamba-voicebank-train-20261101.duckdb', 'voicebank_28_valid', role='validation', seed=1234, sample_rate=16000, segment_size=None, training=False)
indices = []
for speaker in ('p286', 'p287'):
    matches = [i for i, e in enumerate(ds.examples) if Path(e.recipe['sources']['clean']['path']).name.startswith(speaker + '_')]
    indices.extend(matches[:4])
if len(indices) != 8: raise RuntimeError('missing validation speakers')
samples = [ds._load(i) for i in indices]
(out / 'selection.json').write_text(json.dumps({'source': 'voicebank_28_valid', 'selection': 'first four registered per held-out validation speaker; pipeline diagnostic, not final quality judgment', 'case_ids': [x[2] for x in samples]}, indent=2))
model = build(None).to('cuda').eval()
for step in (4, 20, 40, 60, 80, 100):
    path = ROOT / ('data/results/training/semamba-voicebank-pilot-20261101/checkpoints/step_00000004.pt' if step == 4 else f'data/results/training/semamba-voicebank-100-20261101/checkpoints/step_{step:08d}.pt')
    while not path.is_file(): time.sleep(2)
    cp = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(cp['task']['generator'], strict=True)
    rows = []
    with torch.inference_mode():
        for clean, noisy, case_id in samples:
            enhanced = _enhance_16k(model, noisy.to('cuda')).cpu()
            rows.append({'case_id': case_id, 'clean': clean, 'noisy': noisy, 'enhanced': enhanced})
    torch.save(rows, out / f'waves_{step:04d}.pt')
    scored = [{'case_id': r['case_id'], 'noisy': score_waveform(r['clean'], r['noisy'], 16000), 'enhanced': score_waveform(r['clean'], r['enhanced'], 16000)} for r in rows]
    (out / f'scores_{step:04d}.json').write_text(json.dumps(scored, indent=2))
    print('validation', step, flush=True)
