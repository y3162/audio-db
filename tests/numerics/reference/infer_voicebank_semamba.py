"""Run both pretrained generators on every member of a named DuckDB set."""
import json
import sys
from pathlib import Path

import duckdb
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'src/se/official/semambapp')]
from model.semambapp import SEMambapp
from model.stfts import mag_phase_stft, mag_phase_istft
from src.evaluation.audio.render import render_recipe
from src.se.models.semambapp.inference import build, _enhance_16k
from tests.numerics.reference.compare_semamba_step import _cfg

db = ROOT / 'data/db/semamba-voicebank-test-20261101-compat.duckdb'
out = Path(sys.argv[1])
if out.exists():
    raise SystemExit('output already exists')
out.mkdir(parents=True)
con = duckdb.connect(str(db), read_only=True)
rows = con.execute("SELECT c.case_id, c.recipe_json::VARCHAR FROM eval.set_items i JOIN eval.cases c USING (case_id) WHERE i.set_name = ? ORDER BY i.position", ['voicebank_demand_official_test']).fetchall()
con.close()
if len(rows) != 824:
    raise SystemExit(f'unexpected test-set size: {len(rows)}')
device = torch.device('cuda')
weights = torch.load(ROOT / 'data/checkpoints/semambapp/semambapp.pth', map_location='cpu', weights_only=True)['generator']
official = SEMambapp(_cfg(48, 4, 16, 4)).to(device).eval()
independent = build(None).to(device).eval()
official.load_state_dict(weights, strict=True)
independent.load_state_dict(weights, strict=True)
with torch.inference_mode():
    for index, (case_id, recipe) in enumerate(rows):
        start, stride, shard = map(int, sys.argv[2:5]) if len(sys.argv) == 5 else (0, 1, 0)
        if index < start or (index - start) % stride != shard:
            continue
        rendered = render_recipe(json.loads(recipe), db.parent)
        clean, noisy = rendered.reference, rendered.observed
        x = noisy.to(device)
        peak = x.abs().max()
        x = x / (peak + 1e-9)
        mag, phase, _ = mag_phase_stft(x[None], 400, 100, 400, 0.3)
        am, ap, _ = official(mag, phase)
        a = (mag_phase_istft(am, ap, 400, 100, 400, 0.3)[0] * peak).cpu()
        b = (_enhance_16k(independent, x) * peak).cpu()
        torch.save({'case_id': case_id, 'sample_rate': 16000, 'clean': clean, 'noisy': noisy, 'official': a, 'independent': b, 'frames': mag.shape[-1]}, out / f'{index:04d}.pt')
        print(index, case_id, noisy.numel(), a.numel(), b.numel(), flush=True)
