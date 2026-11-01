"""Score disjoint index shards as inference finishes; each record is persistent."""
import json
import sys
import time
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tests.training.reference.semamba_quality import score_conditions

part, out = int(sys.argv[1]), Path(sys.argv[2])
if part not in range(3) or out.exists(): raise SystemExit('invalid shard or output exists')
out.mkdir(parents=True)
sources = [ROOT / 'data/results/training/semamba-voicebank-inference-20261101-v2'] + [ROOT / f'data/results/training/semamba-voicebank-shard-{i}-20261101' for i in (1, 2, 3)]
for i in range(part, 824, 3):
    while True:
        candidates = [p / f'{i:04d}.pt' for p in sources if (p / f'{i:04d}.pt').is_file()]
        if len(candidates) > 1: raise RuntimeError(f'duplicate waveform {i}')
        if candidates:
            try:
                sample = torch.load(candidates[0], map_location='cpu', weights_only=True)
                break
            except (RuntimeError, EOFError):
                pass  # writer has created the file but not finished writing it
        time.sleep(1)
    waves = [sample[name] for name in ('clean', 'noisy', 'official', 'independent')]
    lengths = [int(x.numel()) for x in waves]
    n = min(lengths)
    result = score_conditions(*(x[:n] for x in waves), 16000)
    result.update(case_id=sample['case_id'], original_lengths=lengths, common_prefix_samples=n)
    (out / f'{i:04d}.json').write_text(json.dumps(result))
    print(i, flush=True)
