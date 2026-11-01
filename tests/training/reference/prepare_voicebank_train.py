"""Prepare paired 28-speaker VoiceBank train with speaker-held-out validation."""
import json
import sys
from pathlib import Path
import torchaudio

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.data.recipe import canonical_recipe, case_content_hash, case_id_for, sha256_file
from src.data.training_sets import write_staging
from src.data.staging import rebuild_database

corpus = ROOT / 'data/corpora/VoiceBank-DEMAND'
stage = ROOT / 'data/staging/semamba-voicebank-train-20261101'
db = ROOT / 'data/db/semamba-voicebank-train-20261101.duckdb'
if stage.exists() or db.exists(): raise SystemExit('refusing to overwrite existing output')
clean = corpus / 'clean_trainset_28spk_wav'
noisy = corpus / 'noisy_trainset_28spk_wav'
names = sorted(p.name for p in clean.glob('*.wav'))
if len(names) != 11572 or set(names) != {p.name for p in noisy.glob('*.wav')}:
    raise SystemExit('unexpected files or unmatched pairs')
test_speakers = {p.name.split('_')[0] for p in (corpus / 'clean_testset_wav').glob('*.wav')}
valid_speakers = {'p286', 'p287'}
if test_speakers & {n.split('_')[0] for n in names}: raise SystemExit('train/test speaker overlap')
cases, roles = [], {'train': [], 'valid': []}
for index, name in enumerate(names):
    speaker = name.split('_')[0]
    role = 'valid' if speaker in valid_speakers else 'train'
    a, b = clean / name, noisy / name
    x, y = torchaudio.info(a), torchaudio.info(b)
    if (x.num_frames, x.sample_rate, x.num_channels) != (y.num_frames, y.sample_rate, y.num_channels): raise SystemExit(f'shape mismatch: {name}')
    sources = {kind: {'path': f'../corpora/VoiceBank-DEMAND/{p.parent.name}/{name}', 'sha256': sha256_file(p)} for kind, p in [('clean', a), ('noisy', b)]}
    recipe = canonical_recipe({'recipe_version': 1, 'sources': sources, 'operations': [], 'output': {'sample_rate': 16000, 'reference': 'dry_clean'}})
    metadata = {'corpus': 'VoiceBank-DEMAND', 'original_filename': name, 'pairing': 'official matching filenames', 'validation_speakers': sorted(valid_speakers)}
    body = {'kind': 'paired', 'dataset': 'VoiceBank-DEMAND', 'split': role, 'sample_rate': 16000, 'transcript': None, 'speaker_id': speaker, 'recipe': recipe, 'metadata': metadata}
    item = {'case_id': case_id_for(recipe), 'recipe_json': json.dumps(recipe, sort_keys=True), 'content_hash': case_content_hash(body), 'metadata': json.dumps(metadata, sort_keys=True), **{k: body[k] for k in ('kind', 'dataset', 'split', 'sample_rate', 'transcript', 'speaker_id')}}
    cases.append(item); roles[role].append(item['case_id'])
    if index % 500 == 0: print(index, flush=True)
sets = [(f'voicebank_28_{role}', f'VoiceBank official 28-speaker train split, {role}; validation speakers p286 and p287.', ids) for role, ids in roles.items()]
write_staging(stage, cases, sets)
print(json.dumps({'train': len(roles['train']), 'valid': len(roles['valid']), 'test_speakers': sorted(test_speakers), 'registered': rebuild_database(stage, db)}, indent=2))
